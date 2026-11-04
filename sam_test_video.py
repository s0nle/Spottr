"""
Spottr - SAM 3.1 VIDEO tracking test on Modal.

Tracks every "person" through a video clip with SAM 3.1's multi-object video
tracker, then gives you back:
  - results/<name>_tracked.mp4  the clip with a box + persistent ID on each person
  - results/<name>_tracks.json  every person's position, size and score per frame
                                (this is the data the rules engine will read later)
  - a summary in the terminal: how many IDs, people per frame, and every time a
    person disappeared and came back under the SAME ID (the underwater test)

Run from the VS Code terminal (with your .venv active):
    modal run sam_test_video.py --path pool_clip.mp4
    modal run sam_test_video.py --path pool_clip.mp4 --seconds 30 --threshold 0.4
    modal run sam_test_video.py --path pool_clip.mp4 --prompt "person,head" --threshold 0.3
    modal run sam_test_video.py --path pool_clip.mp4 --fps 15 --max-objects 96

Needs the same setup as sam_test.py:
    - A Modal Secret named "huggingface-secret" containing HF_TOKEN
    - Approved access to facebook/sam3.1 on Hugging Face
    - The weights are shared with sam_test.py through the "sam31-weights" Volume
"""

import json
from pathlib import Path

import modal

app = modal.App("spottr-sam31-video-test")

# Defaults for the CLI flags --max-objects / --multiplex-count (see main).
# How many people the tracker can hold at once. SAM 3.1 processes them in
# buckets of 16, so 64 = 4 buckets. Raise it for very crowded pools.
MAX_OBJECTS = 64
# Objects processed per multiplex bucket. 16 matches Meta's default.
MULTIPLEX_COUNT = 16

# The video tracker runs its detector on a batch of frames at once, which does
# not fit in the A10G's 22 GB. L40S (48 GB) is the cheapest GPU with room.
# Use "H100" for the real speed test.
GPU = "L40S"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    # Reduces memory wasted to fragmentation (suggested by the OOM error)
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "huggingface_hub", "pillow", "numpy", "einops", "opencv-python-headless"
    )
    .run_commands(
        'pip install "sam3[notebooks] @ git+https://github.com/facebookresearch/sam3.git"'
    )
)

weights = modal.Volume.from_name("sam31-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"
CHECKPOINT = "sam3.1_multiplex.pt"


def _np(x):
    """Turns a tensor / array / list from SAM into a numpy array."""
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    if x is None:
        return np.zeros((0,))
    return np.asarray(x)


def _drop_offload_kwarg(obj):
    """Known SAM 3.1 bug (Meta issue #544): the multiplex tracker rejects an
    'offload_state_to_cpu' argument the base predictor passes when a session
    starts. This wraps init_state so that one argument is dropped."""
    import functools

    fn = getattr(obj, "init_state", None)
    if not callable(fn) or getattr(fn, "_spottr_patched", False):
        return

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        kwargs.pop("offload_state_to_cpu", None)
        return fn(*args, **kwargs)

    wrapped._spottr_patched = True
    setattr(obj, "init_state", wrapped)


@app.cls(
    gpu=GPU,
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={WEIGHTS_DIR: weights},
    timeout=1800,  # hard stop after 30 min so a stuck run can't burn credits
    scaledown_window=60,
)
class Sam31Video:
    @modal.enter()
    def load(self):
        import os

        from huggingface_hub import hf_hub_download

        ckpt_path = os.path.join(WEIGHTS_DIR, CHECKPOINT)
        if not os.path.exists(ckpt_path):
            print("Downloading SAM 3.1 weights (first run only, ~3.5 GB)...")
            hf_hub_download(
                repo_id="facebook/sam3.1", filename=CHECKPOINT, local_dir=WEIGHTS_DIR
            )
            weights.commit()

        # The predictor is built lazily in track(), so --max-objects and
        # --multiplex-count (which arrive per call) can take effect. A call
        # with different values rebuilds once, then reuses the new build.
        self.predictor = None
        self._build_cfg = None
        print("Weights ready, predictor builds on first track() call")

    def _build_predictor(self, max_objects: int, multiplex_count: int):
        import inspect
        import os

        from sam3.model_builder import build_sam3_multiplex_video_predictor

        ckpt_path = os.path.join(WEIGHTS_DIR, CHECKPOINT)
        sig = inspect.signature(build_sam3_multiplex_video_predictor)
        print("build_sam3_multiplex_video_predictor", sig)
        wanted = dict(
            checkpoint_path=ckpt_path,
            load_from_HF=False,
            max_num_objects=max_objects,
            multiplex_count=multiplex_count,
            use_fa3=False,
            compile=False,
            warm_up=False,
        )
        # Only pass the settings this version of Meta's code actually accepts
        kwargs = {k: v for k, v in wanted.items() if k in sig.parameters}
        print("Building with:", {k: v for k, v in kwargs.items() if k != "checkpoint_path"})
        self.predictor = build_sam3_multiplex_video_predictor(**kwargs)

        _drop_offload_kwarg(self.predictor)
        _drop_offload_kwarg(getattr(self.predictor, "model", None))

        # Print any setting that controls how many frames the detector batches
        # together. Those are the knobs for fitting a smaller GPU later.
        model = getattr(self.predictor, "model", None)
        for owner_name, owner in (
            ("model", model),
            ("detector", getattr(model, "detector", None)),
        ):
            if owner is None:
                continue
            for name in dir(owner):
                if name.startswith("__"):
                    continue
                if any(w in name.lower() for w in ("chunk", "batch")):
                    try:
                        val = getattr(owner, name)
                    except Exception:  # noqa: BLE001
                        continue
                    if isinstance(val, (int, float, bool, str)):
                        print(f"  {owner_name}.{name} = {val}")
        print("Loaded SAM 3.1 multiplex video tracker")

    @modal.method()
    def track(
        self,
        video_bytes: bytes,
        prompt: str,
        threshold: float,
        seconds: float,
        max_width: int,
        fps_cap: float = 0.0,
        max_objects: int = MAX_OBJECTS,
        multiplex_count: int = MULTIPLEX_COUNT,
    ) -> dict:
        import subprocess
        import time

        import cv2
        import numpy as np
        import torch

        # --- 1. Trim + downscale (+ optional fps cap) so memory stays reasonable
        vf = f"scale='min({max_width},iw)':-2"
        if fps_cap and fps_cap > 0:
            vf += f",fps={fps_cap}"
        with open("/tmp/in.mp4", "wb") as f:
            f.write(video_bytes)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", "/tmp/in.mp4",
                "-t", str(seconds),
                "-vf", vf,
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "/tmp/clip.mp4",
            ],
            check=True,
        )
        cap = cv2.VideoCapture("/tmp/clip.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"Clip: {W}x{H}, {fps:.1f} fps, {n_frames} frames")

        # --- 2. Track with SAM 3.1: one pass per prompt -----------------------
        # SAM 3.1 tracks one text prompt per session, so "person,head" runs two
        # passes over the same clip. Each object is tagged with its prompt.
        prompts = [s.strip() for s in prompt.split(",") if s.strip()]
        if not prompts:
            raise ValueError("prompt is empty (need e.g. 'person' or 'person,head')")
        # Build (or rebuild) the predictor for the requested capacity. The
        # first call pays the build cost; later calls with the same values reuse it.
        cfg = (int(max_objects), int(multiplex_count))
        if self.predictor is None or self._build_cfg != cfg:
            print(f"Building predictor: max_num_objects={cfg[0]}, multiplex_count={cfg[1]}")
            self._build_predictor(*cfg)
            self._build_cfg = cfg
        frames = {}  # frame index -> list of objects from every prompt
        track_times = {}
        p = self.predictor
        for pr in prompts:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                session_id = p.handle_request(
                    {"type": "start_session", "resource_path": "/tmp/clip.mp4"}
                )["session_id"]
                try:
                    p.handle_request({
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": pr,
                        "output_prob_thresh": threshold,
                    })
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    n_done = 0
                    for resp in p.handle_stream_request({
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "forward",
                        "output_prob_thresh": threshold,
                    }):
                        fi = int(resp["frame_index"])
                        out = resp["outputs"]
                        ids = _np(out.get("out_obj_ids")).astype(int).reshape(-1)
                        probs = _np(out.get("out_probs")).reshape(-1)
                        boxes = _np(out.get("out_boxes_xywh"))
                        boxes = boxes.reshape(-1, 4) if boxes.size else np.zeros((0, 4))
                        masks = _np(out.get("out_binary_masks"))
                        if masks.ndim == 4:
                            masks = masks[:, 0]
                        n = min(len(ids), len(probs), len(boxes))
                        if masks.ndim == 3:
                            n = min(n, len(masks))

                        objs = frames.setdefault(fi, [])
                        for k in range(n):
                            x, y, w, h = (float(v) for v in boxes[k])
                            cx, cy, area = x + w / 2, y + h / 2, w * h
                            if masks.ndim == 3:
                                ys, xs = np.nonzero(masks[k] > 0.5)
                                mh, mw = masks[k].shape
                                area = float(len(xs)) / (mh * mw)
                                if len(xs):
                                    cx, cy = float(xs.mean()) / mw, float(ys.mean()) / mh
                            objs.append({
                                "prompt": pr,
                                "id": int(ids[k]),
                                "p": round(float(probs[k]), 3),
                                "box": [round(x, 4), round(y, 4), round(w, 4), round(h, 4)],
                                "cx": round(cx, 4),   # centre, 0-1 across the frame
                                "cy": round(cy, 4),
                                "area": round(area, 6),  # share of the frame the outline covers
                            })
                        n_done += 1
                    torch.cuda.synchronize()
                    track_times[pr] = round(time.perf_counter() - start, 2)
                finally:
                    p.handle_request({"type": "close_session", "session_id": session_id})
            torch.cuda.empty_cache()
            print(f"'{pr}': tracked {n_done} frames in {track_times[pr]:.1f} s "
                  f"({n_done / max(track_times[pr], 1e-6):.1f} frames/s)")
        track_s = sum(track_times.values())

        # --- 3. Draw boxes + IDs and encode a viewable mp4 --------------------
        # person = thin box, other prompts (head) = thick box; label shows both
        cap = cv2.VideoCapture("/tmp/clip.mp4")
        writer = cv2.VideoWriter(
            "/tmp/raw.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H)
        )
        fi = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            objs = frames.get(fi, [])
            for o in objs:
                pi = prompts.index(o["prompt"])
                hue = (o["id"] * 47 + pi * 90) % 180
                color = cv2.cvtColor(
                    np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR
                )[0, 0].tolist()
                x, y, w, h = o["box"]
                p1 = (int(x * W), int(y * H))
                p2 = (int((x + w) * W), int((y + h) * H))
                cv2.rectangle(img, p1, p2, color, 2 if pi == 0 else 3)
                cv2.putText(img, f"{o['prompt']} {o['id']} {o['p']:.2f}",
                            (p1[0], max(14, p1[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            counts = "  ".join(
                f"{pr}: {sum(1 for o in objs if o['prompt'] == pr)}" for pr in prompts
            )
            cv2.rectangle(img, (0, 0), (W, 30), (0, 0, 0), -1)
            cv2.putText(img, f"frame {fi}  t={fi / fps:5.1f}s  {counts}",
                        (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            writer.write(img)
            fi += 1
        cap.release()
        writer.release()
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", "/tmp/raw.avi",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p", "/tmp/tracked.mp4"],
            check=True,
        )
        with open("/tmp/tracked.mp4", "rb") as f:
            video_out = f.read()

        return {
            "fps": fps,
            "width": W,
            "height": H,
            "prompts": prompts,
            "settings": {
                "threshold": threshold,
                "seconds": seconds,
                "max_width": max_width,
                "fps_cap": fps_cap,
                "max_objects": int(max_objects),
                "multiplex_count": int(multiplex_count),
            },
            "frames": [{"f": k, "objs": frames[k]} for k in sorted(frames)],
            "track_seconds": round(track_s, 2),
            "track_seconds_per_prompt": track_times,
            "video": video_out,
        }


def summarize(result: dict, min_gap_s: float = 0.3) -> None:
    """Prints per-prompt stats and every disappear/reappear per ID."""
    fps = result["fps"]
    frames = result["frames"]
    n = len(frames)
    print(f"\nFrames tracked: {n}  ({n / fps:.1f} s of video, video is {fps:.0f} fps)")

    for pr in result.get("prompts", ["person"]):
        secs = result.get("track_seconds_per_prompt", {}).get(pr, result["track_seconds"])
        counts = [sum(1 for o in fr["objs"] if o.get("prompt", pr) == pr) for fr in frames]
        seen = {}
        for fr in frames:
            for o in fr["objs"]:
                if o.get("prompt", pr) == pr:
                    seen.setdefault(o["id"], []).append(fr["f"])

        print(f"\n=== '{pr}' ===")
        print(f"Tracking speed: {n / max(secs, 1e-6):.1f} frames/s")
        if counts:
            print(f"Per frame: min {min(counts)}, max {max(counts)}, "
                  f"average {sum(counts) / len(counts):.1f}")
        print(f"Distinct IDs: {len(seen)}  "
              f"(more IDs than real people = the tracker swapped or re-created someone)")
        for pid in sorted(seen):
            fs = seen[pid]
            print(f"  {pr} {pid:<4} visible {len(fs):>5} frames, "
                  f"{fs[0] / fps:6.1f}s -> {fs[-1] / fps:6.1f}s")
            for a, b in zip(fs, fs[1:]):
                if (b - a - 1) / fps >= min_gap_s:
                    print(f"           gone {a / fps:.1f}s-{b / fps:.1f}s "
                          f"({(b - a - 1) / fps:.1f}s), came back as same ID")

    print("\nUnderwater test: an ID that has a 'gone ... came back as same ID' line "
          "kept its identity while hidden. A person who comes back with a NEW ID "
          "means the tracker lost them.")


@app.local_entrypoint()
def main(
    path: str,
    prompt: str = "person",
    threshold: float = 0.5,
    seconds: float = 15.0,
    max_width: int = 1280,
    fps: float = 0.0,
    max_objects: int = MAX_OBJECTS,
    multiplex_count: int = MULTIPLEX_COUNT,
    min_gap: float = 0.3,
):
    """Runs on your laptop: sends the clip to the GPU and saves the results.

    prompt     one prompt, or several separated by commas: "person,head"
               (each prompt is its own tracking pass, so two prompts take ~2x as long)
    seconds    only the first N seconds are tracked (keeps GPU memory and cost down)
    max_width  the clip is shrunk to at most this many pixels wide before tracking
    fps        cap the clip to this many frames/s before tracking (0 = keep source
               fps; 10-15 is usually enough and roughly halves GPU time)
    max_objects
               how many people the tracker can hold at once (default 64)
    multiplex_count
               objects processed per multiplex bucket (default 16)
    min_gap    gaps shorter than this (seconds) are not reported as
               disappear/reappear in the summary
    """
    p = Path(path)
    if not p.exists():
        print(f"No video found at {p}")
        return

    print(f"Sending {p.name} ({p.stat().st_size / 1e6:.1f} MB) to Modal...")
    result = Sam31Video().track.remote(
        p.read_bytes(), prompt, threshold, seconds, max_width,
        fps, max_objects, multiplex_count,
    )

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    video_path = out_dir / f"{p.stem}_tracked.mp4"
    video_path.write_bytes(result.pop("video"))
    json_path = out_dir / f"{p.stem}_tracks.json"
    json_path.write_text(json.dumps(result, indent=1))

    summarize(result, min_gap_s=min_gap)
    print(f"\nSaved {video_path}")
    print(f"Saved {json_path}")
