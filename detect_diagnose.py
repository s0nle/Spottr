"""
Spottr - detection diagnosis. Why does SAM 3.1 miss swimmers?

Runs the SAME image model the live path uses (spottr_live.py) on raw HD frames
under several conditions, with a very low threshold so we also see the
"almost detected" scores:

  live   frame shrunk to 640x360 + JPEG quality 70 (what the live path sends)
  full   the raw 2560x1440 frame
  tiles  the raw frame cut into 2x2 overlapping tiles (= ~2x zoom), each run
         separately, boxes mapped back to the full frame

Prompts: "person", "swimmer", "head" (live + full), "person" (tiles).
Nothing is filtered or de-duplicated here, so the analysis can test the
threshold and the live de-duplication rule afterwards.

Run (with .venv active):
    modal run detect_diagnose.py
    modal run detect_diagnose.py --frames diag/frames --gpus 4

Saves diag/detections.json
"""

import json
from pathlib import Path

import modal

app = modal.App("spottr-detect-diagnose")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("huggingface_hub", "pillow", "numpy", "einops")
    .run_commands(
        'pip install "sam3[notebooks] @ git+https://github.com/facebookresearch/sam3.git"'
    )
)
weights = modal.Volume.from_name("sam31-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"
CHECKPOINT = "sam3.1_multiplex.pt"
LOW_THRESHOLD = 0.05  # report almost everything; the analysis applies real thresholds


@app.cls(
    gpu="L40S",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={WEIGHTS_DIR: weights},
    timeout=1800,
    scaledown_window=30,
)
class Diagnose:
    @modal.enter()
    def load(self):
        import os

        from huggingface_hub import hf_hub_download
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        ckpt = os.path.join(WEIGHTS_DIR, CHECKPOINT)
        if not os.path.exists(ckpt):
            hf_hub_download(repo_id="facebook/sam3.1", filename=CHECKPOINT,
                            local_dir=WEIGHTS_DIR)
            weights.commit()
        # "sam31": exactly what spottr_live.py loads (the video checkpoint
        # put into the image model -> reports missing backbone weights)
        self.procs = {}
        m31 = build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=False)
        self.procs["sam31"] = Sam3Processor(m31, confidence_threshold=LOW_THRESHOLD)
        # "sam3": Meta's official SAM 3 IMAGE weights (facebook/sam3), which
        # match the image model. Skipped if this account can't download them.
        try:
            m3 = build_sam3_image_model()
            self.procs["sam3"] = Sam3Processor(m3, confidence_threshold=LOW_THRESHOLD)
            print("Loaded facebook/sam3 image weights too")
        except Exception as e:  # noqa: BLE001
            print(f"facebook/sam3 image weights not available: {e!r}")
        print("Processor resolution:", getattr(self.procs["sam31"], "resolution", "?"))

    def _run(self, img, prompts, model="sam31"):
        """All detections for one image, boxes as fractions of that image."""
        import torch

        W, H = img.size
        out_all = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            proc = self.procs[model]
            state = proc.set_image(img)
            for pr in prompts:
                out = proc.set_text_prompt(state=state, prompt=pr)
                boxes = out["boxes"].float().cpu().tolist()
                scores = out["scores"].float().cpu().tolist()
                if scores:
                    masks = out["masks"].reshape(len(scores), -1).float() > 0.5
                    areas = masks.float().mean(1).cpu().tolist()
                else:
                    areas = []
                out_all[pr] = [
                    {"s": round(s, 3),
                     "box": [round(b[0] / W, 4), round(b[1] / H, 4),
                             round(b[2] / W, 4), round(b[3] / H, 4)],
                     "mask_area": round(a, 6)}
                    for b, s, a in zip(boxes, scores, areas)
                ]
        return out_all

    @modal.method()
    def frames(self, batch: list) -> list:
        import io

        from PIL import Image

        results = []
        for name, jpeg in batch:
            full = Image.open(io.BytesIO(jpeg)).convert("RGB")
            W, H = full.size

            # live: shrink to 640 wide, re-encode as JPEG quality 70
            small = full.resize((640, round(H * 640 / W)), Image.BILINEAR)
            buf = io.BytesIO()
            small.save(buf, "JPEG", quality=70)
            live = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

            r = {"name": name, "w": W, "h": H}
            for model in self.procs:
                r[f"{model}_live"] = self._run(live, ["person", "swimmer", "head"], model)
                r[f"{model}_full"] = self._run(full, ["person", "swimmer", "head"], model)

                # tiles: 2x2 with 15% overlap, mapped back to full-frame fractions
                tiles = []
                tw, th = int(W * 0.575), int(H * 0.575)
                for ty in (0, H - th):
                    for tx in (0, W - tw):
                        dets = self._run(full.crop((tx, ty, tx + tw, ty + th)),
                                         ["person"], model)
                        for d in dets["person"]:
                            x1, y1, x2, y2 = d["box"]
                            d["box"] = [round((tx + x1 * tw) / W, 4),
                                        round((ty + y1 * th) / H, 4),
                                        round((tx + x2 * tw) / W, 4),
                                        round((ty + y2 * th) / H, 4)]
                            d["tile"] = [tx, ty]
                            tiles.append(d)
                r[f"{model}_tiles"] = {"person": tiles}
            results.append(r)
        return results


@app.local_entrypoint()
def main(frames: str = "diag/frames", gpus: int = 4, batch_size: int = 25):
    files = sorted(Path(frames).glob("*.jpg"))
    if not files:
        print(f"No frames in {frames}")
        return
    batches = [[(f.name, f.read_bytes()) for f in files[i:i + batch_size]]
               for i in range(0, len(files), batch_size)]
    print(f"{len(files)} frames in {len(batches)} batches on up to {gpus} L40S GPUs...")
    runner = Diagnose.with_options(max_containers=gpus)()
    results = []
    for i, res in enumerate(runner.frames.map(batches, return_exceptions=True)):
        if isinstance(res, BaseException):
            print(f"  batch {i}: FAILED {res!r}")
            continue
        results.extend(res)
        print(f"  batch {i + 1}/{len(batches)} done")
    out = Path("diag/detections.json")
    out.write_text(json.dumps(results))
    print(f"Saved {out} ({len(results)} frames)")
