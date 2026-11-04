"""
Spottr - track FULL raw recordings with SAM 3.1 on Modal H100s.

The video tracker keeps every frame of a clip in GPU memory, so a 10-minute
file can't go in one piece. This script takes the 60-second chunks in
raw_footage/chunks/ (cut with ffmpeg at keyframes, no gaps), tracks them all
in parallel on H100s with the same tracker as sam_test_video.py, then stitches
the results back into one timeline per recording.

Run (with .venv active):
    modal run sam_track_full.py::full
    modal run sam_track_full.py::full --fps 10 --max-width 1280 --max-gpus 10

Saves, per recording (e.g. pool_20261102_091918):
    results/full/<name>_tracks.json   every detection, time in seconds from the
                                      start of the recording
    results/full/<name>_tracked.mp4   all chunks' tracked videos joined together

IDs restart in each chunk, so they are saved as "<chunk>-<id>" (e.g. "03-2").
"""

import json
import subprocess
from pathlib import Path

from sam_test_video import Sam31Video, app

CHUNK_DIR = Path("raw_footage/chunks")
OUT_DIR = Path("results/full")
MIN_CHUNK_S = 5.0  # skip tiny leftover pieces at the end of a recording


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


@app.local_entrypoint()
def full(
    prompt: str = "person",
    threshold: float = 0.5,
    max_width: int = 1280,
    fps: float = 10.0,
    max_gpus: int = 10,
):
    chunks = sorted(CHUNK_DIR.glob("*.mkv"))
    jobs = []
    for c in chunks:
        d = duration(c)
        if d < MIN_CHUNK_S:
            print(f"skip {c.name} ({d:.1f} s)")
            continue
        rec, idx = c.stem.rsplit("_", 1)
        jobs.append({"path": c, "rec": rec, "idx": int(idx), "dur": d})
    if not jobs:
        print(f"No chunks in {CHUNK_DIR}")
        return

    # Start time of each chunk inside its recording
    starts = {}
    for j in sorted(jobs, key=lambda j: (j["rec"], j["idx"])):
        j["start"] = starts.get(j["rec"], 0.0)
        starts[j["rec"]] = j["start"] + j["dur"]

    print(f"Tracking {len(jobs)} chunks on up to {max_gpus} H100s in parallel...")
    tracker = Sam31Video.with_options(gpu="H100", max_containers=max_gpus)()
    args = [
        (j["path"].read_bytes(), prompt, threshold, j["dur"] + 1, max_width, fps)
        for j in jobs
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_rec = {}
    for j, res in zip(jobs, tracker.track.starmap(args, return_exceptions=True)):
        if isinstance(res, BaseException):
            print(f"  {j['path'].name}: FAILED {res!r}")
            j["error"] = repr(res)
            per_rec.setdefault(j["rec"], []).append((j, None))
            continue
        vid = OUT_DIR / f"{j['path'].stem}_tracked.mp4"
        vid.write_bytes(res.pop("video"))
        n_ids = len({o["id"] for fr in res["frames"] for o in fr["objs"]})
        print(f"  {j['path'].name}: {len(res['frames'])} frames, {n_ids} IDs, "
              f"tracked in {res['track_seconds']:.0f} s")
        per_rec.setdefault(j["rec"], []).append((j, res))

    for rec, items in per_rec.items():
        items.sort(key=lambda x: x[0]["idx"])
        frames, failed, videos = [], [], []
        for j, res in items:
            if res is None:
                failed.append({"chunk": j["idx"], "start_s": round(j["start"], 2),
                               "error": j["error"]})
                continue
            videos.append(OUT_DIR / f"{j['path'].stem}_tracked.mp4")
            for fr in res["frames"]:
                t = j["start"] + fr["f"] / res["fps"]
                objs = [dict(o, id=f"{j['idx']:02d}-{o['id']}") for o in fr["objs"]]
                frames.append({"t": round(t, 3), "chunk": j["idx"], "objs": objs})

        first = next((r for _, r in items if r is not None), {})
        out = {
            "recording": rec,
            "prompts": first.get("prompts", [prompt]),
            "settings": first.get("settings", {}),
            "chunks": [{"chunk": j["idx"], "start_s": round(j["start"], 2),
                        "dur_s": round(j["dur"], 2)} for j, _ in items],
            "failed": failed,
            "frames": frames,
        }
        json_path = OUT_DIR / f"{rec}_tracks.json"
        json_path.write_text(json.dumps(out))

        # Join the chunk videos (same encoder settings, so no re-encode needed)
        if videos:
            lst = OUT_DIR / f"{rec}_list.txt"
            lst.write_text("".join(f"file '{v.resolve().as_posix()}'\n" for v in videos))
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                 "-i", str(lst), "-c", "copy", str(OUT_DIR / f"{rec}_tracked.mp4")],
                check=True,
            )
            lst.unlink()
        print(f"\n{rec}: {len(frames)} frames saved -> {json_path}"
              + (f"  ({len(failed)} chunks failed)" if failed else ""))
