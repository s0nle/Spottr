"""Re-run Spottr's tracker on a saved live log, offline (no GPU, seconds).

The live client saves every raw detection per check ("raw" in the log), so
tracker settings can be tuned here and compared without another GPU run.

    python diag/replay_tracker.py results/live_log_XXXX.json
    python diag/replay_tracker.py results/live_log_XXXX.json --confirm-score 0.7 --keep-pool 30
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spottr_live import Tracker  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--zone", default="pool_zone.json")
ap.add_argument("--create-pool", type=float, default=0.35)
ap.add_argument("--create-deck", type=float, default=0.50)
ap.add_argument("--sustain-pool", type=float, default=0.20)
ap.add_argument("--sustain-deck", type=float, default=0.35)
ap.add_argument("--confirm-score", type=float, default=0.60)
ap.add_argument("--keep-pool", type=float, default=60.0)
ap.add_argument("--keep-deck", type=float, default=8.0)
ap.add_argument("--quiet", action="store_true", help="hide BACK events")
a = ap.parse_args()

z = np.array(json.load(open(a.zone))["points"], np.float32)


def in_pool(b):
    return cv2.pointPolygonTest(z, ((b[0] + b[2]) / 2, min(b[3], 0.999)), False) >= 0


frames = [f for f in json.load(open(a.log))["frames"] if "raw" in f]
if not frames:
    sys.exit("This log has no raw detections (it was made before raw logging was added).")
T = Tracker(in_pool, create_pool=a.create_pool, create_deck=a.create_deck,
            sustain_pool=a.sustain_pool, sustain_deck=a.sustain_deck,
            keep_pool=a.keep_pool, keep_deck=a.keep_deck, confirm_score=a.confirm_score)
per10 = collections.defaultdict(list)
for f in frames:
    dets = [dict(d) for d in f["raw"]]
    T.update(dets, f["t"])
    per10[int(f["t"] // 10)].append(T.counts(f["t"]))

print(f"{len(frames)} checks, {T.next_id} IDs started, "
      f"entered {T.entries}x, exited {T.exits}x")
for e in T.events:
    if a.quiet and e["event"] == "BACK":
        continue
    print(f"  {e['t']:7.1f}s  {e['event']:6s} person {e['id']}: {e['text']}")
print("\nevery 10 s: in pool (verified) max (avg) / under / LOST / unverified / deck")
for k, cs in sorted(per10.items()):
    print(f"  {k * 10:4d}s  pool {max(c['pool'] for c in cs)} "
          f"({sum(c['pool'] for c in cs) / len(cs):.1f})  under {max(c['under'] for c in cs)}"
          f"  LOST {max(c.get('lost', 0) for c in cs)}"
          f"  unverified {max(c.get('unverified', 0) for c in cs)}"
          f"  deck {max(c['deck'] for c in cs)}")
