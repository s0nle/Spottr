"""Regression check for the tracker: replays saved live logs and compares the
pool count (every 10 s), entries and exits with the saved known-good result.

    python diag/regress.py            # check (exit code 1 on any change)
    python diag/regress.py --update   # save the current result as known-good

Run it after every tracker change. A change is not always a bug, but it must
be looked at (and the baseline updated on purpose).
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spottr_live import Tracker  # noqa: E402

LOGS = ["results/live_log_20261102_232854.json",   # 4-min clip, person only
        "results/live_log_20261102_232056.json",   # first 80 s, person only
        "results/live_log_20261103_014709.json"]   # 4-min clip, person + head
BASELINE = ROOT / "diag" / "regress_baseline.json"

zone = np.array(json.load(open(ROOT / "pool_zone.json"))["points"], np.float32)


def in_pool(b):
    return cv2.pointPolygonTest(zone, ((b[0] + b[2]) / 2, min(b[3], 0.999)), False) >= 0


def run(log):
    frames = [f for f in json.load(open(ROOT / log))["frames"] if "raw" in f]
    T = Tracker(in_pool)
    per10 = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for f in frames:
            T.update([dict(d) for d in f["raw"]], f["t"])
            c = T.counts(f["t"])
            k = str(int(f["t"] // 10) * 10)
            per10[k] = max(per10.get(k, 0), c["pool"])
    return {"pool_max_per_10s": per10, "entries": T.entries, "exits": T.exits}


def main():
    now = {log: run(log) for log in LOGS}
    if "--update" in sys.argv:
        BASELINE.write_text(json.dumps(now, indent=1))
        print(f"baseline saved: {BASELINE}")
        return 0
    base = json.loads(BASELINE.read_text())
    bad = 0
    for log in LOGS:
        a, b = base.get(log), now[log]
        if a == b:
            print(f"OK    {log}")
            continue
        bad += 1
        print(f"DIFF  {log}")
        if a["entries"] != b["entries"] or a["exits"] != b["exits"]:
            print(f"      entries {a['entries']}->{b['entries']}  exits {a['exits']}->{b['exits']}")
        for k in sorted(set(a["pool_max_per_10s"]) | set(b["pool_max_per_10s"]), key=int):
            x, y = a["pool_max_per_10s"].get(k), b["pool_max_per_10s"].get(k)
            if x != y:
                print(f"      {k:>4}s  pool {x} -> {y}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
