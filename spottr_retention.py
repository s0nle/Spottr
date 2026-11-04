"""Footage retention for Spottr: delete recordings older than N days (3 for
the demo). Runs automatically when the live client starts.

    python spottr_retention.py          # show what WOULD be deleted
    python spottr_retention.py --apply  # delete it

Covers raw recordings (raw_footage/*.mkv, raw_footage/chunks/), live logs
and videos (results/live_log_*.json, results/*.mp4, Muse reports) and alarm
snapshot folders (results/incidents/<session>/).

Never deletes:
  - anything listed in diag/keep.txt (the regression-test logs and the test
    footage they come from)
  - raw_footage/clips/ (hand-cut test clips)
"""

import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PATTERNS = ["raw_footage/*.mkv", "raw_footage/*.mp4", "raw_footage/chunks/*",
            "results/live_log_*.json", "results/*_report.md", "results/*.mp4",
            "results/incidents/*"]


def _keep():
    p = ROOT / "diag" / "keep.txt"
    keep = set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                keep.add((ROOT / line).resolve())
    return keep


def old_files(days=3.0):
    cutoff = time.time() - days * 86400
    keep = _keep()
    clips = (ROOT / "raw_footage" / "clips").resolve()
    out = []
    for pat in PATTERNS:
        for f in ROOT.glob(pat):
            r = f.resolve()
            if r in keep or clips in r.parents or r == clips:
                continue
            if f.stat().st_mtime < cutoff:
                out.append(f)
    return out


def apply(days=3.0, dry=False):
    files = old_files(days)
    for f in files:
        if dry:
            print(f"  would delete {f.relative_to(ROOT)}")
            continue
        try:
            shutil.rmtree(f) if f.is_dir() else f.unlink()
        except OSError as e:
            print(f"  could not delete {f.name}: {e}")
    if files and not dry:
        print(f"Retention: deleted {len(files)} item(s) older than {days:g} days")
    return files


if __name__ == "__main__":
    dry = "--apply" not in sys.argv
    n = apply(3.0, dry=dry)
    print(f"{len(n)} item(s) older than 3 days" + (" (dry run, nothing deleted)" if dry else ""))
