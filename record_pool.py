"""Record the pool camera to disk at full quality, with no GPUs and no AI.

The camera's own H.264 stream is saved as is (no re-encoding, so nothing is
lost): 2560x1440 at the camera's frame rate, in 10-minute files
raw_footage/pool_YYYYMMDD_HHMMSS.mkv. Those files show up in the dashboard's
"Choose clip" list, so they can be played through Spottr later.

    python record_pool.py              # records until you press Ctrl+C
    python record_pool.py --minutes 30 # stops by itself after 30 minutes

The camera address comes from SPOTTR_CAMERA in .env; if it is not there you
are asked for the camera account (the password is typed hidden, never shown
or saved). Video only: no audio (people's voices stay out of the files).
"""

import argparse
import getpass
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "raw_footage"
DEFAULT_IP = "192.168.137.179"


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def camera_url():
    url = os.environ.get("SPOTTR_CAMERA", "").strip()
    if url:
        return url
    print("Camera account (set in the Tapo app: Advanced Settings > Camera Account)")
    ip = input(f"Camera IP [{DEFAULT_IP}]: ").strip() or DEFAULT_IP
    user = input("Username: ").strip()
    pw = getpass.getpass("Password (hidden): ")
    return f"rtsp://{quote(user, safe='')}:{quote(pw, safe='')}@{ip}:554/stream1"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=0, help="stop after this many minutes (0 = until Ctrl+C)")
    ap.add_argument("--segment", type=int, default=10, help="minutes per file (default 10)")
    args = ap.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg not found. Install it with:  winget install ffmpeg")
    load_env()
    url = camera_url()
    OUT.mkdir(exist_ok=True)

    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-stats",
           "-rtsp_transport", "tcp", "-i", url,
           "-map", "0:v", "-c", "copy",               # the camera's video, untouched
           "-f", "segment", "-segment_time", str(args.segment * 60),
           "-reset_timestamps", "1", "-strftime", "1",
           str(OUT / "pool_%Y%m%d_%H%M%S.mkv")]
    if args.minutes:
        cmd[cmd.index("-map"):cmd.index("-map")] = ["-t", str(int(args.minutes * 60))]

    print(f"Recording to {OUT}\\pool_<date>_<time>.mkv  ({args.segment}-minute files)")
    print("Press Ctrl+C to stop.\n")
    t0 = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        p.wait()
    except KeyboardInterrupt:
        try:
            p.stdin.write(b"q"); p.stdin.flush()   # let ffmpeg close the file cleanly
        except OSError:
            pass
        p.wait(15)
    mins = (time.time() - t0) / 60
    files = sorted((f for f in OUT.glob("pool_*.mkv") if f.stat().st_mtime >= t0 - 1),
                   key=lambda f: f.stat().st_mtime)
    print(f"\nStopped after {mins:.1f} min. Files:")
    for f in files:
        print(f"  {f.relative_to(ROOT)}  ({f.stat().st_size / 1e6:.0f} MB)")
    if not files:
        print("ffmpeg could not record: check the camera is on, the laptop hotspot is on, "
              "and the username/password are right.")


if __name__ == "__main__":
    main()
