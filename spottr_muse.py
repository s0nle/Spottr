"""Muse Spark (Meta Model API) for Spottr: a second opinion on a swimmer whose
timer is running, and a written report after a session.

Needs MODEL_API_KEY (from dev.meta.ai) in .env or the environment. Without
it every function returns None and Spottr runs normally.

    python spottr_muse.py report results/live_log_XXXX.json   # write a session report

API: OpenAI-compatible chat completions at https://api.meta.ai/v1, images as
base64 data URLs. Uses only the standard library (no extra install).
"""

import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.meta.ai/v1/chat/completions"
MODEL = "muse-spark-1.3"


def load_env(path=None):
    """Reads KEY=VALUE lines from .env into os.environ (never overrides)."""
    p = Path(path) if path else Path(__file__).resolve().parent / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


load_env()


def available():
    return bool(os.environ.get("MODEL_API_KEY"))


def _chat(content, max_tokens=4000, timeout=30, effort="low"):
    """One chat call. content = list of OpenAI-style parts. Returns text or None.
    Muse is a reasoning model: its thinking counts against max_tokens (a 300
    budget was used up by thinking and returned no answer), so the budget is
    large and effort is "low" for quick second opinions."""
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        return None
    body = json.dumps({"model": MODEL, "max_tokens": max_tokens, "reasoning_effort": effort,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(API, data=body, method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        return out["choices"][0]["message"]["content"]
    except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as e:
        # never print the key; the error text has no secrets in it
        print(f"Muse call failed: {type(e).__name__}: {str(e)[:200]}")
        return None


def _img(jpeg_bytes):
    return {"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()}}


SECOND_OPINION = """You are assisting a lifeguard. Image 1 is a close crop of ONE swimmer
(swimmer #{id}); image 2 is the whole pool for context. Spottr's camera
system flagged this swimmer: {reason}.

Look only at that swimmer. Answer with ONLY a JSON object:
{{"head_above_water": true/false/null,
  "posture": "swimming" | "resting at wall" | "standing" | "floating face down" |
             "vertical struggling" | "under water" | "unclear",
  "verdict": "ok" | "concern" | "emergency",
  "confidence": 0.0-1.0,
  "reason": "<one short sentence>"}}
"emergency" = face down and still, under water and not moving, or struggling
upright with head tilted back (instinctive drowning response).
"ok" only if you can clearly see the person is fine (head up and breathing,
swimming normally, resting at the wall). If unsure, say "concern"."""


def second_opinion(crop_jpeg, frame_jpeg, tid, reason):
    """Returns {"verdict", "reason", ...} or None (no key / failed / unparsable)."""
    text = _chat([{"type": "text", "text": SECOND_OPINION.format(id=tid, reason=reason)},
                  _img(crop_jpeg), _img(frame_jpeg)], max_tokens=3000)
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        out = json.loads(m.group(0)) if m else None
    except ValueError:
        out = None
    if not out or out.get("verdict") not in ("ok", "concern", "emergency"):
        return None
    return out


REPORT = """You are writing a short incident and session report for a pool
lifeguard team. Below is the event log of one Spottr session (an AI camera
that counts swimmers, times how long each is under water, and raises
alarms). {n_img} alarm snapshots follow, in the order listed.

Write the report in plain English, under 400 words, with these sections:
1. Summary (2-3 sentences: how long, how many swimmers, how many alarms)
2. Alarms (one line each: time, swimmer, level, why, and what the snapshot
   shows; say clearly if a snapshot looks like a false alarm)
3. Things to check (anything a lifeguard should review)
Do not invent events that are not in the log.

EVENT LOG (seconds since start):
{events}"""


def session_report(log_path, max_images=8):
    """Writes <log>_report.md next to the log. Returns its path or None."""
    if not available():
        print("No MODEL_API_KEY: skipping the Muse report.")
        return None
    log_path = Path(log_path)
    L = json.loads(log_path.read_text())
    ev = [e for e in L.get("events", []) if e.get("event") != "BACK"]
    ev += [{"t": a["t"], "event": a["to"], "id": a["id"], "text": a["reason"]}
           for a in L.get("alarms", [])]
    ev.sort(key=lambda e: e["t"])
    lines = "\n".join(f"{e['t']:7.1f}s  {e['event']:9s} #{e['id']}  {e.get('text', '')}"
                      for e in ev) or "(no events)"
    snaps = [Path(s) for s in L.get("snapshots", []) if Path(s).exists()][:max_images]
    parts = [{"type": "text", "text": REPORT.format(n_img=len(snaps), events=lines)}]
    for s in snaps:
        parts += [{"type": "text", "text": f"Snapshot: {s.name}"}, _img(s.read_bytes())]
    text = _chat(parts, max_tokens=8000, timeout=120, effort="medium")
    if not text:
        return None
    out = log_path.with_name(log_path.stem + "_report.md")
    out.write_text(f"# Spottr session report ({log_path.name})\n\n{text}\n", encoding="utf-8")
    print(f"Muse report saved: {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "report":
        session_report(sys.argv[2])
    else:
        print(__doc__)
