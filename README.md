<p align="center"><img src="frontend/public/logo.png" width="160" alt="Spottr mascot"></p>

# Spottr — AI Lifeguard Assistant

Drowning is silent: no splashing, no shouting, and it can happen in under a minute.
Spottr watches a pool camera, follows every swimmer, times how long each one has
been underwater or motionless, and alerts the lifeguard before it's too late.

Built for the Meta hackathon with **SAM 3.1**, **Muse Spark** and **Muse Image**.

## What it does

- **Finds every person and head** in each camera frame with SAM 3.1 (full frame plus
  four zoomed-in quarters, so far-away swimmers aren't missed).
- **Tracks each swimmer with an ID**, keeps that ID while they're underwater, and
  separates people in the pool from people on the deck.
- **Times underwater and stillness**, then escalates:
  - **WATCH** at 10 s, **ALARM** at 20 s, **EMERGENCY** if a swimmer is lost mid-pool.
  - Timings follow lifeguard research (the 10/20 rule: spot in 10 s, reach in 20 s).
- **Alarm output**: the screen flashes, a red border closes around the swimmer, and
  an alarm sounds. One click acknowledges.
- **Muse Spark second opinion**: at WATCH, Muse Spark looks at the swimmer and
  reports whether the head is above water and how they're moving. It can raise an
  alarm but can never silence one.
- **Session report**: Muse Spark writes a summary of every event after a session.
- **Fail-safes**: warns when the camera feed freezes or drops.
- **Privacy**: footage is deleted after 3 days; the camera password stays on the laptop.

## How it works

```
RTSP camera ──Wi-Fi──► laptop (spottr_app.py) ──► Modal GPUs running SAM 3.1
                           │                         (spottr_live.py)
                           ├─ tracker + underwater/stillness timers
                           ├─ alarm engine (spottr_alarms.py)
                           ├─ Muse Spark second opinion + report (spottr_muse.py)
                           └─ dashboard in the browser (frontend/, React + Tailwind + shadcn)
```

## Running it

1. Python 3.12+, Node 20+, a [Modal](https://modal.com) account, and a Hugging Face
   token saved in Modal as the secret `huggingface-secret`.
2. Create `.env` in the project folder (it is git-ignored):
   ```
   SPOTTR_TOKEN=<any long random string>
   MODEL_API_KEY=<Meta Model API key, for Muse Spark>      # optional
   SPOTTR_CAMERA=rtsp://user:password@camera-ip:554/stream1 # optional
   SPOTTR_GPU=H100       # optional: H100 (default), A10, L4, ...
   SPOTTR_STREAMS=7      # optional: parallel GPUs (default 7)
   ```
3. Install and build:
   ```
   python -m venv .venv
   .venv\Scripts\pip install modal opencv-python numpy websockets
   cd frontend && npm install && npm run build && cd ..
   ```
4. Start the GPU server, then the dashboard:
   ```
   modal serve spottr_live.py
   python spottr_app.py
   ```
5. Open http://127.0.0.1:8765, paste the camera address, trace the pool outline once,
   and press **Start monitoring**.

## Cost

Any RTSP camera works; ours cost $20. In testing, 4 × A10 GPUs (about $4.40/hour)
fired every alarm at the same moments as 7 × H100s, with about 1.4 s of delay instead of 0.7 s.

## Limits

One camera, not yet tested in a crowded pool, and strong glare can hide swimmers.
Spottr assists a lifeguard; it does not replace one.
