"""
Spottr dashboard backend - runs on the laptop, waits for the dashboard.

    1. Terminal 1:  modal serve spottr_live.py          (the GPU server)
    2. Terminal 2:  python spottr_app.py                 (this file)
    3. Open the dashboard:  http://127.0.0.1:8765
       (while developing the frontend:  cd frontend && npm run dev)

Nothing starts until you press "Start monitoring" in the dashboard. The
camera address you type stays on this laptop (it is never sent to the GPUs:
only JPEG frames are).

Dashboard <-> this app: one WebSocket at ws://127.0.0.1:8765/ws
  app -> page   {"type": "hello"}   zone, saved outline thumbnail, clips, Muse
                {"type": "status"}  phase, camera, GPU streams, frozen feed
                binary              the clean 1280 px camera frame (JPEG)
                {"type": "state"}   swimmers, deck, counts, alarms, events
                {"type": "calib_frame"} / {"type": "report"} / {"type": "error"}
  page -> app   start {source} | stop | ack {id} | ack_all
                grab_frame {source} | save_zone {points} | report
"""

import asyncio
import base64
import json
import os
import struct
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import websockets
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

import spottr_muse
import spottr_retention
from spottr_alarms import ALARM, EMERGENCY, NAMES, OK, WATCH, AlarmEngine
from spottr_live import ACCESS_TOKEN, Tracker, load_zone

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "frontend" / "dist"
ZONE_JSON, ZONE_JPG = ROOT / "pool_zone.json", ROOT / "pool_zone.jpg"
PORT = int(os.environ.get("SPOTTR_PORT", "8765"))
GPU_URL = os.environ.get(
    "SPOTTR_URL", "https://sle573--spottr-live-west-livedetector-web-dev.us-west.modal.run")
STREAMS = int(os.environ.get("SPOTTR_STREAMS", "7"))
MIN_LIVE = 3                 # start monitoring once this many GPUs are live
PROMPTS = "person,head"
MAX_W, QUALITY = 2016, 90
POOL_T, DECK_T, SUSTAIN_POOL, SUSTAIN_DECK = 0.30, 0.50, 0.20, 0.35
FROZEN_S = 4.0               # camera picture unchanged this long = frozen feed
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def now_wall():
    return datetime.now().strftime("%H:%M:%S")


def redact(src):
    if "://" in src and "@" in src:
        pre, rest = src.split("://", 1)
        creds, _, host = rest.partition("@")
        if ":" in creds:
            return f"{pre}://{creds.split(':', 1)[0]}:***@{host}"
    return src


def clips():
    out = []
    for d in ("raw_footage/clips", "raw_footage"):
        for f in sorted((ROOT / d).glob("*.mkv")) + sorted((ROOT / d).glob("*.mp4")):
            out.append(f.relative_to(ROOT).as_posix())
    return out


def resolve_source(src):
    """RTSP stays as is; a clip path is relative to the project folder."""
    src = src.strip()
    if "://" in src:
        return src, False
    p = Path(src)
    if not p.is_absolute():
        p = ROOT / p
    return str(p), True


class Session:
    """One monitoring session: camera -> GPUs -> tracker -> alarms -> page."""

    def __init__(self, app, source):
        self.app = app
        self.source, self.is_file = resolve_source(source)
        self.stop_flag = False
        self.phase = "connecting"
        self.camera = "connecting"
        self.gpus = [{"i": i + 1, "state": "starting", "msg": ""} for i in range(STREAMS)]
        self.conns = {}                       # i -> websocket
        self.inflight = {}                    # i -> {n: (frame, t_cap)}
        self.latest = {"frame": None, "t": 0.0, "n": 0, "done": False}
        self.lock = threading.Lock()
        self.last_change = time.perf_counter()
        self.last_frame = time.perf_counter()
        self.reconnects = 0
        self.frozen = False
        self.t_start = None
        self.wall_start = datetime.now()
        self.last_done = 0
        self.errors = 0
        self.delays, self.gpu_ms, self.checks = deque(maxlen=30), deque(maxlen=30), deque(maxlen=60)
        zone = load_zone(ZONE_JSON)
        self.zone_np = None if zone is None else np.array(zone, np.float32)
        self.tracker = Tracker(self._in_pool, create_pool=0.35, create_deck=DECK_T, create_tile=0.50,
                               sustain_pool=SUSTAIN_POOL, sustain_deck=SUSTAIN_DECK)
        self.alarms = AlarmEngine()
        self.events, self._tev = [], 0
        self._prev_state = {}
        self.muse_on = spottr_muse.available()
        self.muse_pool = ThreadPoolExecutor(max_workers=2) if self.muse_on else None
        self.muse = {}                        # tid -> {asked_t, asked_under, answer_t, verdict, reason, confidence}
        self.log, self.snapshots, self.snap_count = [], [], {}
        self.snap_dir = ROOT / "results" / "incidents" / datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------------------------------------------------------------- helpers
    def _in_pool(self, box):
        if self.zone_np is None:
            return True
        return cv2.pointPolygonTest(self.zone_np, ((box[0] + box[2]) / 2, min(box[3], 0.999)),
                                    False) >= 0

    def event(self, t, kind, tid, text):
        wall = (self.wall_start.timestamp() + (t or 0.0))
        self.events.append({"t": round(t or 0.0, 2), "wall": datetime.fromtimestamp(wall).strftime("%H:%M:%S"),
                            "kind": kind, "id": tid, "text": text})
        self.events = self.events[-200:]

    def status(self, message=""):
        live = sum(g["state"] == "live" for g in self.gpus)
        frozen_s = time.perf_counter() - min(self.last_change, self.last_frame) if self.frozen else 0.0
        return {"type": "status", "phase": self.phase, "camera": self.camera,
                "gpus": self.gpus, "gpus_live": live, "gpus_total": STREAMS,
                "frozen": self.frozen, "frozen_s": round(frozen_s, 1),
                "reconnects": self.reconnects, "muse": self.muse_on,
                "source": redact(self.source) if not self.is_file else Path(self.source).name,
                "message": message}

    async def push_status(self, message=""):
        await self.app.broadcast_json(self.status(message))

    # ---------------------------------------------------------------- camera
    def _open(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        return cap if cap.isOpened() else None

    def reader(self, cap):
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        start, prev = time.perf_counter(), None
        while not self.stop_flag:
            ok, frame = cap.read()
            if not ok:
                if self.is_file:
                    self.latest["done"] = True
                    return
                # camera dropped: keep trying (the frozen-feed banner is up meanwhile)
                cap.release()
                self.camera = "reconnecting"
                while not self.stop_flag:
                    self.reconnects += 1
                    time.sleep(1.0)
                    cap = self._open()
                    if cap:
                        self.camera = "connected"
                        break
                continue
            tnow = time.perf_counter()
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)
            if prev is None or float(np.mean(np.abs(small - prev))) > 0.3:
                self.last_change = tnow
            prev = small
            self.last_frame = tnow
            with self.lock:
                self.latest.update(frame=frame, t=tnow, n=self.latest["n"] + 1)
            if self.is_file:  # play a recording at normal speed, like a live camera
                time.sleep(max(0.0, start + self.latest["n"] / fps - time.perf_counter()))
        cap.release()

    # ---------------------------------------------------------------- GPUs
    def _ws_url(self):
        u = GPU_URL.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
        thr = min(POOL_T, DECK_T, SUSTAIN_POOL, SUSTAIN_DECK)
        return f"{u}/ws?token={ACCESS_TOKEN}&prompts={PROMPTS}&threshold={thr}&tiles=1"

    async def connect_gpu(self, i):
        g = self.gpus[i]
        g.update(state="starting", msg="")
        await self.push_status()
        try:
            ws = await websockets.connect(self._ws_url(), max_size=None, open_timeout=300,
                                          ping_interval=20, ping_timeout=120)
        except Exception as e:  # noqa: BLE001
            text = str(e)
            g.update(state="error", msg="token rejected" if "4401" in text else
                     "GPU server not reachable" if "getaddrinfo" in text or "refused" in text
                     else text[:80])
            await self.push_status()
            return
        if self.stop_flag:
            await ws.close()
            return
        self.conns[i], self.inflight[i] = ws, {}
        g.update(state="live")
        await self.push_status()
        asyncio.create_task(self.receiver(i, ws))

    async def receiver(self, i, ws):
        try:
            async for msg in ws:
                if self.stop_flag:
                    break
                try:
                    res = json.loads(msg)
                except ValueError:
                    self.errors += 1
                    continue
                got = self.inflight.get(i, {}).pop(res.get("id"), None)
                if got is None or res.get("error"):
                    self.errors += 1 if res.get("error") else 0
                    continue
                if res["id"] <= self.last_done:
                    continue  # an answer from a slower GPU, already overtaken
                self.last_done = res["id"]
                await self.handle(res, *got)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.conns.pop(i, None)
            self.inflight.pop(i, None)
            if not self.stop_flag:  # a stream dropped: say so, then reconnect it
                self.gpus[i].update(state="dropped", msg="reconnecting")
                self.event(self._t(), "GPU", None,
                           f"Stream {i + 1} dropped — restarting ({len(self.conns)} / {STREAMS} live)")
                await self.push_status()
                await asyncio.sleep(2)
                if not self.stop_flag:
                    asyncio.create_task(self.connect_gpu(i))

    def _t(self):
        return (time.perf_counter() - self.t_start) if self.t_start else 0.0

    async def sender(self):
        last_n = 0
        while not self.stop_flag:
            with self.lock:
                frame, t_cap, n, done = (self.latest["frame"], self.latest["t"],
                                         self.latest["n"], self.latest["done"])
            if done and n == last_n:
                await self.app.finish(self, "The recording ended.")
                return
            free = [i for i, ws in self.conns.items() if not self.inflight.get(i)]
            if frame is None or n == last_n or not free:
                await asyncio.sleep(0.004)
                continue
            last_n = n
            i = free[0]
            h, w = frame.shape[:2]
            if w > MAX_W:
                frame = cv2.resize(frame, (MAX_W, int(h * MAX_W / w)))
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
            if not ok:
                continue
            self.inflight[i][n] = (frame, t_cap)
            try:
                await self.conns[i].send(struct.pack("!Id", n, t_cap) + jpg.tobytes())
            except (websockets.exceptions.ConnectionClosed, KeyError):
                self.inflight.get(i, {}).pop(n, None)

    async def watchdog(self):
        """Frozen-feed check twice a second + status to the page."""
        while not self.stop_flag:
            tnow = time.perf_counter()
            frozen = (tnow - self.last_frame > 2.0) or (tnow - self.last_change > FROZEN_S)
            if frozen != self.frozen:
                self.frozen = frozen
                if frozen:
                    self.event(self._t(), "FEED", None,
                               f"Camera picture frozen for {FROZEN_S:.0f} s — reconnecting")
                else:
                    self.event(self._t(), "FEED", None, "Camera picture is moving again")
            await self.push_status()
            await asyncio.sleep(0.5)

    # ---------------------------------------------------------------- main
    async def run(self):
        await self.push_status("Opening the camera…")
        cap = await asyncio.to_thread(self._open)
        if cap is None:
            self.camera, self.phase = "error", "error"
            await self.push_status("Can't open the camera. Check the RTSP address and the hotspot.")
            return
        self.camera = "connected"
        await self.push_status()
        for i in range(STREAMS):
            asyncio.create_task(self.connect_gpu(i))
        # wait until enough GPUs are live (or every one of them failed)
        while not self.stop_flag:
            live = sum(g["state"] == "live" for g in self.gpus)
            failed = sum(g["state"] == "error" for g in self.gpus)
            if live >= min(MIN_LIVE, STREAMS) or (live and live + failed == STREAMS):
                break
            if failed == STREAMS:
                self.phase = "error"
                msg = next((g["msg"] for g in self.gpus if g["msg"]), "")
                await self.push_status("GPU server not reachable — is `modal serve spottr_live.py` "
                                       f"running? ({msg})")
                cap.release()
                return
            await asyncio.sleep(0.3)
        if self.stop_flag:
            cap.release()
            return
        self.phase = "live"
        self.t_start = time.perf_counter()
        self.last_change = self.last_frame = time.perf_counter()
        self.event(0.0, "START", None, "Monitoring started")
        threading.Thread(target=self.reader, args=(cap,), daemon=True).start()
        await self.push_status()
        await asyncio.gather(self.sender(), self.watchdog())

    async def stop(self):
        self.stop_flag = True
        for ws in list(self.conns.values()):
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass
        return self.save_log()

    # ---------------------------------------------------------------- per check
    def _next_step(self, tr, level):
        """(level name, seconds) of the next alarm step for this swimmer, or None."""
        st = tr["state"]
        A = self.alarms
        steps = []
        if st in ("SUBMERGED", "UNSEEN") or (st == "LOST" and not tr.get("lost_at_edge")):
            u = tr["under_s"]
            steps += [(WATCH, A.watch_under - u), (ALARM, A.alarm_under - u), (EMERGENCY, A.emergency_under - u)]
        head_up = self.tracker.has_heads and max(tr.get("head_hist") or [0.0]) >= self.tracker.head_bar
        if st in ("SURFACE", "SUBMERGED") and not head_up:
            s = tr.get("still_s", 0.0)
            steps += [(WATCH, A.still_watch - s), (ALARM, A.still_alarm - s)]
        if st == "OUTVIEW":
            steps += [(WATCH, A.outview_watch - tr["under_s"]), (ALARM, self.tracker.keep["pool"] - tr["under_s"])]
        ahead = [(lv, s) for lv, s in steps if lv > level and s > 0]
        if not ahead:
            return None
        lv, s = min(ahead, key=lambda x: x[1])
        return {"level": NAMES[lv], "in_s": round(s, 1)}

    def _crop(self, img, box, pad=0.6):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = max(x2 - x1, 0.03), max(y2 - y1, 0.03)
        xa, xb = int(max(0, x1 - pad * bw) * w), int(min(1, x2 + pad * bw) * w)
        ya, yb = int(max(0, y1 - pad * bh) * h), int(min(1, y2 + pad * bh) * h)
        c = img[ya:max(yb, ya + 2), xa:max(xb, xa + 2)]
        if c.shape[1] < 384:
            c = cv2.resize(c, (384, int(c.shape[0] * 384 / max(c.shape[1], 1))), interpolation=cv2.INTER_CUBIC)
        return cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()

    def _ask_muse(self, tid, crop, full, reason, t_obs, under):
        # one dict per swimmer: the last answer stays visible while it re-checks
        info = self.muse.setdefault(tid, {})
        info.update(asked_t=round(t_obs, 1), asked_under=round(under, 1), pending=True)  # under = the swimmer's timer when asked
        r = spottr_muse.second_opinion(crop, full, tid, reason)
        if not r:
            info["pending"] = False
            return
        info.update(pending=False, answer_s=round(self._t() - t_obs, 1), verdict=r["verdict"],
                    reason=r.get("reason", ""), confidence=r.get("confidence"),
                    head_above_water=r.get("head_above_water"), posture=r.get("posture"))
        self.alarms.muse_opinion(tid, r["verdict"], r.get("reason", ""), t=t_obs)
        extra = " → watch cancelled" if r["verdict"] == "ok" else ""
        self.event(self._t(), "MUSE", tid, f"“{r['verdict']}: {r.get('reason', '')}”{extra}")

    async def handle(self, res, frame, t_cap):
        t = t_cap - self.t_start
        tnow = time.perf_counter()
        self.delays.append((tnow - t_cap) * 1000)
        self.gpu_ms.append(res.get("gpu_ms", 0.0))
        self.checks.append(tnow)
        tr_obj = self.tracker
        tr_obj.update(res["dets"], t)
        counts = tr_obj.counts(t)
        changes = self.alarms.update(tr_obj, t)

        # tracker events -> dashboard events (skip the noisy ones)
        for e in tr_obj.events[self._tev:]:
            if e["event"] not in ("BACK", "STILL", "MOVING"):
                self.event(e["t"], e["event"], e["id"], e["text"])
        self._tev = len(tr_obj.events)
        for c in changes:
            self.event(t, c["to"] if c["to"] != "OK" else "CLEAR", c["id"], c["reason"] or "back to OK")
            tr = tr_obj.tracks.get(c["id"])
            if (tr and c["to"] not in ("OK",) and self.alarms.level(c["id"]) > OK
                    and self.snap_count.get(c["id"], 0) < 5):
                self.snap_dir.mkdir(parents=True, exist_ok=True)
                p = self.snap_dir / f"{t:07.1f}s_id{c['id']}_{c['to']}.jpg"
                p.write_bytes(self._crop(frame, tr["box"], pad=2.0))
                self.snapshots.append(str(p))
                self.snap_count[c["id"]] = self.snap_count.get(c["id"], 0) + 1

        # Muse second opinion: once a swimmer is at WATCH or above, every 10 s
        if self.muse_pool:
            for tid, s in self.alarms.state.items():
                m = self.muse.get(tid)
                if s["level"] >= WATCH and (not m or (not m.get("pending") and t - m["asked_t"] >= 10.0)):
                    tr = tr_obj.tracks.get(tid)
                    if tr:
                        small = cv2.resize(frame, (1280, int(frame.shape[0] * 1280 / frame.shape[1])))
                        self.muse.setdefault(tid, {}).update(asked_t=round(t, 1), pending=True)
                        self.muse_pool.submit(self._ask_muse, tid, self._crop(frame, tr["box"]),
                                              cv2.imencode(".jpg", small)[1].tobytes(), s["reason"], t,
                                              max(tr["under_s"], tr.get("still_s", 0.0)))

        swimmers, deck, unverified = [], [], []
        for tid, tr in tr_obj.tracks.items():
            if not tr["confirmed"]:
                continue
            if tr["zone"] == "deck":
                if t - tr["last"] < 2.0:
                    deck.append({"id": tid, "box": tr["box"]})
                continue
            if not tr["verified"]:
                unverified.append({"id": tid, "box": tr["box"]})
                continue
            prev = self._prev_state.get(tid)
            if tr["state"] == "OUTVIEW" and prev != "OUTVIEW":
                side = "left" if tr["box"][0] < 0.5 else "right"
                self.event(t, "OUT", tid, f"swam past the {side} edge")
            self._prev_state[tid] = tr["state"]
            s = self.alarms.state.get(tid, {})
            lvl = s.get("level", OK)
            head_up = tr_obj.has_heads and max(tr.get("head_hist") or [0.0]) >= tr_obj.head_bar
            swimmers.append({
                "id": tid, "box": [round(v, 4) for v in tr["box"]], "state": tr["state"],
                "under_s": round(tr["under_s"], 1), "still_s": round(tr.get("still_s", 0.0), 1),
                "head_known": tr_obj.has_heads, "head_up": bool(head_up),
                "alarm": NAMES[lvl], "reason": s.get("reason", ""),
                "alarm_since": round(t - s.get("since", t), 1), "acked": s.get("acked", False),
                "border": self.alarms.border(tid, t), "next": self._next_step(tr, lvl),
                "muse": self.muse.get(tid), "lost_at_edge": tr.get("lost_at_edge", False),
                "side": "left" if tr["box"][0] < 0.5 else "right",
                "seen_ago": round(t - tr["last"], 1),
                "in_pool_s": round(t - tr.get("entered_at", tr["first"]), 1),
            })
        rank = {"EMERGENCY": 4, "ALARM": 3, "WATCH": 2, "OK": 0}
        swimmers.sort(key=lambda w: (-rank[w["alarm"]], w["state"] not in ("LOST", "OUTVIEW"),
                                     -w["under_s"]))
        counts = dict(counts, lost=sum(w["state"] == "LOST" for w in swimmers),
                      above=sum(w["state"] == "SURFACE" for w in swimmers))
        span = (self.checks[-1] - self.checks[0]) if len(self.checks) > 1 else 0
        perf = {"checks_s": round((len(self.checks) - 1) / span, 1) if span > 0 else 0.0,
                "delay_ms": round(sorted(self.delays)[len(self.delays) // 2]),
                "gpu_ms": round(sorted(self.gpu_ms)[len(self.gpu_ms) // 2]),
                "gpus_live": len(self.conns), "gpus_total": STREAMS, "errors": self.errors}
        state = {"type": "state", "t": round(t, 2), "session_s": round(t, 1), "perf": perf,
                 "counts": counts, "swimmers": swimmers, "deck": deck, "unverified": unverified,
                 "top": NAMES[self.alarms.top()], "sounding": self.alarms.sounding(),
                 "events": self.events[-40:][::-1]}
        self.log.append({"t": round(t, 3), "delay_ms": round(self.delays[-1], 1),
                         "gpu_ms": res.get("gpu_ms", 0.0), "counts": counts,
                         "raw": res["dets"], "alarm_top": state["top"]})
        small = cv2.resize(frame, (1280, int(frame.shape[0] * 1280 / frame.shape[1])))
        jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        await self.app.broadcast_frame(jpg, json.dumps(state))

    def save_log(self):
        if not self.log:
            return None
        (ROOT / "results").mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = ROOT / "results" / f"live_log_{stamp}.json"
        out.write_text(json.dumps({
            "run": {"utc": stamp, "source": redact(self.source), "streams": STREAMS,
                    "prompts": PROMPTS.split(","), "zone": None if self.zone_np is None else self.zone_np.tolist()},
            "events": self.tracker.events, "alarms": self.alarms.events,
            "dashboard_events": self.events, "snapshots": self.snapshots,
            "muse": [dict(id=k, **v) for k, v in self.muse.items()], "frames": self.log}, indent=1))
        return out


class App:
    def __init__(self):
        self.pages = {}             # websocket -> {queue, frame, ev}
        self.session = None
        self.last_status = {"type": "status", "phase": "idle", "camera": "idle", "gpus": [],
                            "gpus_live": 0, "gpus_total": STREAMS, "frozen": False,
                            "muse": spottr_muse.available(), "message": ""}

    # ---- page messages
    async def broadcast_json(self, obj):
        if obj.get("type") == "status":
            self.last_status = obj
        await self._send_all([json.dumps(obj)])

    async def broadcast_frame(self, jpg, state_text):
        # frames: only the newest pair is kept per page (a slow tab skips frames)
        for slot in self.pages.values():
            slot["frame"] = [jpg, state_text]
            slot["ev"].set()

    async def _send_all(self, msgs):
        # control messages (status, hello, report): always delivered, in order
        for slot in self.pages.values():
            slot["queue"].extend(msgs)
            slot["ev"].set()

    def _reply(self, ws, obj):
        slot = self.pages.get(ws)
        if slot:
            slot["queue"].append(json.dumps(obj))
            slot["ev"].set()

    async def _pump(self, ws, slot):
        """One sender per dashboard tab, so a stuck or slow tab can never hold
        up monitoring or the other tabs."""
        while True:
            await slot["ev"].wait()
            slot["ev"].clear()
            msgs, slot["queue"] = slot["queue"], []
            if slot["frame"]:
                msgs += slot["frame"]
                slot["frame"] = None
            for m in msgs:
                await ws.send(m)

    def hello(self):
        zone = load_zone(ZONE_JSON)
        thumb = base64.b64encode(ZONE_JPG.read_bytes()).decode() if ZONE_JPG.exists() else None
        saved = (datetime.fromtimestamp(ZONE_JSON.stat().st_mtime).strftime("%b %d, %H:%M")
                 if ZONE_JSON.exists() else None)
        A = AlarmEngine()
        return {"type": "hello", "zone": zone, "zone_saved": saved, "zone_thumb": thumb,
                "clips": clips(), "muse": spottr_muse.available(), "streams": STREAMS,
                "thresholds": {"watch_under": A.watch_under, "alarm_under": A.alarm_under,
                               "emergency_under": A.emergency_under, "still_watch": A.still_watch,
                               "still_alarm": A.still_alarm, "outview_watch": A.outview_watch}}

    async def finish(self, session, message):
        if self.session is session:
            out = await session.stop()
            self.session = None
            await self.broadcast_json({"type": "status", "phase": "stopped", "camera": "idle",
                                       "gpus": [], "gpus_live": 0, "gpus_total": STREAMS,
                                       "frozen": False, "muse": spottr_muse.available(),
                                       "message": message, "log": out.name if out else None})
            if out and spottr_muse.available():
                asyncio.create_task(self._report(out))

    async def _report(self, log_path):
        p = await asyncio.to_thread(spottr_muse.session_report, log_path)
        if p:
            await self.broadcast_json({"type": "report", "text": Path(p).read_text(encoding="utf-8"),
                                       "file": Path(p).name})
        else:
            await self.broadcast_json({"type": "error", "message": "Muse could not write the report."})

    # ---- commands
    async def command(self, ws, m):
        cmd = m.get("cmd")
        s = self.session
        if cmd == "start":
            if s:
                return
            src = (m.get("source") or "").strip()
            if not src:
                self._reply(ws, {"type": "error", "message": "Enter the camera's RTSP address."})
                return
            self.session = Session(self, src)
            asyncio.create_task(self._run(self.session))
        elif cmd == "stop" and s:
            await self.finish(s, "Monitoring stopped.")
        elif cmd == "ack" and s:
            s.alarms.ack(int(m["id"]))
        elif cmd == "ack_all" and s:
            for tid in list(s.alarms.state):
                s.alarms.ack(tid)
        elif cmd == "grab_frame":
            src, _ = resolve_source(m.get("source") or "")
            jpg = await asyncio.to_thread(self._grab, src)
            self._reply(ws, {"type": "calib_frame", "jpeg": jpg} if jpg else
                        {"type": "error", "message": "Couldn't get a picture from the camera."})
        elif cmd == "save_zone":
            pts = [[round(float(x), 4), round(float(y), 4)] for x, y in m.get("points", [])]
            if len(pts) < 3:
                self._reply(ws, {"type": "error", "message": "Need at least 3 corners."})
                return
            ZONE_JSON.write_text(json.dumps({"points": pts}, indent=1))
            if m.get("jpeg"):
                ZONE_JPG.write_bytes(base64.b64decode(m["jpeg"]))
            await self._send_all([json.dumps(self.hello())])
        elif cmd == "report" and s:
            out = s.save_log()
            if out:
                asyncio.create_task(self._report(out))

    def _grab(self, src):
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        frame = None
        for _ in range(30):  # cameras often start grey
            ok, f = cap.read()
            if ok:
                frame = f
        cap.release()
        if frame is None:
            return None
        small = cv2.resize(frame, (1280, int(frame.shape[0] * 1280 / frame.shape[1])))
        return base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()

    async def _run(self, session):
        try:
            await session.run()
        except Exception as e:  # noqa: BLE001 - never leave the page hanging
            session.phase = "error"
            await session.push_status(f"Spottr stopped: {type(e).__name__}: {e}")
        if session.phase == "error" and self.session is session:
            await session.stop()
            self.session = None

    async def handler(self, ws):
        slot = {"queue": [json.dumps(self.hello()), json.dumps(self.last_status)],
                "frame": None, "ev": asyncio.Event()}
        slot["ev"].set()
        self.pages[ws] = slot
        pump = asyncio.create_task(self._pump(ws, slot))
        try:
            async for msg in ws:
                try:
                    m = json.loads(msg)
                except (ValueError, TypeError):
                    continue
                await self.command(ws, m)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.pages.pop(ws, None)
            pump.cancel()


TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".png": "image/png",
         ".svg": "image/svg+xml", ".jpg": "image/jpeg", ".woff2": "font/woff2", ".json": "application/json"}


def static(connection, request):
    """Serves the built dashboard (frontend/dist) on the same port; /ws is the WebSocket."""
    if request.path.startswith("/ws"):
        return None
    if not DIST.exists():
        return connection.respond(503, "Dashboard not built yet: cd frontend && npm run build "
                                       "(or use npm run dev while developing)\n")
    path = request.path.split("?")[0].lstrip("/") or "index.html"
    f = (DIST / path).resolve()
    if DIST.resolve() not in f.parents or not f.is_file():
        f = DIST / "index.html"
    body = f.read_bytes()
    return Response(200, "OK", Headers({"Content-Type": TYPES.get(f.suffix, "application/octet-stream"),
                                        "Content-Length": str(len(body)), "Cache-Control": "no-cache"}), body)


async def main():
    if not ACCESS_TOKEN:
        raise SystemExit("SPOTTR_TOKEN is missing from .env")
    try:
        spottr_retention.apply(days=3.0)
    except Exception as e:  # noqa: BLE001
        print(f"Retention check failed: {e}")
    app = App()
    async with serve(app.handler, "127.0.0.1", PORT, process_request=static, max_size=2 ** 24):
        print(f"Spottr dashboard: http://127.0.0.1:{PORT}   (WebSocket ws://127.0.0.1:{PORT}/ws)")
        print(f"GPU server: {GPU_URL}  ·  Muse Spark: {'on' if spottr_muse.available() else 'off'}")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
