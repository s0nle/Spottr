"""
Spottr - LIVE test: camera -> laptop -> Modal GPU (SAM 3.1) -> boxes + IDs back.

One file, two parts:
  SERVER (runs on Modal): SAM 3.1 finds every "person" / "head" in each frame it
          receives over a WebSocket and sends the boxes straight back.
  CLIENT (runs on your laptop): reads the camera (or a video file), always sends
          the NEWEST frame (old frames are dropped, so delay never builds up),
          keeps IDs with a simple tracker, shows a live window and measures delay.

HOW TO RUN (two terminals in VS Code, both with .venv active):

  Terminal 1 - start the GPU server and leave it running:
      modal serve spottr_live.py
    Copy the URL it prints (ends in .modal.run).

  Terminal 2 - start the client:
      pip install opencv-python websockets numpy        (first time only)
      $cam = "rtsp://yourusername:yourpassword@192.168.137.179:554/stream1"
      python spottr_live.py --url https://YOUR-URL.modal.run --source $cam

    Use stream1 (the camera's HD stream). stream2 is only 640x360, which hides
    swimmers under the water (tested: score 0.11 vs 0.41 in HD on the same frame).

    Fake-live test with a saved clip (plays it at normal speed like a camera):
      python spottr_live.py --url https://YOUR-URL.modal.run --source raw_footage/pool_20261102_091918.mkv

  POOL OUTLINE (do this once per camera position):
      python spottr_live.py --set-zone --source $cam
    Click the corners of the WATER, then press Enter to save pool_zone.json.
    u = undo last point, Esc = cancel. If the camera moves, do it again.
    Inside the outline the threshold is lower (--pool-threshold, default 0.30)
    so faint swimmers under the water still count. Outside it (the deck) the
    normal --threshold (default 0.50) applies, and those people are shown in
    grey and NOT counted. A box counts as "in the pool" when its bottom-center
    is inside the outline (a swimmer's body is in the water; someone standing
    has their feet on the deck).

  Useful flags: --max-minutes 10 (auto-stop so the GPU can't run forever),
  --no-show (headless, for SSH), --record live.mp4 (save the live view).

  Press q in the video window to stop. Then Ctrl+C in terminal 1 to stop the GPU.

The first frame can take 1-2 minutes while Modal starts the GPU and loads SAM 3.1.
"""

import os

import modal


def _load_dotenv():
    """KEY=VALUE lines from .env (next to this file) into os.environ. On the
    Modal GPU there is no .env: the token arrives as a Modal secret instead."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v


_load_dotenv()

# Anyone with your server URL AND this token can use your GPU while
# `modal serve` is running. Secrets live in .env (never in this file):
#   SPOTTR_TOKEN=...        any long random string
#   SPOTTR_CAMERA=rtsp://user:password@192.168.137.179:554/stream1   (optional)
#   MODEL_API_KEY=...       Meta Model API key, for Muse Spark (optional)
ACCESS_TOKEN = os.environ.get("SPOTTR_TOKEN", "")

# GPU type (set SPOTTR_GPU in .env to change). H100 is fastest; "L40S", "A10" or
# "L4" cost less per hour but each check takes longer.
GPU = os.environ.get("SPOTTR_GPU", "H100")

# Run the GPU on the US west coast, close to the pool, AND route the connection
# through the west coast (otherwise frames still detour through us-east).
# Note: Modal charges 1.75x for functions pinned to a narrow region like us-west.
# To go back to the default (us-east, normal price), delete the two lines
# `region=REGION,` and `routing_region=REGION,` in @app.cls below.
REGION = "us-west"

# New app name because Modal only sets the routing region on a function's first deploy
app = modal.App("spottr-live-west")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("huggingface_hub", "pillow", "numpy", "einops", "fastapi[standard]")
    .run_commands(
        'pip install "sam3[notebooks] @ git+https://github.com/facebookresearch/sam3.git"'
    )
)

weights = modal.Volume.from_name("sam31-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"
CHECKPOINT = "sam3.1_multiplex.pt"


# =============================================================================
# SERVER (Modal GPU)
# =============================================================================
@app.cls(
    gpu=GPU,
    region=REGION,
    routing_region=REGION,
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret"),
             # the token travels to the GPU as a secret (built from your .env)
             modal.Secret.from_dict({"SPOTTR_TOKEN": ACCESS_TOKEN})],
    volumes={WEIGHTS_DIR: weights},
    timeout=86400,  # one live session can last up to 24 hours (Modal's maximum)
    scaledown_window=120,  # GPU shuts down 2 min after the last frame
)
class LiveDetector:
    @modal.enter()
    def load(self):
        import os

        from huggingface_hub import hf_hub_download
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        ckpt_path = os.path.join(WEIGHTS_DIR, CHECKPOINT)
        if not os.path.exists(ckpt_path):
            hf_hub_download(repo_id="facebook/sam3.1", filename=CHECKPOINT,
                            local_dir=WEIGHTS_DIR)
            weights.commit()
        self.model = build_sam3_image_model(checkpoint_path=ckpt_path, load_from_HF=False)
        self.Sam3Processor = Sam3Processor
        print("SAM 3.1 loaded, ready for frames")

    def detect(self, jpeg: bytes, prompts: list, threshold: float, tiles: bool) -> tuple:
        """SAM 3.1 on the whole frame (every prompt) and, if tiles is on, on 4
        overlapping zoomed tiles ("person" only). All boxes are merged into one
        box per person. Returns (detections, ms)."""
        import io
        import time

        import torch
        from PIL import Image

        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        W, H = img.size
        processor = self.Sam3Processor(self.model, confidence_threshold=threshold)
        t0 = time.perf_counter()
        raw = []

        def run(im, ox, oy, src, prs):
            cw, ch = im.size
            state = processor.set_image(im)  # the heavy part, shared by all prompts
            for pr in prs:
                out = processor.set_text_prompt(state=state, prompt=pr)
                for (x1, y1, x2, y2), s in zip(out["boxes"].float().cpu().tolist(),
                                               out["scores"].float().cpu().tolist()):
                    raw.append({
                        "prompt": pr, "score": round(s, 3), "src": src,
                        # box as fractions of the FULL frame, so any resolution works
                        "box": [round((ox + x1) / W, 4), round((oy + y1) / H, 4),
                                round((ox + x2) / W, 4), round((oy + y2) / H, 4)],
                    })

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            run(img, 0, 0, "full", prompts)
            if tiles:
                # 2x2 tiles, 15% overlap: each is ~2x zoom on a 2016 px frame
                tw, th = int(W * 0.575), int(H * 0.575)
                for ty in (0, H - th):
                    for tx in (0, W - tw):
                        # all prompts: "head" on far swimmers needs the zoom
                        # (extra prompts are cheap, set_image is the heavy part)
                        run(img.crop((tx, ty, tx + tw, ty + th)), tx, ty, "tile", prompts)
        torch.cuda.synchronize()
        return merge_dets(raw), round((time.perf_counter() - t0) * 1000, 1)

    @modal.asgi_app()
    def web(self):
        import asyncio
        import json
        import struct

        from fastapi import FastAPI, WebSocket, WebSocketDisconnect

        api = FastAPI()

        @api.websocket("/ws")
        async def ws(websocket: WebSocket):
            q = websocket.query_params
            if not ACCESS_TOKEN or q.get("token") != ACCESS_TOKEN:
                await websocket.close(code=4401)
                return
            prompts = [p.strip() for p in q.get("prompts", "person").split(",") if p.strip()]
            threshold = float(q.get("threshold", "0.5"))
            tiles = q.get("tiles", "1") == "1"
            await websocket.accept()
            print(f"Client connected: prompts={prompts} threshold={threshold} tiles={tiles}")
            try:
                while True:
                    msg = await websocket.receive_bytes()
                    # message = 4-byte frame number + 8-byte send time + JPEG.
                    # One bad frame must not kill the session: any per-frame
                    # failure is reported as an error reply, loop continues.
                    frame_id, t_sent = -1, 0.0
                    try:
                        if len(msg) < 12:
                            raise ValueError(f"message too short ({len(msg)} bytes)")
                        frame_id, t_sent = struct.unpack("!Id", msg[:12])
                        dets, ms = await asyncio.to_thread(
                            self.detect, msg[12:], prompts, threshold, tiles
                        )
                        await websocket.send_text(json.dumps({
                            "id": frame_id, "t_sent": t_sent, "gpu_ms": ms, "dets": dets,
                        }))
                    except WebSocketDisconnect:
                        raise
                    except Exception as e:  # noqa: BLE001 - report, keep serving
                        print(f"Frame {frame_id} failed: {e!r}")
                        try:
                            await websocket.send_text(json.dumps({
                                "id": frame_id, "t_sent": t_sent, "gpu_ms": 0.0,
                                "dets": [], "error": f"{type(e).__name__}: {e}",
                            }))
                        except WebSocketDisconnect:
                            raise
            except WebSocketDisconnect:
                print("Client disconnected")

        return api


# =============================================================================
# CLIENT (your laptop)
# =============================================================================
def merge_dets(dets, iou_t=0.5, contain_t=0.7, max_ratio=8.0, max_tile_area=0.05):
    """One box per person from every pass (full frame, tiles) and prompt.

    Two boxes are the same person when they overlap a lot (IoU) or one sits
    mostly inside the other (a head inside a body, half a person cut by a
    tile edge). A tiny box inside a HUGE box is NOT merged (max_ratio), so a
    big false box over the water can't swallow a real swimmer.
    The merged box is the full-frame "person" box if there is one (best
    shape), the score is the best score, and `tile_only` says no full-frame
    pass saw it (tiles make more junk, so the client is stricter with those).
    """
    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    # "head" boxes are kept apart (one per head): they tell the tracker
    # whether a swimmer's head is above the water, they don't make people
    heads = [d for d in dets if d["prompt"] == "head"]
    dets = [d for d in dets if d["prompt"] != "head"]
    head_out = []
    for d in sorted(heads, key=lambda d: -d["score"]):
        for h in head_out:
            k = h["box"]
            ix = area([max(d["box"][0], k[0]), max(d["box"][1], k[1]),
                       min(d["box"][2], k[2]), min(d["box"][3], k[3])])
            if ix / max(min(area(d["box"]), area(k)), 1e-9) > 0.5:
                h["tile_only"] = h["tile_only"] and d["src"] == "tile"
                break
        else:
            head_out.append({"prompt": "head", "score": d["score"], "box": d["box"],
                             "tile_only": d["src"] == "tile"})

    groups = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        for g in groups:
            k = g["rep"]
            ix = area([max(d["box"][0], k[0]), max(d["box"][1], k[1]),
                       min(d["box"][2], k[2]), min(d["box"][3], k[3])])
            a1, a2 = area(d["box"]), area(k)
            iou = ix / max(a1 + a2 - ix, 1e-9)
            contain = ix / max(min(a1, a2), 1e-9)
            ratio = max(a1, a2) / max(min(a1, a2), 1e-9)
            if iou > iou_t or (contain > contain_t and ratio <= max_ratio):
                g["members"].append(d)
                break
        else:
            groups.append({"rep": d["box"], "members": [d]})
    out = []
    for g in groups:
        ms = g["members"]
        # Tiles make big false boxes over the water (seen 5-36% of the frame);
        # real swimmers found ONLY by tiles were <= 3.7%. A big real person
        # (near the camera) is always also found by the full-frame pass.
        if all(m["src"] == "tile" for m in ms) and area(g["rep"]) > max_tile_area:
            continue
        best = max(ms, key=lambda m: (m["src"] == "full" and m["prompt"] == "person",
                                      m["src"] == "full", m["score"]))
        out.append({
            "prompt": "person",
            "score": max(m["score"] for m in ms),
            "box": best["box"],
            "tile_only": all(m["src"] == "tile" for m in ms),
            "seen_by": sorted({f"{m['src']}:{m['prompt']}" for m in ms}),
        })
    return out + head_out


class Tracker:
    """One ID per person across the pool AND the deck.

    Thresholds (score needed):
      create    to start a NEW person: pool 0.35, deck 0.50, tile-only 0.50
      sustain   to keep an EXISTING person: pool 0.20, deck 0.35
    A new person is only confirmed (counted, drawn solid) after being seen in
    2 of 3 checks in a row AND at least once clearly (score >= 0.60); junk on
    splashes and ripples never gets there. Unconfirmed tracks that miss 2 of
    3 checks are dropped.

    Zone (pool / deck) comes from the bottom-center of the box. A change of
    zone must hold for 1.5 s and come from a CLEAR detection, then logs
    ENTER (deck -> pool) or EXIT (pool -> deck) - but only if the track
    really moved there: ENTER needs travel without morphing (a static
    blob straddling the water line, or a giant box collapsing on pool
    infrastructure, is not someone jumping in). A hand-off links a person
    who just vanished to a new one right next to them in the other zone
    (climbing out / jumping in); nobody is handed off from under the
    water, and jump-ins need the same solid motion on the deck side.

    VERIFIED swimmers (seen ENTERING from the deck, present at startup, or
    promoted below) are the head count. Anything that just appears in the
    water is UNVERIFIED (purple): lane stripes on the pool floor do this. It
    stays on screen with its timer running but is not counted, and it fades
    quietly instead of raising LOST. It is promoted to VERIFIED (counted) once
    it swims >= 0.20 of the frame in 10 s, or is seen steadily for 5 s with a
    median score >= 0.55 (see _maybe_verify): swimmers who come in from a side
    of the frame get counted. A hot stripe can pass STEADY too (scores to
    0.98); motion can't separate it from a floater, so counting it is the
    accepted tradeoff - the head count stays honest everywhere else.

    A verified swimmer not CLEARLY seen (score >= strong_score) for keep_pool
    seconds becomes LOST: still counted, shown in red, until seen again
    (FOUND). Never silently removed (only after keep_lost). Deck people are
    dropped after keep_deck seconds (walked out of view).

    A pool swimmer who vanishes at the BOTTOM edge (near wall under the
    camera) for > 2 s is an EXIT. The left/right edges don't count: the water
    continues there, so the swimmer may be under it.

    Tracks may jump at most max_jump between sightings (box-size limits let a
    big near-camera box jump across the pool onto a lane stripe).

    Per pool swimmer:  SURFACE   seen, score at least 70% of this swimmer's own
                                 usual surface score (capped at 0.50)
                       SUBMERGED seen, but scoring clearly below that (diag:
                                 under-water 0.34-0.53, surface 0.70-0.95)
                       UNSEEN    not detected
                       OUTVIEW   at / past the left or right edge of the frame
                                 (timer runs, not counted as "under")
    `under_s` = seconds since last seen at the SURFACE. Back at the SURFACE
    needs 2 of the last 3 sightings at surface level AND (when the server
    sends "head" boxes) a clear head (>= head_bar) on them in the last 3
    sightings: a body lying under the water still scores ~0.7, its head
    doesn't (4-min clip, 74-86 s: body-only said 4.7 s, real was 12.5 s).

    After a jump-in the hand-off may pick a wrong box (a piece of the diver,
    a corner fixture); _fix_dive / _relink move the ID to the real
    splash-down within 8 s.

    Score alone can't catch a motionless floater: they score high, so they
    read SURFACE forever. `still_s` = seconds since the swimmer last moved,
    judged from the trail (middle of the last 3 sightings vs the middle of
    everything older in the window), so one jittery box can't reset it.
    Only VERIFIED pool swimmers raise STILL: passive at the SURFACE warns
    (long fuse, --still-warn), passive while SUBMERGED alarms (short fuse,
    --still-alarm). Moving again logs MOVING. UNSEEN/LOST freeze the timer.
    """

    def __init__(self, in_pool, create_pool=0.35, create_deck=0.50, create_tile=0.50,
                 sustain_pool=0.20, sustain_deck=0.35, keep_pool=60.0, keep_deck=8.0,
                 confirm_score=0.60, zone_hold_s=1.5, handoff_s=5.0, handoff_dist=0.12,
                 near_factor=1.5, strong_score=0.50, max_jump=0.12, keep_lost=600.0,
                 startup_s=10.0, still_window=10.0, still_min_n=4,
                 still_floor=0.02, still_size_k=0.30,
                 still_warn_s=30.0, still_alarm_s=15.0, enter_travel=0.06,
                 enter_size_ratio=2.5, head_bar=0.60):
        self.in_pool = in_pool
        self.zone_hold_s = zone_hold_s      # a zone change must hold this long
        self.handoff_s = handoff_s          # climbing out / jumping in: link a new
        self.handoff_dist = handoff_dist    # person to one who just vanished nearby
        self.near_factor = near_factor      # new boxes this close to a seen person = part of them
        # Only a sighting >= strong_score keeps a track from going LOST. Lane
        # stripes score 0.2-0.5 most of the time, so they can't keep a stolen
        # track alive; a swimmer under the water 60 s+ SHOULD go LOST (alarm).
        self.strong_score = strong_score
        # Largest distance (fraction of the frame) a track may jump between
        # sightings. Box-size-based limits let a big near-camera box jump across
        # the pool onto a lane stripe (seen: 0.25-0.47 jumps).
        self.max_jump = max_jump
        self.keep_lost = keep_lost  # a LOST swimmer stays on screen this long
        # Swimmers already in the pool when Spottr starts can't be seen entering:
        # anyone confirmed in the first startup_s counts as verified. (Start with
        # calm water: a lane stripe confirmed in this window is counted too.)
        # Short on purpose: real occupants are visible immediately, while
        # a flickering stripe takes seconds to confirm.
        self.startup_s, self.t0 = startup_s, None
        # stillness: judging window, sightings needed, move bar (floor +
        # size-relative), and the warn/alarm fuses (surface / submerged)
        self.still_window = still_window
        self.still_min_n = still_min_n
        self.still_floor, self.still_size_k = still_floor, still_size_k
        self.still_warn_s, self.still_alarm_s = still_warn_s, still_alarm_s
        # entries/exits must have moved this far: static ghosts straddling
        # zones or flickering deck->pool never become counted swimmers
        self.enter_travel = enter_travel
        # ... without morphing: a giant box collapsing on poolside
        # infrastructure "travels" as it shrinks (seen: 0.99 -> 0.22 tall)
        self.enter_size_ratio = enter_size_ratio
        # a new person is only counted after one CLEAR sighting: real swimmers
        # surface at 0.70-0.95, splash/ripple junk sits around 0.30-0.50
        self.confirm_score = confirm_score
        self.create = {"pool": create_pool, "deck": create_deck}
        self.sustain = {"pool": sustain_pool, "deck": sustain_deck}
        self.create_tile = create_tile
        self.keep = {"pool": keep_pool, "deck": keep_deck}
        # Timer reset needs the HEAD too: stripes score up to 0.98 as "person"
        # but never fire "head"; submerged heads leak up to 0.56 (diag), so
        # head_bar sits above that. Off automatically when the server sends
        # no head boxes (old logs, --prompt person).
        self.head_bar = head_bar
        self.has_heads, self._heads = False, []
        self._relink_wait = {}  # tid -> (cx, cy, t) of a first re-link sighting
        self.tracks, self.next_id, self.events = {}, 0, []
        self.entries = self.exits = 0

    def _event(self, t, kind, tid, text):
        e = {"t": round(t, 2), "event": kind, "id": tid, "text": text}
        self.events.append(e)
        print(f"  [{t:7.1f}s] {kind:9s} person {tid}: {text}")

    @staticmethod
    def _near_camera(box):
        # a tall box reaching the bottom of the frame = someone standing at the
        # near wall right by the camera (on the deck, not in the water)
        return box[3] > 0.97 and (box[3] - box[1]) > 0.40

    @staticmethod
    def _drop_parts(dets):
        """A box mostly inside a CLEARLY real person (full-frame, score >= 0.80,
        3x+ bigger) is part of them: a hand, a phone, a face of someone standing
        at the camera. (Tile-only boxes can't absorb anything, so a big false
        tile box over the water can never swallow a real swimmer.)"""
        def area(b):
            return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        big = [d for d in dets if d["score"] >= 0.80 and not d.get("tile_only")]
        keep = []
        for d in dets:
            a = area(d["box"])
            part = False
            for p in big:
                if p is d or area(p["box"]) < 3 * a:
                    continue
                b, q = d["box"], p["box"]
                ix = area([max(b[0], q[0]), max(b[1], q[1]), min(b[2], q[2]), min(b[3], q[3])])
                if ix / max(a, 1e-9) > 0.7:
                    part = True
                    break
            if not part:
                keep.append(d)
        return keep

    def update(self, dets, t):
        if self.t0 is None:
            self.t0 = t
        # head boxes (prompt "head") only vet surface resets, see _head_score
        self._heads = [d for d in dets if d.get("prompt") == "head"]
        dets = [d for d in dets if d.get("prompt") != "head"]
        if self._heads:
            self.has_heads = True  # this run asks SAM for heads
        dets = self._drop_parts(dets)
        for d in dets:
            d["zone"] = ("pool" if self.in_pool(d["box"]) and not self._near_camera(d["box"])
                         else "deck")
        cand = [d for d in dets if d["score"] >= self.sustain[d["zone"]]]
        centers = [((d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2,
                    max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1])) for d in cand]

        # every (track, detection) pair that is close enough, nearest first
        pairs = []
        for tid, tr in self.tracks.items():
            gap = t - tr["last"]
            for k, (cx, cy, size) in enumerate(centers):
                dist = ((cx - tr["cx"]) ** 2 + (cy - tr["cy"]) ** 2) ** 0.5
                # A box on the OTHER side of the pool outline can only carry a
                # track if it is clearly a person (>= confirm_score): a dark
                # post on the far deck (0.35-0.50) kept a far swimmer's ID
                # "alive" for 90 s while the swimmer was out of view.
                if (cand[k]["zone"] != tr["zone"]
                        and cand[k]["score"] < self.confirm_score):
                    continue
                if cand[k]["score"] < self.create[cand[k]["zone"]]:
                    # weak detection: may only continue a track right where it
                    # is (stops ripples, reflections and stripes dragging it away)
                    limit = min(0.04, max(0.02, 0.6 * min(size, tr["size"])))
                else:
                    limit = min(self.max_jump, max(0.05, max(size, tr["size"])))
                    # No longer limit after a gap: a missing swimmer's ID jumped
                    # 0.11-0.19 onto a post and lane stripes scoring 0.50-0.56.
                    # Coming back up further away is handled by _relink (clear
                    # full-frame box >= confirm_score only).
                    # just dived in: covers a lot of picture fast. CLEAR boxes
                    # only: a 0.37 tile box on a lane stripe 0.28 away took
                    # over a swimmer who had just jumped in
                    if (t < tr.get("fast_until", 0.0)
                            and cand[k]["score"] >= self.strong_score):
                        limit = 0.30
                if dist < limit:
                    # One swimmer often gets two boxes (a strong full-frame one
                    # and a weaker zoomed-tile one on part of the body). Prefer
                    # the strong one: following the weak one made the swimmer's
                    # real box start a second ID when the weak one vanished.
                    cost = dist + 0.06 * (1.0 - cand[k]["score"]) + (
                        0.02 if cand[k].get("tile_only") else 0.0)
                    pairs.append((cost, tid, k))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, tid, k in pairs:
            if tid in used_t or k in used_d:
                continue
            used_t.add(tid), used_d.add(k)
            self._hit(self.tracks[tid], cand[k], centers[k], t)

        # unmatched detections start a new person only above the CREATE bar
        for k, d in enumerate(cand):
            if k in used_d:
                continue
            bar = max(self.create[d["zone"]], self.create_tile if d["tile_only"] else 0.0)
            if d["score"] < bar:
                continue
            cx, cy, size = centers[k]
            # right next to someone already seen in THIS check = part of them
            # (splash, kick, arm, a second box on the same swimmer)
            if any(tr["last"] == t and tr["confirmed"]
                   and self._part_of(d["box"], (cx, cy, size), tr)
                   for tr in self.tracks.values()):
                continue
            if self._handoff(d, (cx, cy, size), t, used_t):
                continue
            if self._relink(d, (cx, cy, size), t, used_t):
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = dict(id=tid, cx=cx, cy=cy, size=size, box=d["box"],
                                    score=d["score"], zone=d["zone"], pending=None,
                                    first=t, last=t, last_strong=t, last_surface=t,
                                    hist=[True],
                                    confirmed=False, state="SURFACE", under_s=0.0,
                                    last_move=t, still_s=0.0, still=False,
                                    sm_cx=cx, sm_cy=cy,
                                    best=d["score"], surface_ref=d["score"],
                                    origin=(cx, cy), moved=0.0, born=d["zone"],
                                    # in the pool count only once VERIFIED. Pool
                                    # births always start UNVERIFIED (a stripe at
                                    # the near wall auto-counted otherwise); real
                                    # entrants promote via SWAM within seconds of
                                    # swimming. Deck births are verified: nobody
                                    # drowns on the deck.
                                    verified=(d["zone"] == "deck"))
            used_t.add(tid)

        self._fix_dive(t)

        # not seen this check
        for tid, tr in list(self.tracks.items()):
            if tid in used_t:
                continue
            tr["hist"].append(False)
            if not tr["confirmed"]:
                # never confirmed and missing 2 of the last 3 checks: junk
                if len(tr["hist"]) >= 3 and sum(tr["hist"][-3:]) <= 1:
                    del self.tracks[tid]
                continue
            if tr["zone"] == "pool":
                # Vanished while touching the frame edge = walked/swam out of
                # the camera's view (e.g. climbed out right under the camera,
                # where the water reaches the bottom of the frame). Not underwater.
                # Only the BOTTOM edge: the water continues past the left/right
                # edges, so a swimmer gone there may be under the water.
                # And only if they were at the SURFACE when last seen: someone
                # who went under at the near wall is under the water, not gone.
                at_edge = tr["box"][3] > 0.97
                was_up = tr["last_surface"] >= tr["last"]
                # just jumped in at the near wall and gone = under the water
                dived = t - tr.get("entered_at", -1e9) < 8.0
                if at_edge and was_up and not dived and t - tr["last"] > 2.0:
                    if tr["verified"]:  # unverified IDs were never counted
                        self.exits += 1
                        self._event(t, "EXIT", tid, "left the camera view at the frame edge")
                    del self.tracks[tid]
                    continue
                if tr["state"] != "LOST":
                    # vanished at a side edge = probably swam out of view
                    # (the water continues there): shown, timer running, but
                    # not counted as "under"
                    tr["state"] = "OUTVIEW" if self._side_edge(tr["box"]) else "UNSEEN"
                tr["under_s"] = t - tr["last_surface"]

        # not CLEARLY seen for keep_s:
        #   verified pool swimmer -> LOST: stays counted and on screen (red)
        #                            until seen again; never silently removed
        #   unverified / deck     -> dropped quietly
        for tid, tr in list(self.tracks.items()):
            if not tr["confirmed"]:
                continue
            quiet = t - tr["last_strong"]
            if tr["zone"] == "pool" and tr["verified"]:
                if quiet > self.keep["pool"] and tr["state"] != "LOST":
                    tr["lost_at_edge"] = tr["state"] == "OUTVIEW"
                    where = " (last seen at the frame edge)" if tr["lost_at_edge"] else ""
                    tr["state"] = "LOST"
                    self._event(t, "LOST", tid,
                                f"not clearly seen for {quiet:.0f}s in the pool{where}")
                elif quiet > self.keep_lost:
                    self._event(t, "DROPPED", tid, f"LOST for {quiet:.0f}s, removed")
                    del self.tracks[tid]
            elif quiet > self.keep[tr["zone"]]:
                del self.tracks[tid]
        return [tr for tr in self.tracks.values()]

    def _hit(self, tr, d, center, t):
        gap = t - tr["last"]
        if tr["confirmed"] and gap > 1.0:
            self._event(t, "BACK", tr["id"], f"seen again after {gap:.1f}s")
        tr.update(cx=center[0], cy=center[1], size=center[2], box=d["box"],
                  score=d["score"], last=t, best=max(tr["best"], d["score"]))
        # sizes over the last 3 s (someone crouching to jump in shrinks fast)
        tr["sizes"] = [(ts, sz) for ts, sz in tr.get("sizes", []) if t - ts <= 3.0]
        tr["sizes"].append((t, center[2]))
        # CLEAR sighting = this swimmer's own surface level (a far swimmer scores
        # 0.4-0.6 just for being small; a fixed 0.50 raised a false LOST)
        clear = d["score"] >= min(self.strong_score, 0.70 * tr["surface_ref"])
        if tr["state"] == "LOST":
            # Clearing LOST needs a CLEAR full-frame sighting: a 0.36 tile box on
            # a lane stripe once "FOUND" a lost swimmer (false all-clear)
            clear = d["score"] >= self.confirm_score and not d.get("tile_only")
        if clear:
            tr["last_strong"] = t  # weak matches can't keep a ghost alive
            if tr["state"] == "LOST":
                self._event(t, "FOUND", tr["id"], "LOST swimmer seen clearly again")
                tr["state"] = "UNSEEN"
        tr["moved"] = max(tr["moved"], ((center[0] - tr["origin"][0]) ** 2
                                        + (center[1] - tr["origin"][1]) ** 2) ** 0.5)
        # last 10 s of sightings, for promoting an unverified ID (below)
        tr["trail"] = [p for p in tr.get("trail", []) if t - p[0] <= 10.0]
        tr["trail"].append((t, center[0], center[1], d["score"], center[2]))
        if tr["confirmed"] and not tr["verified"] and tr["zone"] == "pool":
            self._maybe_verify(tr, t)
        tr["hist"].append(True)
        if (not tr["confirmed"] and sum(tr["hist"][-3:]) >= 2
                and tr["best"] >= self.confirm_score):
            tr["confirmed"] = True
            if t - self.t0 <= self.startup_s:
                tr["verified"] = True  # was already there when Spottr started
            elif d["zone"] == "pool" and tr["born"] == "deck":
                # born as a "deck" box but confirmed in the water without an
                # ENTER: a piece of someone at the camera (a crouching body
                # became a counted ghost swimmer this way). Must earn it.
                tr["verified"] = False
            self._event(t, "NEW", tr["id"], f"on the {d['zone']}"
                        + ("" if tr["verified"] or d["zone"] == "deck" else " (unverified)"))
            tr["zone"] = d["zone"]
        # a zone change must hold for zone_hold_s (people at the pool edge jitter),
        # and only a CLEAR detection can move someone between pool and deck
        # (a weak box in the water under a deck person is often their reflection)
        if d["zone"] != tr["zone"] and d["score"] >= self.create[d["zone"]]:
            if not (tr["pending"] and tr["pending"][0] == d["zone"]):
                tr["pending"] = (d["zone"], t)
            # ... and the track must really have gone there. ENTER (into
            # the pool) needs solid motion: a morphing box is not a jump-in.
            # EXIT keeps the travel check: climbing out genuinely changes
            # the box shape. Static ghosts hold pending forever.
            # EXIT: travel, OR the box stood up (2x taller, now upright): a
            # far swimmer climbing out barely moves in the picture but goes
            # from a head box to a standing body. A swimmer resting at the
            # wall doesn't double in height, so this can't drop them.
            bw, bh = d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]
            stood_up = self._size_ratio(tr) >= 2.0 and bh > 1.3 * bw
            solid = (self._solid(tr) if d["zone"] == "pool"
                     else self._travel(tr) >= self.enter_travel or stood_up)
            if t - tr["pending"][1] >= self.zone_hold_s and solid:
                if tr["confirmed"] and not tr["verified"]:
                    # an unverified pool ID was never counted, so its moving
                    # to the deck is not an exit; it is a real deck person now
                    tr["verified"] = True
                elif tr["confirmed"]:
                    if d["zone"] == "pool":
                        self.entries += 1
                        tr["verified"] = True
                        tr["fast_until"] = t + 3.0
                        tr["entered_at"] = t
                        self._event(t, "ENTER", tr["id"], "deck -> pool")
                    else:
                        self.exits += 1
                        self._event(t, "EXIT", tr["id"], "pool -> deck")
                tr["zone"], tr["pending"] = d["zone"], None
                tr["last_move"], tr["still"], tr["still_s"] = t, False, 0.0
                tr["last_surface"] = t
        elif d["zone"] == tr["zone"]:
            tr["pending"] = None
        if tr["zone"] == "pool":
            # SUBMERGED = clearly below THIS swimmer's own surface score (a far
            # swimmer scores lower just for being small; a fixed bar would call
            # every far swimmer "under")
            bar = min(0.50, 0.70 * tr["surface_ref"])
            tr["head"] = self._head_score(d["box"])
            tr["head_hist"] = (tr.get("head_hist", []) + [tr["head"]])[-3:]
            tr["head_best"] = max(tr.get("head_best", 0.0), tr["head"])
            # at the SURFACE = body at this swimmer's surface level AND (when
            # this run asks for heads) a clear head on them in any of the last
            # 3 sightings. Heads are missed on 10-25% of clear sightings at
            # random; needing one EVERY check made 12 s false timers.
            # Stripes never get a head at all, so they still can't reset it.
            up = d["score"] >= bar and (not self.has_heads
                                        or max(tr["head_hist"]) >= self.head_bar)
            tr["surf_hist"] = (tr.get("surf_hist", []) + [up])[-3:]
            if tr["state"] == "LOST":
                pass  # stays LOST until a CLEAR sighting (handled above)
            elif up and sum(tr["surf_hist"]) >= 2:
                # back at the SURFACE only after 2 of the last 3 sightings:
                # one splash-boosted score must not reset the under-water clock
                tr["state"], tr["last_surface"] = "SURFACE", t
                tr["surface_ref"] = 0.8 * tr["surface_ref"] + 0.2 * d["score"]
            elif self._side_edge(d["box"]):
                # cut off by the side of the frame: a low score here means
                # "half out of view", not "under water"
                tr["state"] = "OUTVIEW"
            else:
                tr["state"] = "SUBMERGED"  # incl. one surface sighting not yet confirmed
            tr["under_s"] = t - tr["last_surface"]
        else:
            tr["state"], tr["under_s"] = "DECK", 0.0
        self._stillness(tr, center, t, gap)

    def _stillness(self, tr, center, t, gap):
        """still_s = seconds since this swimmer last moved (the face-down test).

        Movement is judged from the trail: the middle of the last 3
        sightings vs the middle of everything older in the window. One
        jittery box can't reset the timer, real swimming shifts it within
        2 checks. Reappearing somewhere else after a gap also counts.
        Only confirmed VERIFIED pool swimmers raise STILL (UNSEEN/LOST
        freeze the timer: motion is unknown then)."""
        if tr["zone"] != "pool":
            tr["last_move"], tr["still"], tr["still_s"] = t, False, 0.0
            return
        bar = max(self.still_floor, self.still_size_k * max(center[2], 0.02))
        moved = False
        if gap > 10.0:
            # out of sight for a long time: motion unknown, so the still
            # clock restarts (it said "still 135 s" right after a FOUND)
            moved = True
        elif gap > 1.0:
            px, py = tr.get("sm_cx", center[0]), tr.get("sm_cy", center[1])
            moved = ((center[0] - px) ** 2 + (center[1] - py) ** 2) ** 0.5 > bar
        a = 0.4
        tr["sm_cx"] = a * center[0] + (1 - a) * tr.get("sm_cx", center[0])
        tr["sm_cy"] = a * center[1] + (1 - a) * tr.get("sm_cy", center[1])
        if not moved:
            pts = [(x, y) for (ts, x, y, _, _) in tr.get("trail", [])
                   if t - ts <= self.still_window]
            if len(pts) >= self.still_min_n:
                old, new = pts[:-3], pts[-3:]
                med = lambda ps, i: sorted(p[i] for p in ps)[len(ps) // 2]  # noqa: E731
                travel = ((med(new, 0) - med(old, 0)) ** 2
                          + (med(new, 1) - med(old, 1)) ** 2) ** 0.5
                moved = travel > bar
        if moved:
            if tr.get("still"):
                tr["still"] = False
                if tr["confirmed"] and tr.get("verified"):
                    self._event(t, "MOVING", tr["id"],
                                f"moving again after {tr.get('still_s', 0.0):.0f}s still")
            tr["last_move"] = t
        tr["still_s"] = t - tr.get("last_move", t)
        if (not tr.get("still") and tr["confirmed"] and tr.get("verified")
                and tr["state"] in ("SURFACE", "SUBMERGED")):
            fuse = self.still_alarm_s if tr["state"] == "SUBMERGED" else self.still_warn_s
            # a clear head up = resting (holding the wall, standing in the
            # shallow end), not the face-down case: no STILL alarm
            head_up = (self.has_heads
                       and max(tr.get("head_hist") or [0.0]) >= self.head_bar)
            if tr["still_s"] >= fuse and not head_up:
                tr["still"] = True
                self._event(t, "STILL", tr["id"],
                            f"not moving for {tr['still_s']:.0f}s ({tr['state'].lower()})")

    def _head_score(self, box):
        """Best "head" score on this person: a head box whose center is inside
        their box (grown 10%, a head can poke out of a tight swimmer box) and
        that is not bigger than them. 0.0 if none."""
        gx = 0.1 * (box[2] - box[0]) + 0.005
        gy = 0.1 * (box[3] - box[1]) + 0.005
        a = (box[2] - box[0]) * (box[3] - box[1])
        best = 0.0
        for h in self._heads:
            hb = h["box"]
            cx, cy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
            if (box[0] - gx <= cx <= box[2] + gx and box[1] - gy <= cy <= box[3] + gy
                    and (hb[2] - hb[0]) * (hb[3] - hb[1]) <= 1.2 * a):
                best = max(best, h["score"])
        return best

    @staticmethod
    def _side_edge(box):
        """Box touching the left or right side of the frame."""
        return box[0] < 0.02 or box[2] > 0.98

    @staticmethod
    def _iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
        return ix / max(ua, 1e-9)

    def _handoff(self, d, center, t, used_t):
        """A new person right next to someone who vanished from the OTHER zone in
        the last few seconds is the same person climbing out / jumping in (the
        box changes shape too much - head in the water vs standing body - for
        normal matching). Keeps the ID and logs EXIT / ENTER."""
        # CLEAR boxes only: a 0.36 junk box in the bottom-right corner took the
        # ID of someone diving in; their real splash-down (0.59) came 3 s later
        if d["score"] < self.strong_score:
            return False
        bx = ((d["box"][0] + d["box"][2]) / 2, d["box"][3])  # feet / bottom-center
        best = None
        for tid, tr in self.tracks.items():
            if (tid in used_t or not tr["confirmed"] or tr["zone"] == d["zone"]
                    or not (0 < t - tr["last"] <= self.handoff_s)):
                continue
            # (No motion test on the deck side: someone standing still at
            # the camera and then diving moved only 0.045 - a size test and
            # a "must have moved" test each blocked a real jump-in. The deck
            # person must vanish AND a clear box appear in the water nearby
            # within handoff_s, which a static sign doesn't do.)
            # nobody climbs out from under the water: a swimmer last seen
            # SUBMERGED / UNSEEN / LOST is never handed to a deck person walking by
            if tr["zone"] == "pool" and (tr["state"] != "SURFACE" or not tr["verified"]):
                continue  # (an unverified ID was never counted: no exit)
            tb = ((tr["box"][0] + tr["box"][2]) / 2, tr["box"][3])
            dist = ((bx[0] - tb[0]) ** 2 + (bx[1] - tb[1]) ** 2) ** 0.5
            # someone big right at the camera jumping in lands further away in
            # the picture (seen: 0.32 of the frame): reach = their largest size
            # in the last 3 s (their box shrinks while crouching to jump)
            if tr["zone"] == "deck":
                recent = max([sz for _, sz in tr.get("sizes", [])] or [tr["size"]])
                reach = max(self.handoff_dist, 0.5 * recent)
                # (A wrong landing box - e.g. a fixture in the bottom-right
                # corner - is corrected afterwards by _fix_dive / _relink.
                # A size test failed: a real landing box was 0.09 next to a
                # deck box ~1.0 tall.)
            else:
                reach = self.handoff_dist
            if dist < reach and (best is None or dist < best[0]):
                best = (dist, tid)
        if best is None:
            return False
        tr = self.tracks[best[1]]
        used_t.add(best[1])
        if d["zone"] == "pool":
            self.entries += 1
            tr["verified"] = True
            tr["fast_until"] = t + 3.0
            tr["entered_at"] = t
            self._event(t, "ENTER", tr["id"], "jumped in (deck -> pool)")
        else:
            self.exits += 1
            self._event(t, "EXIT", tr["id"], "climbed out (pool -> deck)")
        tr.update(zone=d["zone"], pending=None, last_surface=t, last_move=t,
                  still=False, still_s=0.0,
                  cx=center[0], cy=center[1], size=center[2])
        tr["last"] = t - 1e-6  # so _hit doesn't log a BACK for the hand-off
        tr["confirmed"] = True
        self._hit(tr, d, center, t)
        return True

    def _travel(self, tr):
        """How far this track has moved: median of the newer half of the
        trail vs median of the older half (0.0 with fewer than 3
        sightings). Jitter-proof: one bad box can't shift a median."""
        p = tr.get("trail", [])
        if len(p) < 3:
            return 0.0
        h = len(p) // 2
        old, new = p[:h], p[h:]
        med = lambda ps, i: sorted(q[i] for q in ps)[len(ps) // 2]  # noqa: E731
        return ((med(new, 1) - med(old, 1)) ** 2
                + (med(new, 2) - med(old, 2)) ** 2) ** 0.5

    def _size_ratio(self, tr):
        """Biggest / smallest box size on the trail (1.0 = rock steady)."""
        sizes = [p[4] for p in tr.get("trail", [])]
        return max(sizes) / max(min(sizes), 1e-9) if sizes else 1.0

    def _solid(self, tr):
        """True when this track really moved here: enough trail travel
        with a stable box size (trail points carry (t, cx, cy, score,
        size)). A collapsing giant box 'travels' as it shrinks."""
        return (self._travel(tr) >= self.enter_travel
                and self._size_ratio(tr) < self.enter_size_ratio)

    def _maybe_verify(self, tr, t):
        """Promote an UNVERIFIED pool ID (never seen entering: swam in from a
        side of the frame, missed jump-in, under water at startup) to a counted
        swimmer. This can only ADD swimmers, never remove one. Two signs:
          SWAM IN     travelled >= 0.20 of the frame within 10 s, without
                      morphing (a collapsing box "travels" too). Lane stripes
                      hop between stripe spots but stayed <= 0.11.
          STEADY      seen for >= 5 s with a median score >= 0.55. Stripes
                      average 0.30-0.44 (single sightings up to 0.72), people
                      0.56-0.74.
        Far swimmers move only ~0.06 in 10 s, so for them only STEADY works."""
        p = tr["trail"]
        if len(p) < 10 or p[-1][0] - p[0][0] < 5.0:
            return
        a = (sorted(q[1] for q in p[:3])[1], sorted(q[2] for q in p[:3])[1])
        b = (sorted(q[1] for q in p[-3:])[1], sorted(q[2] for q in p[-3:])[1])
        travel = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        med = sorted(q[3] for q in p)[len(p) // 2]
        swam = (travel >= 0.20
                and self._size_ratio(tr) < self.enter_size_ratio)
        why = (f"swam {travel:.2f} of the frame in 10 s" if swam else
               f"seen steadily for {p[-1][0] - p[0][0]:.0f}s (median score {med:.2f})"
               if med >= 0.55 else None)
        # When the server sends heads: a clear head at least once. On the
        # empty pool and on stripes "head" never fired (0 of 386 checks).
        if why and self.has_heads and tr.get("head_best", 0.0) < self.head_bar:
            why = None
        if why:
            tr["verified"] = True
            self._event(t, "VERIFIED", tr["id"], why)

    def _part_of(self, box, center, tr):
        """Is a new box just a piece of person `tr` (splash, arm, a second box)?
        Close to them (at most 0.10 of the frame, whatever their size), or mostly
        inside their box. Before, the distance grew with their box size: someone
        standing at the camera (box ~ the whole frame) stopped anyone new from
        appearing anywhere."""
        dist = ((center[0] - tr["cx"]) ** 2 + (center[1] - tr["cy"]) ** 2) ** 0.5
        if dist < min(0.10, self.near_factor * max(center[2], tr["size"], 0.02)):
            return True
        q = tr["box"]
        a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        ix = (max(0.0, min(box[2], q[2]) - max(box[0], q[0]))
              * max(0.0, min(box[3], q[3]) - max(box[1], q[1])))
        return ix / max(a, 1e-9) > 0.5

    def _fix_dive(self, t):
        """Right after a jump-in the hand-off can pick the wrong box (a piece
        of the diver still at the camera, a corner fixture) while the real
        splash-down starts a new, UNVERIFIED ID. If the jumper's ID has
        vanished and such a new ID appeared nearby around the dive, the new
        one IS the jumper: it takes over the jumper's ID (one person, one
        count). Only within 8 s of the dive and 0.45 of the frame."""
        for tid, tr in list(self.tracks.items()):
            ent = tr.get("entered_at")
            if (ent is None or t - ent > 8.0 or tr["zone"] != "pool"
                    or t - tr["last"] < 0.5):
                continue
            best = None
            for uid, u in self.tracks.items():
                # an unverified ID seen CLEARLY right now (>= confirm_score):
                # the splash-down, or the diver surfacing on top of an old
                # stripe ID (stripes stay below that; a 0.23 stripe at the
                # side edge was wrongly merged when birth time was the test)
                if (uid == tid or u["verified"] or u["zone"] != "pool"
                        or u["last"] != t or u["score"] < self.confirm_score):
                    continue
                dist = ((u["cx"] - tr["cx"]) ** 2 + (u["cy"] - tr["cy"]) ** 2) ** 0.5
                if dist < 0.45 and (best is None or dist < best[0]):
                    best = (dist, uid)
            if best is None:
                continue
            u = self.tracks.pop(best[1])
            for k in ("cx", "cy", "size", "box", "score", "last", "last_strong",
                      "state", "trail", "sizes", "surf_hist", "head", "head_hist",
                      "sm_cx", "sm_cy", "last_move", "still_s"):
                if k in u:
                    tr[k] = u[k]
            # the most recent surface time wins: the merged ID may be an old
            # stripe ID that "was never at the surface" (made a fake 31 s)
            tr["last_surface"] = max(tr["last_surface"], u["last_surface"])
            tr["under_s"] = t - tr["last_surface"]
            tr["best"] = max(tr["best"], u["best"])
            self._event(t, "DIVE", tid, f"splash-down found (was new ID {best[1]})")

    def _relink(self, d, center, t, used_t):
        """A CLEAR new box in the water close to a VERIFIED swimmer who just
        dropped out of view is that swimmer coming back up (often after
        swimming under water), not a new person: keep their ID. Without this
        the swimmer got a second, unverified ID and the old ID stayed behind as
        a ghost. Only clear full-frame boxes (>= confirm_score) can re-link, so
        lane stripes (0.2-0.5) can't take over a missing swimmer's ID."""
        if (d["zone"] != "pool" or d.get("tile_only")
                or d["score"] < self.confirm_score):
            return False
        best = None
        for tid, tr in self.tracks.items():
            if (tid in used_t or not tr["confirmed"] or not tr["verified"]
                    or tr["zone"] != "pool"):
                continue
            gap = t - tr["last"]
            # how far they can have swum under water: grows with time away
            reach = min(self.max_jump + 0.10, 0.10 + 0.03 * gap)
            if t - tr.get("entered_at", -1e9) < 8.0:
                # just jumped in: the hand-off often lands on a piece of them
                # still at the camera (crouching legs), and the real
                # splash-down comes up to 0.41 away a second later
                reach = 0.45
            dist = ((center[0] - tr["cx"]) ** 2 + (center[1] - tr["cy"]) ** 2) ** 0.5
            if dist < reach and (best is None or dist < best[0]):
                best = (dist, tid)
        if best is None:
            return False
        # Needs 2 clear sightings in the same spot (a lane stripe once hit
        # 0.72 once): the first one is only remembered (and starts no new
        # ID), the second within 1.5 s re-links.
        tid = best[1]
        w = self._relink_wait.get(tid)
        if not (w and t - w[2] <= 1.5
                and ((center[0] - w[0]) ** 2 + (center[1] - w[1]) ** 2) ** 0.5 < 0.06):
            self._relink_wait[tid] = (center[0], center[1], t)
            return True
        self._relink_wait.pop(tid, None)
        used_t.add(tid)
        self._hit(self.tracks[tid], d, center, t)
        return True

    def counts(self, t):
        live = [tr for tr in self.tracks.values() if tr["confirmed"]]
        pool = [tr for tr in live if tr["zone"] == "pool" and tr["verified"]]
        unverified = [tr for tr in live if tr["zone"] == "pool" and not tr["verified"]]
        deck = [tr for tr in live if tr["zone"] == "deck" and t - tr["last"] < 2.0]
        return {"pool": len(pool), "deck": len(deck), "unverified": len(unverified),
                "lost": sum(tr["state"] == "LOST" for tr in pool),
                "still": sum(tr.get("still") and tr["state"] in ("SURFACE", "SUBMERGED")
                             for tr in pool),
                # under/unseen for 3+ s: shorter gaps are normal (a far swimmer
                # is missed ~1 check in 3); the alarm levels start at 10 s anyway
                "under": sum(tr["state"] in ("SUBMERGED", "UNSEEN", "LOST")
                             and tr["under_s"] >= 3.0 for tr in pool),
                # at / past a side edge of the frame: timer runs, shown apart
                "outview": sum(tr["state"] == "OUTVIEW" for tr in pool),
                "entries": self.entries, "exits": self.exits}


def load_zone(path):
    """Pool outline as a list of [x, y] fractions, or None if there is none."""
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return None
    pts = json.loads(p.read_text()).get("points", [])
    return pts if len(pts) >= 3 else None


def set_zone(cap, path):
    """Shows one camera frame; you click the corners of the water, Enter saves."""
    import json

    import cv2
    import numpy as np

    ok, frame = False, None
    for _ in range(30):  # skip the first frames, cameras often start grey
        ok, f = cap.read()
        if ok:
            ok, frame = True, f
    if frame is None:
        print("Could not read a frame from the source.")
        return
    H, W = frame.shape[:2]
    scale = min(1.0, 1400 / W)  # fit the window on a laptop screen
    view = cv2.resize(frame, (int(W * scale), int(H * scale)))
    pts = []
    win = "Click the WATER corners - Enter=save  u=undo  Esc=cancel"

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([round(x / view.shape[1], 4), round(y / view.shape[0], 4)])

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    while True:
        img = view.copy()
        px = [(int(x * view.shape[1]), int(y * view.shape[0])) for x, y in pts]
        for p in px:
            cv2.circle(img, p, 5, (0, 255, 255), -1)
        if len(px) >= 2:
            cv2.polylines(img, [np.array(px)], len(px) >= 3,
                          (255, 200, 0), 2)
        cv2.imshow(win, img)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
        elif k == 27:
            print("Cancelled, nothing saved.")
            break
        elif k in (13, 10):
            if len(pts) < 3:
                print("Need at least 3 points.")
                continue
            path.write_text(json.dumps({"points": pts}, indent=1))
            print(f"Saved {path} ({len(pts)} points)")
            break
    cv2.destroyAllWindows()


def run_client():
    import argparse
    import asyncio
    import json
    import os
    import struct
    import threading
    import time
    from pathlib import Path

    import cv2
    import numpy as np
    import websockets

    ap = argparse.ArgumentParser(description="Spottr live client")
    ap.add_argument("--url", default="", help="the .modal.run URL from `modal serve`")
    ap.add_argument("--source", default=os.environ.get("SPOTTR_CAMERA", ""),
                    help="RTSP address or a video file (default: SPOTTR_CAMERA in .env)")
    ap.add_argument("--prompt", default="person,head",
                    help='"person,head" (default: heads vet the under-water timer) '
                         'or "person" (body score only)')
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="score needed OUTSIDE the pool outline (deck)")
    ap.add_argument("--pool-threshold", type=float, default=0.30,
                    help="score needed INSIDE the pool outline (lower, so faint "
                         "swimmers under the water still count)")
    ap.add_argument("--zone", default="pool_zone.json", help="pool outline file")
    ap.add_argument("--set-zone", action="store_true",
                    help="click the pool outline on a camera frame, save it, and exit")
    # SAM works at 1008 px. With tiles, each tile is ~0.575 of the frame, so a
    # 2016 px frame gives every tile full detail. Without tiles 1008 is enough.
    ap.add_argument("--max-width", type=int, default=2016, help="frames are shrunk to this width")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality 1-100")
    ap.add_argument("--no-tiles", action="store_true",
                    help="whole frame only (faster, misses more faint swimmers)")
    ap.add_argument("--streams", type=int, default=7,
                    help="parallel GPU connections (each one = its own H100 on Modal)")
    ap.add_argument("--inflight", type=int, default=1,
                    help="frames on their way per connection")
    ap.add_argument("--sustain-pool", type=float, default=0.20,
                    help="lowest score that keeps an EXISTING swimmer alive in the pool")
    ap.add_argument("--sustain-deck", type=float, default=0.35,
                    help="lowest score that keeps an EXISTING person alive on the deck")
    ap.add_argument("--keep-pool", type=float, default=60.0,
                    help="seconds an unseen swimmer stays counted in the pool")
    ap.add_argument("--still-warn", type=float, default=30.0,
                    help="still seconds at the SURFACE before a STILL warning")
    ap.add_argument("--still-alarm", type=float, default=15.0,
                    help="still seconds while SUBMERGED before a STILL alarm")
    ap.add_argument("--keep", type=float, default=8.0,
                    help="seconds an unseen deck person stays remembered")
    ap.add_argument("--record", default="", help="optional: save the live view, e.g. live.mp4")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="auto-stop after N minutes (0 = no limit, default 0)")
    ap.add_argument("--no-sound", action="store_true", help="no alarm beeps")
    ap.add_argument("--no-muse", action="store_true",
                    help="no Muse Spark second opinion / report (even with a key)")
    ap.add_argument("--feed-port", type=int, default=8765,
                    help="local WebSocket feed for the frontend (0 = off)")
    ap.add_argument("--no-show", action="store_true",
                    help="headless mode: no video window (for SSH / servers)")
    args = ap.parse_args()
    if not args.source:
        ap.error("--source is required (or put SPOTTR_CAMERA=rtsp://... in .env)")
    if not ACCESS_TOKEN:
        ap.error("SPOTTR_TOKEN is missing: add SPOTTR_TOKEN=<long random string> to .env")
    try:  # footage retention (demo: 3 days); test data in diag/keep.txt is kept
        import spottr_retention
        spottr_retention.apply(days=3.0)
    except Exception as e:  # noqa: BLE001 - never stop a live session over cleanup
        print(f"Retention check failed: {e}")
    prompts = [p.strip() for p in args.prompt.split(",") if p.strip()]

    # ---- camera reader thread: always holds only the NEWEST frame ------------
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    is_file = Path(args.source).exists()
    cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Could not open {args.source}")
        return

    # ---- pool outline ----------------------------------------------------------
    zone_path = Path(args.zone)
    if args.set_zone:
        set_zone(cap, zone_path)
        cap.release()
        return
    if not args.url:
        print("--url is required (except with --set-zone)")
        return
    zone = load_zone(zone_path)
    if zone is None:
        print(f"WARNING: no pool outline ({zone_path}). The whole frame counts as pool, "
              f"so deck people will be counted. Run with --set-zone first.")
    else:
        print(f"Pool outline: {zone_path} ({len(zone)} points). "
              f"In pool >= {args.pool_threshold}, deck >= {args.threshold}")
    zone_np = None if zone is None else np.array(zone, dtype=np.float32)

    def in_pool(box):
        """Bottom-center of the box inside the outline (fractions of the frame)."""
        if zone_np is None:
            return True
        x1, y1, x2, y2 = box
        pt = ((x1 + x2) / 2, min(y2, 0.999))
        return cv2.pointPolygonTest(zone_np, pt, False) >= 0
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    latest = {"frame": None, "t": 0.0, "n": 0, "done": False}
    lock = threading.Lock()

    def reader():
        start = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                latest["done"] = True
                break
            with lock:
                latest.update(frame=frame, t=time.perf_counter(), n=latest["n"] + 1)
            if is_file:  # play a file at normal speed, like a real camera
                due = start + latest["n"] / file_fps
                time.sleep(max(0.0, due - time.perf_counter()))

    # started once the GPUs are connected, so a video file isn't half over
    # by the time the first GPU has warmed up
    reader_thread = threading.Thread(target=reader, daemon=True)

    tracker = Tracker(in_pool, create_pool=max(args.pool_threshold, 0.35),
                      create_deck=args.threshold, create_tile=0.50,
                      sustain_pool=args.sustain_pool, sustain_deck=args.sustain_deck,
                      keep_pool=args.keep_pool, keep_deck=args.keep,
                      still_warn_s=args.still_warn, still_alarm_s=args.still_alarm)
    # ---- alarms, snapshots, Muse second opinion -------------------------------
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime as _dt

    import spottr_muse
    from spottr_alarms import ALARM, EMERGENCY, NAMES, WATCH, AlarmEngine, Beeper

    alarms = AlarmEngine()
    beeper = None if args.no_sound else Beeper(alarms)
    session = _dt.now().strftime("%Y%m%d_%H%M%S")
    snap_dir = Path("results", "incidents", session)
    snapshots, muse_log = [], []
    snap_count = {}           # tid -> snapshots saved (max 5 per swimmer)
    muse_on = spottr_muse.available() and not args.no_muse
    muse_pool = ThreadPoolExecutor(max_workers=2) if muse_on else None
    muse_asked = {}           # tid -> t_obs of the last question (one per 10 s)
    print("Muse Spark second opinion: " + ("ON" if muse_on else
          "off (no MODEL_API_KEY in .env)" if not args.no_muse else "off (--no-muse)"))

    def crop_jpeg(img, box, pad=0.6):
        """JPEG of one swimmer with some water around them (for Muse / snapshots)."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = max(x2 - x1, 0.03), max(y2 - y1, 0.03)
        xa, xb = int(max(0, x1 - pad * bw) * w), int(min(1, x2 + pad * bw) * w)
        ya, yb = int(max(0, y1 - pad * bh) * h), int(min(1, y2 + pad * bh) * h)
        c = img[ya:max(yb, ya + 2), xa:max(xb, xa + 2)]
        if c.shape[1] < 384:  # far swimmers are tiny: upscale so Muse can see
            k = 384 / max(c.shape[1], 1)
            c = cv2.resize(c, (384, int(c.shape[0] * k)), interpolation=cv2.INTER_CUBIC)
        return cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()

    def ask_muse(tid, crop, full, reason, t_obs):
        r = spottr_muse.second_opinion(crop, full, tid, reason)
        if r:
            alarms.muse_opinion(tid, r["verdict"], r.get("reason", ""), t=t_obs)
            muse_log.append({"t": round(t_obs, 2), "id": tid, **r})
            print(f"  [{t_obs:7.1f}s] MUSE      person {tid}: {r['verdict']} - {r.get('reason', '')}")

    # ---- local feed for the frontend (ws://127.0.0.1:<feed-port>) ------------
    # Each check sends 2 messages to every connected page:
    #   1. binary: the drawn 1280 px frame as JPEG
    #   2. text:   {"type": "state", "t", "counts", "swimmers", "alarms", "events"}
    # The page can send {"cmd": "ack", "id": N} or {"cmd": "ack_all"}.
    feed_clients = set()

    async def feed_handler(ws):
        feed_clients.add(ws)
        try:
            async for msg in ws:
                try:
                    m = json.loads(msg)
                except (ValueError, TypeError):
                    continue
                if m.get("cmd") == "ack" and "id" in m:
                    alarms.ack(int(m["id"]))
                elif m.get("cmd") == "ack_all":
                    for tid in list(alarms.state):
                        alarms.ack(tid)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            feed_clients.discard(ws)

    async def _broadcast(jpg, text):
        for ws in list(feed_clients):
            try:
                await ws.send(jpg)
                await ws.send(text)
            except websockets.exceptions.ConnectionClosed:
                feed_clients.discard(ws)

    def feed_send(frame, counts, swimmers, t_obs):
        if not feed_clients:
            return
        state = {"type": "state", "t": round(t_obs, 2), "counts": counts,
                 "swimmers": swimmers,
                 "alarms": [{"id": tid, "level": NAMES[s["level"]], "reason": s["reason"],
                             "since": round(s["since"], 1), "acked": s["acked"],
                             "border": alarms.border(tid, t_obs)}
                            for tid, s in alarms.state.items() if s["level"] > 0],
                 "sounding": alarms.sounding(),
                 "events": (tracker.events[-15:] + alarms.events[-15:])}
        jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        asyncio.get_running_loop().create_task(_broadcast(jpg, json.dumps(state)))

    stats = {"rtt": [], "gpu": [], "frames": 0, "skipped": 0, "errors": 0,
             "recorded": 0, "stale": 0}
    last_done = {"n": 0}  # newest frame already shown: older answers are stale
    log = []
    writer = None
    # Recording frame rate: match the source so the saved video plays in real
    # time (file fps for files, camera fps otherwise, clamped to 1-30).
    _src_fps = file_fps if is_file else (cap.get(cv2.CAP_PROP_FPS) or 10.0)
    rec_fps = min(30.0, max(1.0, float(_src_fps)))
    rec_state = {"start": None}

    ws_url = args.url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
    # The server returns everything above the LOWER threshold; the laptop then
    # applies the pool / deck threshold per box.
    server_thr = min(args.threshold, args.pool_threshold, args.sustain_pool,
                     args.sustain_deck)
    ws_url += (f"/ws?token={ACCESS_TOKEN}&prompts={','.join(prompts)}"
               f"&threshold={server_thr}&tiles={0 if args.no_tiles else 1}")

    async def loop():
        n_conn = max(1, args.streams)
        print(f"Connecting {n_conn} GPU stream(s) (first time can take 1-2 min "
              f"while each GPU starts)...")
        conns = await asyncio.gather(*[
            websockets.connect(ws_url, max_size=None, open_timeout=300,
                               ping_interval=20, ping_timeout=120)
            for _ in range(n_conn)])
        feed_server = None
        if args.feed_port:
            feed_server = await websockets.serve(feed_handler, "127.0.0.1", args.feed_port,
                                                 max_size=2 ** 20)
            print(f"Frontend feed: ws://127.0.0.1:{args.feed_port}")
        reader_thread.start()
        try:
            t_start = time.perf_counter()
            max_s = (args.max_minutes or 0.0) * 60.0
            print(f"Connected: {n_conn} stream(s), {args.inflight} frame(s) in flight each, "
                  f"tiles {'off' if args.no_tiles else 'on'}"
                  + (f", auto-stop in {args.max_minutes:g} min" if max_s > 0 else "")
                  + ("" if args.no_show else ". Press q in the video window to stop."))
            # per connection: frame number -> (frame, time grabbed), not answered yet
            inflights = [{} for _ in conns]
            state = {"stop": False, "sent_all": False}

            def _time_up():
                return bool(max_s) and (time.perf_counter() - t_start) >= max_s

            async def sender():
                # Always sends the NEWEST frame to whichever connection has
                # room, so all GPUs work at the same time and delay never builds up.
                last_n = 0
                while not state["stop"]:
                    if _time_up():
                        print(f"Time limit reached ({args.max_minutes:g} min), stopping.")
                        state["stop"] = True
                        return
                    with lock:
                        frame, t_cap, n, done = (latest["frame"], latest["t"],
                                                 latest["n"], latest["done"])
                    free = [i for i, f in enumerate(inflights) if len(f) < args.inflight]
                    if frame is None or n == last_n or not free:
                        if done and n == last_n:
                            state["sent_all"] = True
                            return
                        await asyncio.sleep(0.003)
                        continue
                    ci = min(free, key=lambda i: len(inflights[i]))
                    ws, inflight = conns[ci], inflights[ci]
                    stats["skipped"] += max(0, n - last_n - 1)  # dropped on purpose
                    last_n = n
                    h, w = frame.shape[:2]
                    if w > args.max_width:
                        frame = cv2.resize(frame, (args.max_width, int(h * args.max_width / w)))
                    ok, jpg = cv2.imencode(".jpg", frame,
                                           [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                    if not ok:
                        stats["errors"] += 1
                        print(f"Frame {n}: JPEG encode failed, skipped.")
                        continue
                    inflight[n] = (frame, t_cap)
                    try:
                        await ws.send(struct.pack("!Id", n, t_cap) + jpg.tobytes())
                    except websockets.exceptions.ConnectionClosed:
                        print(f"GPU stream {ci} closed.")
                        state["stop"] = True
                        return

            async def receiver(ws, inflight):
                while not state["stop"]:
                    if _time_up():
                        state["stop"] = True
                        return
                    if state["sent_all"] and not inflight:
                        return
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        print("Server connection closed.")
                        state["stop"] = True
                        return
                    try:
                        res = json.loads(msg)
                    except (ValueError, TypeError):
                        stats["errors"] += 1
                        print("Bad reply from server (not JSON), skipped.")
                        continue
                    answered = inflight.pop(res.get("id"), None)
                    if answered is None:
                        # Unknown or late reply (e.g. error for an
                        # unparsable frame id). Count it, keep going.
                        stats["errors"] += 1
                        continue
                    frame, t_cap = answered
                    if res.get("error"):
                        stats["errors"] += 1
                        t_rel = time.perf_counter() - t_start
                        log.append({"t": round(t_rel, 3), "id": res.get("id"),
                                    "error": res["error"]})
                        print(f"Frame {res.get('id')}: server error: {res['error']}")
                        continue
                    # answers from several GPUs can arrive out of order:
                    # never step the tracker backwards in time
                    if res["id"] <= last_done["n"]:
                        stats["stale"] += 1
                        continue
                    last_done["n"] = res["id"]
                    if handle(res, frame, t_cap, t_start):
                        state["stop"] = True

            await asyncio.gather(sender(), *[receiver(w, f) for w, f in zip(conns, inflights)])
        finally:
            for w in conns:
                await w.close()
            if feed_server:
                feed_server.close()

    def handle(res, frame, t_cap, t_start):
        """Tracks + draws one answered frame. Returns True when q is pressed."""
        nonlocal writer
        t_now = time.perf_counter()

        delay_ms = (t_now - t_cap) * 1000  # frame grabbed -> result back
        stats["rtt"].append(delay_ms)
        stats["gpu"].append(res["gpu_ms"])
        stats["frames"] += 1
        t_rel = t_now - t_start

        # ---- track ------------------------------------------------
        # the tracker runs on the time the frame was TAKEN (answers from
        # several GPUs arrive 0.4-1 s late, by different amounts)
        t_obs = t_cap - t_start
        tracks = tracker.update(res["dets"], t_obs)
        counts = tracker.counts(t_obs)

        # ---- alarms -------------------------------------------------
        full = frame  # full-resolution frame, for snapshots and Muse
        changes = alarms.update(tracker, t_obs)
        for c in changes:
            print(f"  [{t_obs:7.1f}s] {c['to']:9s} person {c['id']}: {c['reason']}")
            # snapshot on every step UP (max 5 per swimmer): for the report
            tr = tracker.tracks.get(c["id"])
            if (tr and NAMES[alarms.level(c["id"])] == c["to"] and c["to"] != "OK"
                    and snap_count.get(c["id"], 0) < 5):
                snap_dir.mkdir(parents=True, exist_ok=True)
                p = snap_dir / f"{t_obs:07.1f}s_id{c['id']}_{c['to']}.jpg"
                p.write_bytes(crop_jpeg(full, tr["box"], pad=2.0))
                snapshots.append(str(p))
                snap_count[c["id"]] = snap_count.get(c["id"], 0) + 1
        if muse_pool:  # second opinion on anyone at WATCH or above, every 10 s
            for tid, s in alarms.state.items():
                if s["level"] >= WATCH and t_obs - muse_asked.get(tid, -1e9) >= 10.0:
                    tr = tracker.tracks.get(tid)
                    if tr:
                        muse_asked[tid] = t_obs
                        small = cv2.resize(full, (1280, int(full.shape[0] * 1280 / full.shape[1])))
                        muse_pool.submit(ask_muse, tid, crop_jpeg(full, tr["box"]),
                                         cv2.imencode(".jpg", small)[1].tobytes(),
                                         s["reason"], t_obs)

        # ---- draw (on a 1280 px copy, so the window and recording stay small)
        fh, fw = frame.shape[:2]
        if fw > 1280:
            frame = cv2.resize(frame, (1280, int(fh * 1280 / fw)))
        H, W = frame.shape[:2]
        if zone_np is not None:
            pts = (zone_np * [W, H]).astype(np.int32)
            cv2.polylines(frame, [pts], True, (255, 200, 0), 2)
        colors = {"SURFACE": (0, 220, 0), "SUBMERGED": (0, 200, 255),
                  "UNSEEN": (0, 0, 255), "LOST": (0, 0, 255), "DECK": (190, 190, 190),
                  "OUTVIEW": (255, 160, 0)}  # light blue: at / past the side of the frame
        frame_log = []
        for tr in tracks:
            x1, y1, x2, y2 = tr["box"]
            p1, p2 = (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H))
            unverified = tr["zone"] == "pool" and not tr["verified"]
            # unverified (never seen entering, e.g. a lane stripe): purple
            color = (200, 80, 200) if unverified else colors[tr["state"]]
            if not tr["confirmed"]:  # not yet confirmed (seen once): thin
                cv2.rectangle(frame, p1, p2, color, 1)
                continue
            unseen = tr["state"] in ("UNSEEN", "LOST") or tr["last"] < t_obs
            cv2.rectangle(frame, p1, p2, color, 3 if tr["state"] == "LOST" else
                          (1 if unseen else 2))
            label = f"{tr['id']} {'unverified ' if unverified else ''}{tr['state'].lower()}"
            still = (tr["zone"] == "pool" and tr.get("still")
                     and tr["state"] in ("SURFACE", "SUBMERGED"))
            if tr["zone"] == "pool" and tr["state"] != "SURFACE":
                label += f" {tr['under_s']:.0f}s"
            elif not unseen and not still:
                label += f" {tr['score']:.2f}"
            if still:
                label += f" still {tr['still_s']:.0f}s"
            lvl = alarms.level(tr["id"])
            if lvl >= WATCH:
                # alarm border around the swimmer: grows with time in alarm
                th = alarms.border(tr["id"], t_obs)
                acol = (0, 220, 255) if lvl == WATCH else (0, 0, 255)
                # drawn OUTSIDE the swimmer (gap grows with the thickness) so
                # even a tiny far swimmer stays visible inside the frame
                g = 10 + th
                cv2.rectangle(frame, (p1[0] - g, p1[1] - g), (p2[0] + g, p2[1] + g), acol, th)
                label = f"{NAMES[lvl]} {label}"
                color = acol
            lab_y = p1[1] - 6 - (10 + 2 * alarms.border(tr["id"], t_obs) if lvl >= WATCH else 0)
            cv2.putText(frame, label, (max(0, p1[0] - (10 + alarms.border(tr["id"], t_obs)
                                                       if lvl >= WATCH else 0)), max(66, lab_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            frame_log.append({"id": tr["id"], "zone": tr["zone"], "state": tr["state"],
                              "alarm": NAMES[lvl],
                              "under_s": round(tr["under_s"], 1),
                              "still_s": round(tr["still_s"], 1), "still": tr["still"],
                              "box": tr["box"],
                              "score": tr["score"], "head": round(tr.get("head", 0.0), 3),
                              "seen": not unseen,
                              "verified": tr["verified"]})
        log.append({"t": round(t_obs, 3), "delay_ms": round(delay_ms, 1),
                    "gpu_ms": res["gpu_ms"], "counts": counts, "objs": frame_log,
                    "raw": res["dets"]})  # raw detections: lets the tracker be re-tuned offline

        fps = stats["frames"] / max(t_rel, 1e-6)
        # Two lines so nothing gets cut off on a narrow window
        line1 = (f"{fps:4.1f} checks/s   delay {delay_ms:4.0f} ms   "
                 f"GPU {res['gpu_ms']:4.0f} ms")
        line2 = (f"in pool: {counts['pool']}  (under: {counts['under']}, "
                 f"still: {counts['still']}, LOST: {counts['lost']}, out of view: "
                 f"{counts['outview']})   unverified: {counts['unverified']}   "
                 f"deck: {counts['deck']}   in: {counts['entries']}  out: {counts['exits']}")
        cv2.rectangle(frame, (0, 0), (W, 52), (0, 0, 0), -1)
        cv2.putText(frame, line1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)
        cv2.putText(frame, line2, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)
        top = alarms.top()
        if top >= ALARM:
            # flashing red screen edge (2x per second) + banner with the worst case
            if int(t_rel * 4) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 0, 255),
                              24 if top >= EMERGENCY else 14)
            worst = max(alarms.state.items(), key=lambda kv: (kv[1]["level"], -kv[1]["since"]))
            msg = (f"{NAMES[worst[1]['level']]}: person {worst[0]} - {worst[1]['reason']}"
                   + ("   (a = acknowledge)" if alarms.sounding() else "   (acknowledged)"))
            cv2.rectangle(frame, (0, H - 40), (W, H), (0, 0, 180), -1)
            cv2.putText(frame, msg, (10, H - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
        log[-1]["alarm_top"] = NAMES[top]
        feed_send(frame, counts, frame_log, t_obs)
        if not args.no_show:
            cv2.imshow("Spottr live (q to quit)", frame)
        if args.record:
            if writer is None:
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                         rec_fps, (W, H))
                rec_state["start"] = t_rel
            # Stay in real time: answers arrive slower than the source fps,
            # so repeat this frame to fill the wall-clock gap (capped at 2 s
            # of copies so one long stall can't explode the file).
            target = int((t_rel - rec_state["start"]) * rec_fps) + 1
            copies = max(1, min(target - stats["recorded"], int(rec_fps * 2)))
            for _ in range(copies):
                writer.write(frame)
            stats["recorded"] += copies
        if not args.no_show:
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                return True
            if k == ord("a"):  # lifeguard acknowledges: sound stops until it gets worse
                for tid in list(alarms.state):
                    alarms.ack(tid)
        return False

    try:
        asyncio.run(loop())
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    # ---- summary ---------------------------------------------------------------
    from datetime import datetime, timezone

    def _redact(src: str) -> str:
        # rtsp://user:password@host/... -> rtsp://user:***@host/...
        try:
            if "://" in src and "@" in src:
                pre, rest = src.split("://", 1)
                creds, at, host = rest.partition("@")
                if at and ":" in creds:
                    return f"{pre}://{creds.split(':', 1)[0]}:***@{host}"
        except Exception:  # noqa: BLE001 - redaction must never break saving
            pass
        return src

    total_t = log[-1]["t"] if log else 0.0
    checks_per_s = stats["frames"] / max(total_t, 1e-6)
    print("\n===== Spottr live summary =====")
    if not stats["rtt"]:
        print("No frames were processed.")
        delay_med = delay_p95 = gpu_med = 0.0
    else:
        rtt = sorted(stats["rtt"][3:] or stats["rtt"])  # skip warm-up frames
        gpu = sorted(stats["gpu"][3:] or stats["gpu"])
        pct = lambda xs, q: xs[min(len(xs) - 1, int(q * len(xs)))]  # noqa: E731
        delay_med, delay_p95, gpu_med = pct(rtt, 0.5), pct(rtt, 0.95), pct(gpu, 0.5)
        print(f"Frames analyzed: {stats['frames']} in {total_t:.1f} s "
              f"= {checks_per_s:.1f} checks per second")
        print(f"Frames dropped on purpose (to stay live): {stats['skipped']}")
        print(f"Delay, frame grabbed -> result back: median {delay_med:.0f} ms, "
              f"worst 5% {delay_p95:.0f} ms")
        print(f"  of which GPU time: median {gpu_med:.0f} ms "
              f"(the rest is network + encoding)")
    print(f"Errors (encode / server / bad replies): {stats['errors']}")
    print(f"Stale answers dropped (arrived after a newer frame): {stats['stale']}")
    c = tracker.counts(total_t)
    print(f"People: {tracker.next_id} IDs started, entered pool {c['entries']}x, "
          f"exited {c['exits']}x, in pool at the end: {c['pool']}")
    for e in tracker.events:
        print(f"   {e['t']:7.1f}s  {e['event']:9s} person {e['id']}: {e['text']}")
    Path("results").mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "run": {
            "utc": stamp,
            "source": _redact(args.source),
            "threshold": args.threshold,
            "pool_threshold": args.pool_threshold,
            "zone": zone,
            "max_width": args.max_width,
            "quality": args.quality,
            "inflight": args.inflight,
            "keep": args.keep,
            "keep_pool": args.keep_pool,
            "sustain_pool": args.sustain_pool,
            "sustain_deck": args.sustain_deck,
            "still_warn": args.still_warn,
            "still_alarm": args.still_alarm,
            "tiles": not args.no_tiles,
            "streams": args.streams,
            "max_minutes": args.max_minutes,
            "no_show": args.no_show,
            "record": args.record,
            "rec_fps": rec_fps,
        },
        "prompts": prompts,
        "stats": {
            "frames": stats["frames"],
            "skipped": stats["skipped"],
            "errors": stats["errors"],
            "recorded": stats["recorded"],
            "total_s": round(total_t, 3),
            "checks_per_s": round(checks_per_s, 2),
            "delay_ms": {"median": round(delay_med, 1), "p95": round(delay_p95, 1)},
            "gpu_ms": {"median": round(gpu_med, 1)},
        },
        "events": tracker.events,
        "alarms": alarms.events,
        "snapshots": snapshots,
        "muse": muse_log,
        "frames": log,
    }
    text = json.dumps(payload, indent=1)
    out = Path("results") / f"live_log_{stamp}.json"
    out.write_text(text)
    Path("results", "live_log.json").write_text(text)  # latest copy
    print(f"Saved {out}" + (f" and {args.record}" if args.record else ""))
    for a in alarms.events:
        print(f"   {a['t']:7.1f}s  {a['to']:9s} person {a['id']}: {a['reason']}")
    if beeper:
        beeper.stop = True
    if muse_pool:
        muse_pool.shutdown(wait=True)  # finish pending second opinions
        spottr_muse.session_report(out)  # written report next to the log


if __name__ == "__main__":
    run_client()
