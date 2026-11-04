"""Stillness timer: a verified swimmer who stops moving must raise STILL.

A face-down floater scores high, so the score-only SURFACE/SUBMERGED logic
calls them "fine" forever. `still_s` (seconds since the swimmer last moved)
is the missing timer the drowning alarms will build on.

Run:  .venv\\Scripts\\python.exe test_stillness.py
"""

from spottr_live import Tracker


def det(cx, cy=0.88, score=0.85, w=0.10, h=0.16):
    # y2 = 0.96 > 0.95: counts as verified at birth (near edge under camera)
    return {"prompt": "person", "score": score, "tile_only": False,
            "box": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]}


def run(tr, boxes, t0=0):
    out = None
    for i, b in enumerate(boxes):
        out = tr.update([b], float(t0 + i))
    return out


def test_surface_still_raises_still():
    tr = Tracker(in_pool=lambda box: True)
    tracks = run(tr, [det(0.45)] * 31)  # t = 0..30, motionless at the surface
    me = tracks[0]
    assert me["confirmed"] and me["verified"] and me["state"] == "SURFACE"
    assert me["still"] is True and abs(me["still_s"] - 30.0) < 1e-6
    kinds = [e["event"] for e in tr.events]
    assert "STILL" in kinds, kinds
    assert tr.counts(30.0)["still"] == 1
    return tr


def test_moving_again_clears_still(tr):
    run(tr, [det(0.55), det(0.55)], t0=31)  # jump 0.10, two checks in a row
    me = tr.tracks[0]
    assert me["still"] is False and me["still_s"] == 0.0, (me["still"], me["still_s"])
    kinds = [e["event"] for e in tr.events]
    assert "MOVING" in kinds, kinds


def test_submerged_still_uses_short_fuse():
    tr = Tracker(in_pool=lambda box: True)
    run(tr, [det(0.45)] * 3, t0=0)  # confirm at the surface first
    run(tr, [det(0.45, score=0.30)] * 14, t0=3)  # then weak but seen = SUBMERGED
    me = tr.tracks[0]
    assert me["state"] == "SUBMERGED", me["state"]
    assert me["still"] is True, (me["still"], me["still_s"])
    still = [e for e in tr.events if e["event"] == "STILL"]
    assert len(still) == 1 and still[0]["t"] <= 15.0, still  # alarm fuse, not 30 s


def test_unverified_static_never_alarms():
    tr = Tracker(in_pool=lambda box: True)
    # score 0.50: tracked but never confirmed, never verified (a stripe lookalike)
    run(tr, [det(0.45, cy=0.86, score=0.50)] * 41, t0=0)
    me = tr.tracks[0]
    assert me["verified"] is False
    assert me["still"] is False
    assert not [e for e in tr.events if e["event"] in ("STILL", "MOVING")]


def test_deck_static_never_alarms():
    tr = Tracker(in_pool=lambda box: False)
    run(tr, [det(0.45, score=0.90)] * 41, t0=0)
    me = tr.tracks[0]
    assert me["state"] == "DECK"
    assert me["still"] is False and me["still_s"] == 0.0
    assert not [e for e in tr.events if e["event"] in ("STILL", "MOVING")]


if __name__ == "__main__":
    tr = test_surface_still_raises_still()
    print("PASS surface still -> STILL at 30 s, counted")
    test_moving_again_clears_still(tr)
    print("PASS moving again -> MOVING, timer reset")
    test_submerged_still_uses_short_fuse()
    print("PASS submerged still -> STILL at 15 s fuse")
    test_unverified_static_never_alarms()
    print("PASS unverified static -> no alarm")
    test_deck_static_never_alarms()
    print("PASS deck static -> no alarm")
    print("ALL STILLNESS TESTS PASSED")
