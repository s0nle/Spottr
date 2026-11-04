"""Ghost verification: lane stripes must not become counted swimmers.

A stripe scores up to 0.98, so no score bar keeps it out. Verification must
rest on entry evidence (a tracked, moving person coming from the deck) and
swimming (SWAM), never on a static appearance in the water.

Run:  .venv\\Scripts\\python.exe test_verify.py
"""

from spottr_live import Tracker


def box(cx, cy=0.50, w=0.10, h=0.16, score=0.90):
    return {"prompt": "person", "score": score, "tile_only": False,
            "box": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]}


def run(tr, boxes, t0=0):
    out = None
    for i, b in enumerate(boxes):
        out = tr.update([b], float(t0 + i))
    return out


def aged_tracker(in_pool):
    """A tracker past the startup window, so births prove nothing by age."""
    tr = Tracker(in_pool=in_pool)
    for t in range(12):
        tr.update([], float(t))
    return tr


POOL_RIGHT = lambda b: (b[0] + b[2]) / 2 > 0.5  # left = deck, right = pool


def test_near_edge_birth_starts_unverified():
    tr = aged_tracker(lambda b: True)
    run(tr, [box(0.45, cy=0.88)] * 4, t0=12)  # static, y2 = 0.96, score 0.90
    me = tr.tracks[0]
    assert me["confirmed"] and not me["verified"], (me["confirmed"], me["verified"])
    assert tr.counts(15.0)["pool"] == 0
    assert not [e for e in tr.events if e["event"] in ("STILL", "LOST", "ENTER")]


def test_static_deck_blob_cannot_enter():
    tr = aged_tracker(POOL_RIGHT)
    run(tr, [box(0.49)] * 6, t0=12)   # static deck blob, confirmed deck ID
    run(tr, [box(0.53)] * 10, t0=18)  # pool-zone blob 0.04 away, CLEAR score
    assert tr.entries == 0, [e for e in tr.events if e["event"] == "ENTER"]
    assert tr.counts(27.0)["pool"] == 0


def test_moving_deck_person_enters():
    tr = aged_tracker(POOL_RIGHT)
    run(tr, [box(0.40 + 0.02 * i) for i in range(4)], t0=12)  # walk to edge
    run(tr, [box(0.52 + 0.02 * i) for i in range(6)], t0=16)  # wade in
    assert tr.entries == 1, [e for e in tr.events]
    assert tr.next_id == 1  # same ID throughout
    me = tr.tracks[0]
    assert me["zone"] == "pool" and me["verified"]


def test_verified_swimmer_climbing_out_keeps_id():
    tr = aged_tracker(POOL_RIGHT)
    run(tr, [box(0.40 + 0.02 * i) for i in range(4)], t0=12)
    run(tr, [box(0.52 + 0.02 * i) for i in range(8)], t0=16)  # ENTER ~t=18
    run(tr, [box(cx) for cx in (0.60, 0.54, 0.48, 0.44, 0.40, 0.36)],
        t0=24)  # climb out
    assert tr.entries == 1 and tr.exits == 1, [e for e in tr.events]
    assert tr.next_id == 1
    assert tr.tracks[0]["zone"] == "deck"


def test_static_deck_blob_cannot_jump_in():
    # replay id 3: a static deck blob vanishes, a pool stripe appears in
    # handoff reach but beyond normal matching -> must NOT link + ENTER.
    tr = aged_tracker(POOL_RIGHT)
    run(tr, [box(0.30, cy=0.60, w=0.20, h=0.50)] * 6, t0=12)
    for t in (18, 19):
        tr.update([], float(t))
    run(tr, [box(0.52, cy=0.72)] * 3, t0=20)
    assert tr.entries == 0, [e for e in tr.events if e["event"] == "ENTER"]
    assert tr.next_id == 2  # separate, unverified pool ID
    assert tr.tracks[1]["verified"] is False


def test_moving_deck_person_jumping_in_keeps_id():
    tr = aged_tracker(POOL_RIGHT)
    run(tr, [box(0.08 + 0.04 * i, cy=0.60, w=0.20, h=0.50) for i in range(6)],
        t0=12)
    for t in (18, 19):
        tr.update([], float(t))
    run(tr, [box(0.52, cy=0.72)] * 3, t0=20)
    assert tr.entries == 1, [e for e in tr.events]
    assert tr.next_id == 1
    me = tr.tracks[0]
    assert me["zone"] == "pool" and me["verified"]


def test_morphing_blob_cannot_enter():
    # the tower blob: static position, collapsing box, then a pool box
    # nearby. Centroid "travels" as it shrinks - size ratio gives it away.
    tr = aged_tracker(POOL_RIGHT)
    collapse = [box(0.42, cy=0.85 - h / 2, w=0.20, h=h)
                for h in (0.50, 0.43, 0.36, 0.29, 0.22, 0.15)]
    run(tr, collapse, t0=12)
    run(tr, [box(0.52, cy=0.72)] * 6, t0=18)
    assert tr.entries == 0, [e for e in tr.events if e["event"] == "ENTER"]
    assert tr.counts(23.0)["pool"] == 0


def test_swam_needs_solid_motion():
    # a morphing blob can "swim" 0.20+ on centroids while its size
    # collapses - SWAM must not promote it (scores defeat STEADY too).
    tr = aged_tracker(lambda b: True)
    boxes = [box(0.50 + 0.03 * i, cy=0.70, w=0.10, h=0.50 - 0.035 * i,
                 score=0.65 if i == 0 else 0.40) for i in range(12)]
    run(tr, boxes, t0=12)
    me = tr.tracks[0]
    assert me["confirmed"] and not me["verified"], (me["confirmed"], me["verified"])
    assert not [e for e in tr.events if e["event"] == "VERIFIED"]


def test_startup_window_is_short():
    # occupants are visible immediately; a stripe takes seconds to confirm
    tr = Tracker(in_pool=lambda b: True)
    for t in range(6):
        tr.update([], float(t))
    run(tr, [box(0.45, cy=0.70)] * 3, t0=6)  # confirmed at t=7
    me = tr.tracks[0]
    assert me["confirmed"] and not me["verified"]


def test_steady_still_promotes_static_swimmer():
    # Deliberate tradeoff, locked in: a static high-score floater that never
    # entered on camera must still count (far swimmers only verify this way).
    # A hot stripe can pass too - accepted, motion can't separate them.
    tr = aged_tracker(lambda b: True)
    run(tr, [box(0.45, cy=0.70, score=0.85)] * 20, t0=12)
    me = tr.tracks[0]
    assert me["verified"] is True
    assert [e["event"] for e in tr.events if e["event"] == "VERIFIED"]


if __name__ == "__main__":
    test_near_edge_birth_starts_unverified()
    print("PASS near-edge static birth stays unverified, uncounted")
    test_static_deck_blob_cannot_enter()
    print("PASS static deck blob cannot ENTER")
    test_moving_deck_person_enters()
    print("PASS moving deck person ENTERs with same ID")
    test_verified_swimmer_climbing_out_keeps_id()
    print("PASS climb-out EXITs with same ID")
    test_static_deck_blob_cannot_jump_in()
    print("PASS static deck blob cannot jump in")
    test_moving_deck_person_jumping_in_keeps_id()
    print("PASS moving deck person jumping in keeps ID")
    test_morphing_blob_cannot_enter()
    print("PASS morphing blob cannot ENTER")
    test_swam_needs_solid_motion()
    print("PASS SWAM needs solid motion")
    test_startup_window_is_short()
    print("PASS startup window is short")
    test_steady_still_promotes_static_swimmer()
    print("PASS STEADY still promotes static high-score swimmer")
    print("ALL VERIFICATION TESTS PASSED")
