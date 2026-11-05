"""Spottr alarm engine: turns the tracker's timers into alarm levels.

Levels (per swimmer), from drowning research (see README of this module below):
  0 OK
  1 WATCH      under the water 10 s, or still with no head up for 10 s,
               or out of view (side edge) 20 s
  2 ALARM      under 20 s, or still WHILE submerged 10 s (Poseidon-style),
               or unseen in mid-pool 15 s, or LOST at a frame edge
  3 EMERGENCY  under 30 s (lifeguard 10/20 rule: spotted within 10 s,
               reached within 20 s more = rescue under way by 30 s),
               or LOST in mid-pool

Why these numbers:
  - Ellis & Associates 10/20 rule: recognise a swimmer in trouble within
    10 s, reach them within 20 s.
  - Pia (instinctive drowning response): 20-60 s of silent surface struggle
    before going under.
  - Poseidon (commercial pool system): alarm after ~10 s motionless on the
    bottom (3-15 s across versions).
  - Outcome depends on submersion time; brain damage starts at ~4-6 min.
Healthy adults playing breath-hold games will trip ALARM: accepted for the
demo (staged drills will calibrate).

Rules only for VERIFIED pool swimmers, plus UNVERIFIED ones that have shown
a clear head at least once (a real person who swam in from the side). Lane
stripes never get a head, so they never alarm.

Muse Spark second opinion (optional): may cancel a WATCH or raise any level
to ALARM, but can NEVER lower an ALARM or EMERGENCY.
"""

import time

OK, WATCH, ALARM, EMERGENCY = 0, 1, 2, 3
NAMES = {OK: "OK", WATCH: "WATCH", ALARM: "ALARM", EMERGENCY: "EMERGENCY"}


class AlarmEngine:
    def __init__(self, watch_under=10.0, alarm_under=20.0, emergency_under=30.0,
                 still_watch=10.0, still_alarm=20.0, unseen_alarm=15.0,
                 outview_watch=20.0, muse_valid_s=15.0, clear_hold=3.0):
        # still_alarm is 20 s, not Poseidon's 10 s: a far swimmer treading
        # water in a red cap had no "head" box and read "still, submerged"
        # (4-min clip, 50-57 s). WATCH at 10 s asks Muse, which can raise it.
        self.clear_hold = clear_hold  # a level only drops after 3 s clear
        self.watch_under, self.alarm_under = watch_under, alarm_under
        self.emergency_under = emergency_under
        self.still_watch, self.still_alarm = still_watch, still_alarm
        self.unseen_alarm, self.outview_watch = unseen_alarm, outview_watch
        self.muse_valid_s = muse_valid_s
        self.state = {}    # tid -> {level, since, reason, acked, muse}
        self.events = []

    # ---- rules --------------------------------------------------------------
    def _rules(self, tr, t, head_bar, has_heads):
        """(level, reason) from one track's timers."""
        st = tr["state"]
        head_up = has_heads and max(tr.get("head_hist") or [0.0]) >= head_bar
        edge = st == "LOST" and tr.get("lost_at_edge", False)
        # out of view at a side edge is not "under water" (LOST there = ALARM)
        under = (tr["under_s"] if st in ("SUBMERGED", "UNSEEN")
                 or (st == "LOST" and not edge) else 0.0)
        unseen = t - tr["last"] if st == "UNSEEN" else 0.0
        still = tr.get("still_s", 0.0) if st in ("SURFACE", "SUBMERGED") else 0.0
        out = []
        if st == "LOST":
            out.append((ALARM if edge else EMERGENCY,
                        "LOST at the frame edge" if edge else "LOST in the pool"))
        if under >= self.emergency_under:
            out.append((EMERGENCY, f"under water {under:.0f}s"))
        elif under >= self.alarm_under:
            out.append((ALARM, f"under water {under:.0f}s"))
        elif under >= self.watch_under:
            out.append((WATCH, f"under water {under:.0f}s"))
        if unseen >= self.unseen_alarm:
            out.append((ALARM, f"not seen in mid-pool {unseen:.0f}s"))
        if still >= self.still_alarm and not head_up and (st == "SUBMERGED" or under >= 3):
            out.append((ALARM, f"still, head not up {still:.0f}s"))
        elif still >= self.still_watch and not head_up:
            out.append((WATCH, f"still, head not up {still:.0f}s"))
        if st == "OUTVIEW" and tr["under_s"] >= self.outview_watch:
            out.append((WATCH, f"out of view {tr['under_s']:.0f}s"))
        return max(out, default=(OK, ""), key=lambda x: x[0])

    def update(self, tracker, t):
        """Re-evaluate every swimmer. Returns the list of level changes."""
        changes = []
        live = set()
        for tid, tr in tracker.tracks.items():
            if not tr["confirmed"] or tr["zone"] != "pool":
                continue
            if not tr["verified"] and tr.get("head_best", 0.0) < tracker.head_bar:
                continue  # unverified and never showed a head: likely a stripe
            live.add(tid)
            level, reason = self._rules(tr, t, tracker.head_bar, tracker.has_heads)
            if not tr["verified"]:
                # never seen entering (but showed a head): at most WATCH,
                # it may still be a stripe ID a swimmer briefly passed over
                level = min(level, WATCH)
            s = self.state.setdefault(tid, {"level": OK, "since": t, "reason": "",
                                            "acked": False, "muse": None,
                                            "low_since": None})
            m = s["muse"]
            if m and t - m["t"] <= self.muse_valid_s:
                if m["verdict"] == "emergency" and level < ALARM:
                    level, reason = ALARM, f"Muse: {m['reason']}"
                elif m["verdict"] == "ok" and level == WATCH:
                    level, reason = OK, f"Muse: {m['reason']}"
            # hysteresis: a level drops only after its cause has been gone
            # for clear_hold seconds (states flicker check to check)
            if level < s["level"]:
                if s["low_since"] is None:
                    s["low_since"] = t
                if t - s["low_since"] < self.clear_hold and not (
                        m and m["verdict"] == "ok" and s["level"] == WATCH):
                    level, reason = s["level"], s["reason"]
            else:
                s["low_since"] = None
            if level != s["level"]:
                up = level > s["level"]
                changes.append({"t": round(t, 2), "id": tid, "from": NAMES[s["level"]],
                                "to": NAMES[level], "reason": reason})
                self.events.append(changes[-1])
                s.update(level=level, since=t, reason=reason)
                if up:
                    s["acked"] = False  # a new, higher level always sounds again
            else:
                s["reason"] = reason
        for tid in list(self.state):
            if tid not in live:  # left the pool / dropped
                del self.state[tid]
        return changes

    # ---- inputs from the operator / Muse --------------------------------------
    def ack(self, tid):
        """Lifeguard has seen it: stop the sound for this swimmer at this level."""
        if tid in self.state:
            self.state[tid]["acked"] = True

    def muse_opinion(self, tid, verdict, reason, t=None):
        if tid in self.state:
            self.state[tid]["muse"] = {"verdict": verdict, "reason": reason,
                                       "t": time.perf_counter() if t is None else t}

    # ---- outputs ----------------------------------------------------------------
    def level(self, tid):
        return self.state.get(tid, {}).get("level", OK)

    def top(self):
        return max((s["level"] for s in self.state.values()), default=OK)

    def top_active(self):
        """Highest level the lifeguard has NOT acknowledged yet (drives the
        screen flash and banners; acknowledged alarms stay on their card)."""
        return max((s["level"] for s in self.state.values() if not s["acked"]), default=OK)

    def dismiss(self, tid):
        self.state.pop(tid, None)

    def sounding(self):
        """True while any un-acknowledged ALARM / EMERGENCY is active."""
        return any(s["level"] >= ALARM and not s["acked"] for s in self.state.values())

    def border(self, tid, t):
        """Red border thickness (px) for a swimmer: grows with time in alarm."""
        s = self.state.get(tid)
        if not s or s["level"] < WATCH:
            return 0
        base = {WATCH: 3, ALARM: 6, EMERGENCY: 10}[s["level"]]
        return int(min(30, base + 1.0 * (t - s["since"])))


class Beeper:
    """Alarm sound on the laptop while engine.sounding() is True (Windows beep;
    a terminal bell elsewhere). Runs in its own thread so it never blocks."""

    def __init__(self, engine):
        import threading
        self.engine, self.stop = engine, False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            import winsound
            beep = lambda f, ms: winsound.Beep(f, ms)  # noqa: E731
        except ImportError:
            beep = lambda f, ms: (print("\a", end="", flush=True), time.sleep(ms / 1000))  # noqa: E731
        while not self.stop:
            if self.engine.sounding():
                hi = self.engine.top() >= EMERGENCY
                beep(1400 if hi else 1000, 250)
                time.sleep(0.1 if hi else 0.4)
            else:
                time.sleep(0.2)
