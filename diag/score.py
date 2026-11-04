"""Scores diag/detections.json against frames labeled by eye.

Labels come from watching the raw footage (one frame every 2 s):
  EMPTY       pool is empty: any detection is a false alarm
  IN_WATER    a swimmer is clearly in the water (surface or under)
  UNDERWATER  the swimmer is fully under the water (subset of IN_WATER)

For each setting it prints, at several thresholds:
  caught   share of IN_WATER / UNDERWATER frames with at least one detection
  false    share of EMPTY frames with at least one detection
"""

import json
import sys


def frames(prefix, *ranges):
    out = set()
    for r in ranges:
        a, b = (r, r) if isinstance(r, int) else r
        out |= {f"{prefix}_{i:04d}.jpg" for i in range(a, b + 1)}
    return out


# Corrections after checking at full size: a_0011 has a real swimmer in the
# far-left lane (not empty); a_0097 and a_0168 show no visible swimmer at all
# (hidden by surface glare), so they are not counted as underwater targets.
EMPTY = frames("a", (0, 35)) - frames("a", 11)
UNDERWATER = frames("a", 85, 96, 166, 167) | frames("b", 66, 204)
IN_WATER = (frames("a", (37, 41), (78, 85), 96, 98, (150, 154), 166, 167)
            | frames("b", (6, 11), (66, 69), (198, 204)))

path = sys.argv[1] if len(sys.argv) > 1 else "diag/detections.json"
data = {r["name"]: r for r in json.load(open(path))}
settings = [k for k in next(iter(data.values())) if k not in ("name", "w", "h")]


def best(r, setting, prompts):
    return max((d["s"] for p in prompts for d in r[setting].get(p, [])), default=0.0)


def rate(names, setting, prompts, thr):
    names = [n for n in names if n in data]
    hit = sum(best(data[n], setting, prompts) >= thr for n in names)
    return f"{100 * hit / max(len(names), 1):3.0f}%"


print(f"labeled frames found: empty {len(EMPTY & data.keys())}, "
      f"in water {len(IN_WATER & data.keys())}, underwater {len(UNDERWATER & data.keys())}")
for prompts in (["person"], ["swimmer"], ["person", "swimmer"]):
    print(f"\n=== prompt(s): {' + '.join(prompts)} ===")
    print(f"{'setting':13s} {'thr':>4s}  {'in water':>8s} {'underwater':>10s} {'false (empty)':>13s}")
    for s in settings:
        if not all(p in next(iter(data.values()))[s] for p in prompts):
            continue
        for thr in (0.3, 0.35, 0.4, 0.5):
            print(f"{s:13s} {thr:4.2f}  {rate(IN_WATER, s, prompts, thr):>8s} "
                  f"{rate(UNDERWATER, s, prompts, thr):>10s} {rate(EMPTY, s, prompts, thr):>13s}")
