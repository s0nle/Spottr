"""Offline test of the merge idea on saved detections (no GPU).
Compares, with the pool outline (pool >= thr, deck ignored):
  full_person         what live does now (HD frame, "person")
  full_person+head    + "head" prompt, merged into one box per person
  full+tiles_person   + 2x2 tiles, merged
  all                 full person+head + tiles person, merged
Also counts "people per frame" in the empty pool (false alarms) and the
average number of boxes per in-water frame (double counting shows up here)."""
import json, sys
import cv2, numpy as np
sys.argv = ["x"]
exec(open("diag/score.py").read().split("path = ")[0])
z = np.array(json.load(open("pool_zone.json"))["points"], np.float32)
d = {r["name"]: r for r in json.load(open("diag/detections.json"))}

def inpool(b): return cv2.pointPolygonTest(z, ((b[0]+b[2])/2, min(b[3], .999)), False) >= 0
def area(b): return max(0, b[2]-b[0]) * max(0, b[3]-b[1])
def inter(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])

def merge(dets, iou_t=0.5, contain_t=0.7):
    """One box per person: a box mostly inside a kept box (head in body,
    tile half of a person) or overlapping it is merged; best score wins."""
    keep = []
    for x in sorted(dets, key=lambda x: -x["s"]):
        dup = False
        for k in keep:
            i = inter(x["box"], k["box"])
            iou = i / max(area(x["box"]) + area(k["box"]) - i, 1e-9)
            con = i / max(min(area(x["box"]), area(k["box"])), 1e-9)
            if iou > iou_t or con > contain_t:
                dup = True
                if area(x["box"]) > area(k["box"]):
                    k["box"] = x["box"]      # keep the bigger box (body over head)
                break
        if not dup:
            keep.append(dict(x))
    return keep

def get(r, combo):
    out = []
    for setting, prompt in combo:
        out += r[setting][prompt]
    return merge(out)

combos = {
    "full_person":       [("sam31_full", "person")],
    "full_person+head":  [("sam31_full", "person"), ("sam31_full", "head")],
    "full+tiles_person": [("sam31_full", "person"), ("sam31_tiles", "person")],
    "all":               [("sam31_full", "person"), ("sam31_full", "head"),
                          ("sam31_tiles", "person")],
}
print(f"labels: empty {len(EMPTY)}, in water {len(IN_WATER)}, underwater {len(UNDERWATER)}")
print(f"{'combo':18s} thr   in water  underwater  false(empty)  boxes/in-water frame")
for name, combo in combos.items():
    for thr in (0.25, 0.30, 0.35):
        def pool(n): return [x for x in get(d[n], combo) if inpool(x["box"]) and x["s"] >= thr]
        hit = lambda S: f"{100*sum(bool(pool(n)) for n in S)/len(S):3.0f}%"
        boxes = np.mean([len(pool(n)) for n in IN_WATER])
        print(f"{name:18s} {thr:.2f}  {hit(IN_WATER):>7s}  {hit(UNDERWATER):>9s}  "
              f"{hit(EMPTY):>10s}  {boxes:8.2f}")
