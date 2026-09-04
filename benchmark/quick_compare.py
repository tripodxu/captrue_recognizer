# -*- coding: utf-8 -*-
"""Quick per-dataset accuracy for all 3 methods."""
import json, math, os, sys, time
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import benchmark.algorithms as algo

datasets = [
    ("tests/geetest_test", 5),
    ("tests/tricky_test", 5),
    ("tests/tricky_hard_test", 5),
    ("tests/syn_easy", 5),
    ("tests/syn_medium", 5),
    ("tests/syn_hard", 5),
    ("tests/syn_slider_easy", 5),
    ("tests/syn_slider_hard", 5),
    ("tests/vcode_caltech_5k", 10),
    ("tests/vcode_caltech_30k", 10),
    ("tests/balanced_50_50", 5),
]

methods = [("NPC", algo.detect_npc), ("v4f", algo.detect_v4f), ("v5", algo.detect_v5)]

for rp, tol in datasets:
    jp = os.path.join(rp, "dataset.json")
    if not os.path.exists(jp):
        continue
    with open(jp, encoding="utf-8") as f:
        ds = json.load(f)
    cases = ds["cases"]
    n = len(cases)
    name = ds.get("name", rp)[:35]
    results = {}
    for mname, mfn in methods:
        ok = 0
        for c in cases:
            bg = cv2.imread(os.path.join(rp, c["background"]))
            pz = cv2.imread(os.path.join(rp, c["puzzle"]))
            if bg is None or pz is None:
                continue
            sx, sy = c["position"]
            rx, ry = mfn(bg, pz)
            if math.hypot(sx - rx, sy - ry) <= tol:
                ok += 1
        results[mname] = ok
    npc_o = results["NPC"]
    v4f_o = results["v4f"]
    v5_o = results["v5"]
    print(f"{name:<35} {n:>5}  NPC:{npc_o:>5}({100*npc_o/n:5.1f}%)  "
          f"v4f:{v4f_o:>5}({100*v4f_o/n:5.1f}%)  v5:{v5_o:>5}({100*v5_o/n:5.1f}%)")
