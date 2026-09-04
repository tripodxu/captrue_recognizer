# -*- coding: utf-8 -*-
"""分析 Optimized v4f 失败案例的特征和规律。"""
import json, os, math, sys, cv2
import numpy as np
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BALANCED = os.path.join(ROOT, "tests", "balanced_50_50")
ds = json.load(open(os.path.join(BALANCED, "dataset.json"), encoding="utf-8"))

fail_cases = [c for c in ds["cases"] if c["is_fail_case"]]
pass_cases = [c for c in ds["cases"] if not c["is_fail_case"]]

print("=" * 70)
print("Failure Case Analysis — Optimized v4f")
print("=" * 70)
print(f"Total: {len(ds['cases'])} cases ({len(fail_cases)} fail, {len(pass_cases)} pass)")

# 1. 来源数据集分布
print(f"\n--- 1. Failure by Source Dataset ---")
fail_by_src = Counter(c["source_dataset"] for c in fail_cases)
pass_by_src = Counter(c["source_dataset"] for c in pass_cases)
all_srcs = sorted(set(list(fail_by_src.keys()) + list(pass_by_src.keys())))
for src in all_srcs:
    f = fail_by_src.get(src, 0)
    p = pass_by_src.get(src, 0)
    total = f + p
    print(f"  {src:<25} fail={f:>4}  pass={p:>4}  fail_rate={f/total:.1%}" if total > 0 else "")

# 2. 误差分布
print(f"\n--- 2. Error Distribution (fail cases) ---")
errors = [c["error"] for c in fail_cases]
print(f"  Min error:    {min(errors):.1f} px")
print(f"  Max error:    {max(errors):.1f} px")
print(f"  Mean error:   {np.mean(errors):.1f} px")
print(f"  Median error: {np.median(errors):.1f} px")
print(f"  Std error:    {np.std(errors):.1f} px")
buckets = [(5, 10), (10, 20), (20, 50), (50, 100), (100, 500)]
for lo, hi in buckets:
    cnt = sum(1 for e in errors if lo <= e < hi)
    print(f"  {lo:>3}-{hi:<3} px: {cnt:>4} ({cnt/len(errors):.1%})")

# 3. 图像特征分析 (采样前 200 个)
print(f"\n--- 3. Image Feature Analysis (sample 200 fail + 200 pass) ---")
sample_fail = fail_cases[:200]
sample_pass = pass_cases[:200]

def analyze_img_features(cases):
    feats = {"bg_std": [], "bg_mean": [], "bg_edge_density": [],
             "pz_nonzero_ratio": [], "pz_edge_density": [],
             "bg_entropy": [], "pred_error": []}
    for c in cases:
        bg_path = os.path.join(BALANCED, c["background"])
        pz_path = os.path.join(BALANCED, c["puzzle"])
        bg = cv2.imread(bg_path)
        pz = cv2.imread(pz_path)
        if bg is None or pz is None:
            continue
        bg_g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        pz_g = cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(pz_g, 1, 255, cv2.THRESH_BINARY)
        mb = mask > 0

        # Background features
        feats["bg_mean"].append(float(np.mean(bg_g)))
        feats["bg_std"].append(float(np.std(bg_g)))
        bg_edge = cv2.Canny(bg_g, 100, 200)
        feats["bg_edge_density"].append(float(np.count_nonzero(bg_edge)) / bg_edge.size)

        # Puzzle features
        npx = int(np.sum(mb))
        feats["pz_nonzero_ratio"].append(npx / mask.size if mask.size > 0 else 0)
        pz_edge = cv2.Canny(pz_g, 100, 200)
        pz_edge_masked = cv2.bitwise_and(pz_edge, mask)
        feats["pz_edge_density"].append(float(np.count_nonzero(pz_edge_masked)) / max(npx, 1))

        # Background entropy (rough)
        hist = cv2.calcHist([bg_g], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        feats["bg_entropy"].append(float(-np.sum(hist * np.log2(hist))))

        feats["pred_error"].append(c["error"])
    return feats

f_feats = analyze_img_features(sample_fail)
p_feats = analyze_img_features(sample_pass)

for key in ["bg_mean", "bg_std", "bg_edge_density", "pz_nonzero_ratio", "pz_edge_density", "bg_entropy"]:
    f_vals = f_feats[key]
    p_vals = p_feats[key]
    if not f_vals or not p_vals:
        continue
    print(f"  {key:<20}  fail: mean={np.mean(f_vals):>7.2f} std={np.std(f_vals):>6.2f}"
          f"  pass: mean={np.mean(p_vals):>7.2f} std={np.std(p_vals):>6.2f}")

# 4. 分析: NPC 一致性校验是否导致失败
print(f"\n--- 4. Confidence & Consistency Analysis ---")
print("  (re-running optimized on fail cases to check intermediate results)")

# Re-run optimized on a sample of fail cases to get intermediate values
from benchmark.run_test import detect_optimized

confidence_distribution = {"low_conf_npc_agree": 0, "low_conf_npc_disagree": 0,
                           "high_conf": 0, "ssd_triggered": 0}
fail_confidences = []

for c in fail_cases[:300]:
    bg = cv2.imread(os.path.join(BALANCED, c["background"]))
    pz = cv2.imread(os.path.join(BALANCED, c["puzzle"]))
    if bg is None or pz is None:
        continue
    sx, sy = c["position"]

    # Re-run with debug info
    bg_n = cv2.normalize(bg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pz_n = cv2.normalize(pz, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    bg_g = cv2.cvtColor(bg_n, cv2.COLOR_BGR2GRAY)
    pz_g = cv2.cvtColor(pz_n, cv2.COLOR_BGR2GRAY)

    bv, bl = -1.0, (0, 0)
    for lo, hi in [(30, 100), (50, 150), (80, 180), (100, 200), (120, 240), (150, 250), (180, 300)]:
        be = cv2.Canny(bg_g, lo, hi)
        pe = cv2.Canny(pz_g, lo, hi)
        r = cv2.matchTemplate(be, pe, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(r)
        if mv > bv:
            bv, bl = mv, ml
    cm = max(0.0, min(1.0, bv))

    bn = cv2.Canny(bg_g, 150, 250)
    pn = cv2.Canny(pz_g, 150, 250)
    nr = cv2.matchTemplate(bn, pn, cv2.TM_CCOEFF_NORMED)
    _, _, _, nl = cv2.minMaxLoc(nr)

    npc_agrees = abs(bl[0] - nl[0]) <= 5

    if cm >= 0.35:
        confidence_distribution["high_conf"] += 1
    elif npc_agrees:
        confidence_distribution["low_conf_npc_agree"] += 1
    else:
        confidence_distribution["low_conf_npc_disagree"] += 1
        # Check SSD
        bg_f = bg_g.astype(np.float64)
        pz_f = pz_g.astype(np.float64)
        _, mask = cv2.threshold(pz_g, 1, 255, cv2.THRESH_BINARY)
        mb = mask > 0
        ph, pw = pz_f.shape
        bh, bw = bg_f.shape
        if ph <= bh and pw <= bw:
            npx = int(np.sum(mb))
            if npx > 0:
                confidence_distribution["ssd_triggered"] += 1

    fail_confidences.append(cm)

print(f"  Fail cases (sample 300) by decision path:")
for k, v in confidence_distribution.items():
    print(f"    {k:<30} {v:>4} ({v/300:.1%})")
print(f"  Mean MC confidence on fails: {np.mean(fail_confidences):.3f}")
print(f"  Median MC confidence on fails: {np.median(fail_confidences):.3f}")

# 5. 检查失败案例中高置信度(>=0.35)的比例
high_conf_fails = sum(1 for c in fail_confidences if c >= 0.35)
print(f"  High-conf (>=0.35) failures: {high_conf_fails}/{len(fail_confidences)} ({high_conf_fails/len(fail_confidences):.1%})")
print(f"  These are cases where the algorithm is WRONG but CONFIDENT.")
