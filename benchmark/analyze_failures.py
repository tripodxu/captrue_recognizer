# -*- coding: utf-8 -*-
"""
分析 Optimized v5 失败案例的特征和规律。

用法:
  python benchmark/analyze_failures.py
"""
import json, os, sys
import cv2
import numpy as np
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import benchmark.algorithms as algo


def analyze_img_features(cases, base_dir):
    """提取图像特征用于失败/成功对比。"""
    feats = {k: [] for k in ["bg_mean", "bg_std", "bg_edge_density",
                              "pz_nonzero_ratio", "pz_edge_density", "bg_entropy"]}
    for c in cases:
        bg = cv2.imread(os.path.join(base_dir, c["background"]))
        pz = cv2.imread(os.path.join(base_dir, c["puzzle"]))
        if bg is None or pz is None:
            continue
        bg_g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        pz_g = cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(pz_g, 1, 255, cv2.THRESH_BINARY)
        npx = int(np.sum(mask > 0))

        feats["bg_mean"].append(float(np.mean(bg_g)))
        feats["bg_std"].append(float(np.std(bg_g)))
        bg_edge = cv2.Canny(bg_g, 100, 200)
        feats["bg_edge_density"].append(float(np.count_nonzero(bg_edge)) / bg_edge.size)
        feats["pz_nonzero_ratio"].append(npx / mask.size if mask.size > 0 else 0)
        pz_edge = cv2.bitwise_and(cv2.Canny(pz_g, 100, 200), mask)
        feats["pz_edge_density"].append(float(np.count_nonzero(pz_edge)) / max(npx, 1))

        hist = cv2.calcHist([bg_g], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        feats["bg_entropy"].append(float(-np.sum(hist * np.log2(hist))))
    return feats


def main():
    balanced_dir = os.path.join(ROOT, "tests", "balanced_50_50")
    json_path = os.path.join(balanced_dir, "dataset.json")
    if not os.path.exists(json_path):
        print("balanced_50_50 dataset not found. Run extract_balanced.py first.")
        return

    with open(json_path, encoding="utf-8") as f:
        ds = json.load(f)

    fail_cases = [c for c in ds["cases"] if c["is_fail_case"]]
    pass_cases = [c for c in ds["cases"] if not c["is_fail_case"]]

    print("=" * 70)
    print("Failure Case Analysis")
    print("=" * 70)
    print(f"Total: {len(ds['cases'])} cases ({len(fail_cases)} fail, {len(pass_cases)} pass)")

    # 1. 来源分布
    print("\n--- 1. Failure by Source Dataset ---")
    fail_by_src = Counter(c["source_dataset"] for c in fail_cases)
    pass_by_src = Counter(c["source_dataset"] for c in pass_cases)
    for src in sorted(set(fail_by_src) | set(pass_by_src)):
        f, p = fail_by_src.get(src, 0), pass_by_src.get(src, 0)
        total = f + p
        if total > 0:
            print(f"  {src:<25} fail={f:>4}  pass={p:>4}  fail_rate={f/total:.1%}")

    # 2. 误差分布
    print("\n--- 2. Error Distribution (fail cases) ---")
    errors = [c["error"] for c in fail_cases]
    print(f"  Min={min(errors):.1f}  Max={max(errors):.1f}  "
          f"Mean={np.mean(errors):.1f}  Median={np.median(errors):.1f}  Std={np.std(errors):.1f}")
    for lo, hi in [(5, 10), (10, 20), (20, 50), (50, 100), (100, 500)]:
        cnt = sum(1 for e in errors if lo <= e < hi)
        print(f"  {lo:>3}-{hi:<3} px: {cnt:>4} ({cnt/len(errors):.1%})")

    # 3. 图像特征对比
    print("\n--- 3. Image Feature Comparison (sample 200 each) ---")
    f_feats = analyze_img_features(fail_cases[:200], balanced_dir)
    p_feats = analyze_img_features(pass_cases[:200], balanced_dir)
    for key in f_feats:
        fv, pv = f_feats[key], p_feats[key]
        if not fv or not pv:
            continue
        print(f"  {key:<20}  fail: {np.mean(fv):>7.2f}+/-{np.std(fv):>6.2f}"
              f"  pass: {np.mean(pv):>7.2f}+/-{np.std(pv):>6.2f}")

    # 4. 置信度分析
    print("\n--- 4. Confidence Analysis (fail cases, sample 300) ---")
    counts = {"high_conf": 0, "low_conf_agree": 0, "low_conf_disagree": 0}
    confs = []
    for c in fail_cases[:300]:
        bg = cv2.imread(os.path.join(balanced_dir, c["background"]))
        pz = cv2.imread(os.path.join(balanced_dir, c["puzzle"]))
        if bg is None or pz is None:
            continue
        bg_g = cv2.cvtColor(
            cv2.normalize(bg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX),
            cv2.COLOR_BGR2GRAY,
        )
        pz_g = cv2.cvtColor(
            cv2.normalize(pz, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX),
            cv2.COLOR_BGR2GRAY,
        )
        _, _, mc = algo._multi_canny_match(bg_g, pz_g)
        nx, _, _ = algo._npc_match(bg_g, pz_g)
        # Re-derive MC x for agreement check
        bv, bl = -1.0, (0, 0)
        for lo, hi in algo.CANNY_THRESHOLDS:
            r = cv2.matchTemplate(cv2.Canny(bg_g, lo, hi), cv2.Canny(pz_g, lo, hi),
                                  cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(r)
            if mv > bv:
                bv, bl = mv, ml
        agrees = abs(bl[0] - nx) <= algo.AGREEMENT_TOLERANCE

        if mc >= algo.CONFIDENCE_THRESHOLD:
            counts["high_conf"] += 1
        elif agrees:
            counts["low_conf_agree"] += 1
        else:
            counts["low_conf_disagree"] += 1
        confs.append(mc)

    total_sampled = sum(counts.values())
    for k, v in counts.items():
        print(f"  {k:<25} {v:>4} ({v/total_sampled:.1%})")
    print(f"  Mean MC confidence:  {np.mean(confs):.3f}")
    print(f"  Median MC confidence: {np.median(confs):.3f}")


if __name__ == "__main__":
    main()
