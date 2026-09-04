# -*- coding: utf-8 -*-
"""
对比 v4f (SSD) 与 v5 (无 SSD) 在各数据集上的表现。

用法:
  python benchmark/test_v5.py
"""
import json, math, os, sys, time
import cv2
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import benchmark.algorithms as algo


def detect_v4f(background, puzzle):
    """v4f: MC + NPC + SSD 兜底 (历史版本，用于对比)。"""
    bg_g, pz_g = algo._preprocess(background, puzzle)
    mx, my, mc = algo._multi_canny_match(bg_g, pz_g)
    nx, ny, nc = algo._npc_match(bg_g, pz_g)

    if mc >= algo.CONFIDENCE_THRESHOLD or abs(mx - nx) <= algo.AGREEMENT_TOLERANCE:
        return mx, my

    # SSD
    bg_f = bg_g.astype(np.float64)
    pz_f = pz_g.astype(np.float64)
    _, mask = cv2.threshold(pz_g, 1, 255, cv2.THRESH_BINARY)
    mb = mask > 0
    ph, pw = pz_f.shape
    bh, bw = bg_f.shape
    if ph > bh or pw > bw:
        return mx, my
    npx = int(np.sum(mb))
    if npx == 0:
        return mx, my
    be2, bx = 1e18, 0
    for x in range(0, bw - pw + 1, 2):
        err = float(np.sum(((bg_f[0:ph, x:x + pw] - pz_f) ** 2)[mb])) / npx
        if err < be2:
            be2, bx = err, x
    for x in range(max(0, bx - 2), min(bw - pw + 1, bx + 3)):
        err = float(np.sum(((bg_f[0:ph, x:x + pw] - pz_f) ** 2)[mb])) / npx
        if err < be2:
            be2, bx = err, x
    if max(0.0, 1.0 - be2 / 10000.0) > 0.80:
        return bx, my
    return mx, my


def run_test(m_fn, ds_dir, cases, tol):
    ok = fail = 0
    t = 0.0
    for c in tqdm(cases, ncols=70, leave=False,
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"):
        bg = cv2.imread(os.path.join(ds_dir, c["background"]))
        pz = cv2.imread(os.path.join(ds_dir, c["puzzle"]))
        if bg is None or pz is None:
            fail += 1
            continue
        sx, sy = c["position"]
        t0 = time.perf_counter()
        rx, ry = m_fn(bg, pz)
        t += time.perf_counter() - t0
        if math.hypot(sx - rx, sy - ry) <= tol:
            ok += 1
        else:
            fail += 1
    n = ok + fail
    return ok, n, t


def main():
    tests_dir = os.path.join(ROOT, "tests")
    methods = [
        ("NPC Baseline", algo.detect_npc),
        ("v4f (with SSD)", detect_v4f),
        ("v5 (no SSD)", algo.detect_v5),
    ]
    datasets = [
        "balanced_50_50", "geetest_test", "tricky_test", "tricky_hard_test",
        "syn_easy", "syn_medium", "syn_hard", "syn_slider_easy", "syn_slider_hard",
    ]

    print("=" * 85)
    print("v5 vs v4f — all datasets")
    print("=" * 85)

    totals = {m[0]: [0, 0, 0] for m in methods}

    for ds_name in datasets:
        jp = os.path.join(tests_dir, ds_name, "dataset.json")
        if not os.path.exists(jp):
            continue
        with open(jp, encoding="utf-8") as f:
            ds = json.load(f)
        tol = ds.get("error_tolerance", 5)
        ds_dir = os.path.join(tests_dir, ds_name)
        cases = ds["cases"]
        n = len(cases)
        label = ds.get("name", ds_name)[:30]

        print(f"\n  {label} ({n}, tol={tol})")
        for m_name, m_fn in methods:
            ok, tn, tt = run_test(m_fn, ds_dir, cases, tol)
            acc = ok / tn if tn else 0
            spd = tn / tt if tt else 0
            totals[m_name][0] += ok
            totals[m_name][1] += tn
            totals[m_name][2] += tt
            print(f"    {m_name:<20} {acc:>6.1%} ({ok:>4}/{tn:<4}) {spd:>7.0f} i/s")

    print("\n" + "=" * 75)
    print("TOTALS")
    print("=" * 75)
    for m_name, (ok, tn, tt) in totals.items():
        acc = ok / tn if tn else 0
        spd = tn / tt if tt else 0
        print(f"  {m_name:<20} {acc:>6.1%} ({ok:>5}/{tn:<5}) {spd:>7.0f} i/s")


if __name__ == "__main__":
    main()
