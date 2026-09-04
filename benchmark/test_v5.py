# -*- coding: utf-8 -*-
"""Test method D (MC or NPC, no SSD) on balanced + original datasets."""
import json, math, os, sys, time
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import benchmark.npc_baseline as npc


def detect_v4f(background, puzzle):
    """Original v4f."""
    bg = cv2.normalize(background, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pz = cv2.normalize(puzzle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    bg_g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    pz_g = cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY)
    bv, bl = -1.0, (0, 0)
    for lo, hi in [(30, 100), (50, 150), (80, 180), (100, 200), (120, 240), (150, 250), (180, 300)]:
        be = cv2.Canny(bg_g, lo, hi); pe = cv2.Canny(pz_g, lo, hi)
        r = cv2.matchTemplate(be, pe, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(r)
        if mv > bv: bv, bl = mv, ml
    xm, ym = bl; cm = max(0.0, min(1.0, bv))
    bn = cv2.Canny(bg_g, 150, 250); pn = cv2.Canny(pz_g, 150, 250)
    nr = cv2.matchTemplate(bn, pn, cv2.TM_CCOEFF_NORMED)
    _, _, _, nl = cv2.minMaxLoc(nr)
    if cm >= 0.35 or abs(xm - nl[0]) <= 5: return xm, ym
    bg_f = bg_g.astype(np.float64); pz_f = pz_g.astype(np.float64)
    _, mask = cv2.threshold(pz_g, 1, 255, cv2.THRESH_BINARY); mb = mask > 0
    ph, pw = pz_f.shape; bh, bw = bg_f.shape
    if ph > bh or pw > bw: return xm, ym
    npx = int(np.sum(mb))
    if npx == 0: return xm, ym
    be2, bx = 1e18, 0
    for x in range(0, bw - pw + 1, 2):
        err = float(np.sum(((bg_f[0:ph, x:x + pw] - pz_f) ** 2)[mb])) / npx
        if err < be2: be2, bx = err, x
    for x in range(max(0, bx - 2), min(bw - pw + 1, bx + 3)):
        err = float(np.sum(((bg_f[0:ph, x:x + pw] - pz_f) ** 2)[mb])) / npx
        if err < be2: be2, bx = err, x
    sc = max(0.0, 1.0 - be2 / 10000.0)
    if sc > 0.80: return bx, ym
    return xm, ym


def detect_v5(background, puzzle):
    """v5: MC + NPC confidence comparison, no SSD."""
    bg = cv2.normalize(background, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pz = cv2.normalize(puzzle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    bg_g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    pz_g = cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY)
    bv, bl = -1.0, (0, 0)
    for lo, hi in [(30, 100), (50, 150), (80, 180), (100, 200), (120, 240), (150, 250), (180, 300)]:
        be = cv2.Canny(bg_g, lo, hi); pe = cv2.Canny(pz_g, lo, hi)
        r = cv2.matchTemplate(be, pe, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(r)
        if mv > bv: bv, bl = mv, ml
    xm, ym = bl; cm = max(0.0, min(1.0, bv))
    bn = cv2.Canny(bg_g, 150, 250); pn = cv2.Canny(pz_g, 150, 250)
    nr = cv2.matchTemplate(bn, pn, cv2.TM_CCOEFF_NORMED)
    npc_val, _, _, nl = cv2.minMaxLoc(nr)
    cn = max(0.0, min(1.0, npc_val))
    if cm >= 0.35 or abs(xm - nl[0]) <= 5: return xm, ym
    if cn > cm: return nl[0], nl[1]
    return xm, ym


def run_test(m_fn, ds_dir, cases, tol):
    ok = fail = 0; t = 0.0
    for c in tqdm(cases, ncols=70, leave=False,
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"):
        bg = cv2.imread(os.path.join(ds_dir, c["background"]))
        pz = cv2.imread(os.path.join(ds_dir, c["puzzle"]))
        if bg is None or pz is None: fail += 1; continue
        sx, sy = c["position"]
        t0 = time.perf_counter(); rx, ry = m_fn(bg, pz); t += time.perf_counter() - t0
        if math.hypot(sx - rx, sy - ry) <= tol: ok += 1
        else: fail += 1
    n = ok + fail
    return ok, n, t


tests_dir = os.path.join(ROOT, "tests")
methods = [
    ("NPC Baseline", npc.detect),
    ("v4f Original (SSD)", detect_v4f),
    ("v5 MC-or-NPC (no SSD)", detect_v5),
]

datasets = [
    ("balanced_50_50", None),
    ("geetest_test", None), ("tricky_test", None), ("tricky_hard_test", None),
    ("syn_easy", None), ("syn_medium", None), ("syn_hard", None),
    ("syn_slider_easy", None), ("syn_slider_hard", None),
]

print("=" * 85)
print("v5 (MC-or-NPC, no SSD) vs v4f Original — all datasets")
print("=" * 85)

totals = {m[0]: [0, 0, 0] for m in methods}

for ds_name, max_c in datasets:
    ds_dir = os.path.join(tests_dir, ds_name)
    jp = os.path.join(ds_dir, "dataset.json")
    if not os.path.exists(jp): continue
    ds = json.load(open(jp, encoding="utf-8"))
    tol = ds.get("error_tolerance", 5)
    cases = ds["cases"][:max_c] if max_c else ds["cases"]
    n = len(cases)
    label = ds.get("name", ds_name)[:30]
    print(f"\n  {label} ({n}, tol={tol})")
    for m_name, m_fn in methods:
        ok, tn, tt = run_test(m_fn, ds_dir, cases, tol)
        acc = ok / tn if tn else 0
        spd = tn / tt if tt else 0
        totals[m_name][0] += ok; totals[m_name][1] += tn; totals[m_name][2] += tt
        print(f"    {m_name:<25} {acc:>6.1%} ({ok:>4}/{tn:<4}) {spd:>7.0f} i/s")

print("\n" + "=" * 75)
print("TOTALS")
print("=" * 75)
for m_name in totals:
    ok, tn, tt = totals[m_name]
    acc = ok / tn if tn else 0
    spd = tn / tt if tt else 0
    print(f"  {m_name:<25} {acc:>6.1%} ({ok:>5}/{tn:<5}) {spd:>7.0f} i/s")
