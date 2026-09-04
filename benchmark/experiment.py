# -*- coding: utf-8 -*-
"""
实验：针对失败模式优化 v4f 算法。
在 tests/balanced_50_50 上测试，目标：准确率显著高于 50%。
"""
import json, math, os, sys, time, argparse
import cv2
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import benchmark.npc_baseline as npc

# ============================================================================
#  实验版本
# ============================================================================

def detect_v4f_original(background, puzzle):
    """原始 v4f。"""
    bg = cv2.normalize(background, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pz = cv2.normalize(puzzle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    bg_g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    pz_g = cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY)
    bv, bl = -1.0, (0, 0)
    for lo, hi in [(30, 100), (50, 150), (80, 180), (100, 200), (120, 240), (150, 250), (180, 300)]:
        be = cv2.Canny(bg_g, lo, hi)
        pe = cv2.Canny(pz_g, lo, hi)
        r = cv2.matchTemplate(be, pe, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(r)
        if mv > bv: bv, bl = mv, ml
    xm, ym = bl
    cm = max(0.0, min(1.0, bv))
    bn = cv2.Canny(bg_g, 150, 250)
    pn = cv2.Canny(pz_g, 150, 250)
    nr = cv2.matchTemplate(bn, pn, cv2.TM_CCOEFF_NORMED)
    _, _, _, nl = cv2.minMaxLoc(nr)
    if cm >= 0.35 or abs(xm - nl[0]) <= 5: return xm, ym
    bg_f = bg_g.astype(np.float64)
    pz_f = pz_g.astype(np.float64)
    _, mask = cv2.threshold(pz_g, 1, 255, cv2.THRESH_BINARY)
    mb = mask > 0
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


def detect_v4a_no_ssd(background, puzzle):
    """实验 A: 去掉 SSD 兜底，仅用 MC + NPC。"""
    bg = cv2.normalize(background, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pz = cv2.normalize(puzzle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    bg_g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    pz_g = cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY)
    bv, bl = -1.0, (0, 0)
    for lo, hi in [(30, 100), (50, 150), (80, 180), (100, 200), (120, 240), (150, 250), (180, 300)]:
        be = cv2.Canny(bg_g, lo, hi)
        pe = cv2.Canny(pz_g, lo, hi)
        r = cv2.matchTemplate(be, pe, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(r)
        if mv > bv: bv, bl = mv, ml
    xm, ym = bl
    cm = max(0.0, min(1.0, bv))
    bn = cv2.Canny(bg_g, 150, 250)
    pn = cv2.Canny(pz_g, 150, 250)
    nr = cv2.matchTemplate(bn, pn, cv2.TM_CCOEFF_NORMED)
    _, _, _, nl = cv2.minMaxLoc(nr)
    if cm >= 0.35 or abs(xm - nl[0]) <= 5: return xm, ym
    # No SSD — use NPC as fallback
    return nl[0], nl[1]


def detect_v4b_npc_fallback(background, puzzle):
    """实验 B: 低置信度时优先用 NPC 而非 SSD。"""
    bg = cv2.normalize(background, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pz = cv2.normalize(puzzle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    bg_g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    pz_g = cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY)
    bv, bl = -1.0, (0, 0)
    for lo, hi in [(30, 100), (50, 150), (80, 180), (100, 200), (120, 240), (150, 250), (180, 300)]:
        be = cv2.Canny(bg_g, lo, hi)
        pe = cv2.Canny(pz_g, lo, hi)
        r = cv2.matchTemplate(be, pe, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(r)
        if mv > bv: bv, bl = mv, ml
    xm, ym = bl
    cm = max(0.0, min(1.0, bv))
    bn = cv2.Canny(bg_g, 150, 250)
    pn = cv2.Canny(pz_g, 150, 250)
    nr = cv2.matchTemplate(bn, pn, cv2.TM_CCOEFF_NORMED)
    _, _, _, nl = cv2.minMaxLoc(nr)
    if cm >= 0.35 or abs(xm - nl[0]) <= 5: return xm, ym
    # Low conf + disagree: use NPC directly
    return nl[0], nl[1]


def detect_v4c_normalized_ssd(background, puzzle):
    """实验 C: SSD 使用归一化图像 (修复暗化背景偏差)。"""
    bg = cv2.normalize(background, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pz = cv2.normalize(puzzle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    bg_g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    pz_g = cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY)
    bv, bl = -1.0, (0, 0)
    for lo, hi in [(30, 100), (50, 150), (80, 180), (100, 200), (120, 240), (150, 250), (180, 300)]:
        be = cv2.Canny(bg_g, lo, hi)
        pe = cv2.Canny(pz_g, lo, hi)
        r = cv2.matchTemplate(be, pe, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(r)
        if mv > bv: bv, bl = mv, ml
    xm, ym = bl
    cm = max(0.0, min(1.0, bv))
    bn = cv2.Canny(bg_g, 150, 250)
    pn = cv2.Canny(pz_g, 150, 250)
    nr = cv2.matchTemplate(bn, pn, cv2.TM_CCOEFF_NORMED)
    _, _, _, nl = cv2.minMaxLoc(nr)
    if cm >= 0.35 or abs(xm - nl[0]) <= 5: return xm, ym
    # SSD with NORMALIZED images (fix darkening bias)
    _, mask = cv2.threshold(pz_g, 1, 255, cv2.THRESH_BINARY)
    mb = mask > 0
    ph, pw = pz_g.shape; bh, bw = bg_g.shape
    if ph > bh or pw > bw: return xm, ym
    npx = int(np.sum(mb))
    if npx == 0: return xm, ym
    be2, bx = 1e18, 0
    for x in range(0, bw - pw + 1, 2):
        err = float(np.sum(((bg_g[0:ph, x:x + pw].astype(float) - pz_g.astype(float)) ** 2)[mb])) / npx
        if err < be2: be2, bx = err, x
    for x in range(max(0, bx - 2), min(bw - pw + 1, bx + 3)):
        err = float(np.sum(((bg_g[0:ph, x:x + pw].astype(float) - pz_g.astype(float)) ** 2)[mb])) / npx
        if err < be2: be2, bx = err, x
    sc = max(0.0, 1.0 - be2 / 10000.0)
    if sc > 0.80: return bx, ym
    return xm, ym


def detect_v4d_mc_or_npc(background, puzzle):
    """实验 D: MC 和 NPC 取置信度更高者，完全不用 SSD。"""
    bg = cv2.normalize(background, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pz = cv2.normalize(puzzle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    bg_g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    pz_g = cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY)
    bv, bl = -1.0, (0, 0)
    for lo, hi in [(30, 100), (50, 150), (80, 180), (100, 200), (120, 240), (150, 250), (180, 300)]:
        be = cv2.Canny(bg_g, lo, hi)
        pe = cv2.Canny(pz_g, lo, hi)
        r = cv2.matchTemplate(be, pe, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(r)
        if mv > bv: bv, bl = mv, ml
    xm, ym = bl
    cm = max(0.0, min(1.0, bv))
    bn = cv2.Canny(bg_g, 150, 250)
    pn = cv2.Canny(pz_g, 150, 250)
    nr = cv2.matchTemplate(bn, pn, cv2.TM_CCOEFF_NORMED)
    npc_val, _, _, nl = cv2.minMaxLoc(nr)
    cn = max(0.0, min(1.0, npc_val))
    if cm >= 0.35 or abs(xm - nl[0]) <= 5: return xm, ym
    # Pick higher confidence between MC and NPC
    if cn > cm: return nl[0], nl[1]
    return xm, ym


def detect_v4e_ensemble(background, puzzle):
    """实验 E: MC + NPC + Sobel 投票，去掉 SSD。"""
    bg = cv2.normalize(background, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pz = cv2.normalize(puzzle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    bg_g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    pz_g = cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY)
    bv, bl = -1.0, (0, 0)
    for lo, hi in [(30, 100), (50, 150), (80, 180), (100, 200), (120, 240), (150, 250), (180, 300)]:
        be = cv2.Canny(bg_g, lo, hi)
        pe = cv2.Canny(pz_g, lo, hi)
        r = cv2.matchTemplate(be, pe, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(r)
        if mv > bv: bv, bl = mv, ml
    xm, ym = bl
    cm = max(0.0, min(1.0, bv))
    bn = cv2.Canny(bg_g, 150, 250)
    pn = cv2.Canny(pz_g, 150, 250)
    nr = cv2.matchTemplate(bn, pn, cv2.TM_CCOEFF_NORMED)
    npc_val, _, _, nl = cv2.minMaxLoc(nr)
    if cm >= 0.35 or abs(xm - nl[0]) <= 5: return xm, ym
    # Sobel
    sx_bg = cv2.Sobel(bg_g, cv2.CV_64F, 1, 0, ksize=3)
    sy_bg = cv2.Sobel(bg_g, cv2.CV_64F, 0, 1, ksize=3)
    mag_bg = np.uint8(np.clip(np.sqrt(sx_bg**2 + sy_bg**2), 0, 255))
    sx_pz = cv2.Sobel(pz_g, cv2.CV_64F, 1, 0, ksize=3)
    sy_pz = cv2.Sobel(pz_g, cv2.CV_64F, 0, 1, ksize=3)
    mag_pz = np.uint8(np.clip(np.sqrt(sx_pz**2 + sy_pz**2), 0, 255))
    sobel_r = cv2.matchTemplate(mag_bg, mag_pz, cv2.TM_CCOEFF_NORMED)
    _, sobel_val, _, sobel_loc = cv2.minMaxLoc(sobel_r)
    # Vote: MC, NPC, Sobel
    voters = [(xm, ym), (nl[0], nl[1]), (sobel_loc[0], sobel_loc[1])]
    # Find cluster
    used = [False, False, False]
    best_size, best_med = 0, (xm, ym)
    for i in range(3):
        if used[i]: continue
        cluster = [i]; used[i] = True
        for j in range(i+1, 3):
            if used[j]: continue
            if math.hypot(voters[i][0]-voters[j][0], voters[i][1]-voters[j][1]) <= 5:
                cluster.append(j); used[j] = True
        if len(cluster) > best_size:
            best_size = len(cluster)
            xs = [voters[k][0] for k in cluster]
            ys = [voters[k][1] for k in cluster]
            best_med = (int(np.median(xs)), int(np.median(ys)))
    if best_size >= 2: return best_med
    # No cluster: use best confidence
    confs = [cm, max(0, min(1, npc_val)), max(0, min(1, sobel_val))]
    best_idx = int(np.argmax(confs))
    return voters[best_idx]


def detect_v4f_ssd_normalized(background, puzzle):
    """实验 F: v4f 原版但 SSD 使用归一化后的图像。"""
    bg = cv2.normalize(background, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pz = cv2.normalize(puzzle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    bg_g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    pz_g = cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY)
    bv, bl = -1.0, (0, 0)
    for lo, hi in [(30, 100), (50, 150), (80, 180), (100, 200), (120, 240), (150, 250), (180, 300)]:
        be = cv2.Canny(bg_g, lo, hi)
        pe = cv2.Canny(pz_g, lo, hi)
        r = cv2.matchTemplate(be, pe, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(r)
        if mv > bv: bv, bl = mv, ml
    xm, ym = bl
    cm = max(0.0, min(1.0, bv))
    bn = cv2.Canny(bg_g, 150, 250)
    pn = cv2.Canny(pz_g, 150, 250)
    nr = cv2.matchTemplate(bn, pn, cv2.TM_CCOEFF_NORMED)
    _, _, _, nl = cv2.minMaxLoc(nr)
    if cm >= 0.35 or abs(xm - nl[0]) <= 5: return xm, ym
    # SSD on NORMALIZED grayscale (not raw background)
    _, mask = cv2.threshold(pz_g, 1, 255, cv2.THRESH_BINARY)
    mb = mask > 0
    ph, pw = pz_g.shape; bh, bw = bg_g.shape
    if ph > bh or pw > bw: return xm, ym
    npx = int(np.sum(mb))
    if npx == 0: return xm, ym
    be2, bx = 1e18, 0
    for x in range(0, bw - pw + 1, 2):
        err = float(np.sum(((bg_g[0:ph, x:x + pw].astype(float) - pz_g.astype(float)) ** 2)[mb])) / npx
        if err < be2: be2, bx = err, x
    for x in range(max(0, bx - 2), min(bw - pw + 1, bx + 3)):
        err = float(np.sum(((bg_g[0:ph, x:x + pw].astype(float) - pz_g.astype(float)) ** 2)[mb])) / npx
        if err < be2: be2, bx = err, x
    sc = max(0.0, 1.0 - be2 / 10000.0)
    if sc > 0.60:  # Lower threshold (normalized SSD has different scale)
        return bx, ym
    return xm, ym


# ============================================================================
#  测试框架
# ============================================================================

def run_test(method_name, method_fn, dataset_dir, cases, tolerance):
    ok = fail = 0
    total_time = 0.0
    desc = f"    {method_name:<30}"
    for c in tqdm(cases, desc=desc, ncols=95, leave=False,
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
        bg = cv2.imread(os.path.join(dataset_dir, c["background"]))
        pz = cv2.imread(os.path.join(dataset_dir, c["puzzle"]))
        if bg is None or pz is None: fail += 1; continue
        sx, sy = c["position"]
        t0 = time.perf_counter()
        rx, ry = method_fn(bg, pz)
        total_time += time.perf_counter() - t0
        if math.hypot(sx - rx, sy - ry) <= tolerance: ok += 1
        else: fail += 1
    n = ok + fail
    return {"ok": ok, "fail": fail, "total": n, "acc": ok/n if n else 0,
            "time": total_time, "speed": n/total_time if total_time > 0 else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", "-m", type=int, default=None)
    args = parser.parse_args()

    ds_dir = os.path.join(ROOT, "tests", "balanced_50_50")
    ds = json.load(open(os.path.join(ds_dir, "dataset.json"), encoding="utf-8"))
    tol = ds["error_tolerance"]
    cases = ds["cases"][:args.max_cases] if args.max_cases else ds["cases"]
    n = len(cases)

    methods = [
        ("NPC Baseline", npc.detect),
        ("v4f Original", detect_v4f_original),
        ("A: No SSD (NPC fallback)", detect_v4a_no_ssd),
        ("B: NPC fallback", detect_v4b_npc_fallback),
        ("C: SSD normalized", detect_v4c_normalized_ssd),
        ("D: MC or NPC (no SSD)", detect_v4d_mc_or_npc),
        ("E: Ensemble vote", detect_v4e_ensemble),
        ("F: SSD norm + lower thr", detect_v4f_ssd_normalized),
    ]

    print("=" * 90)
    print(f"Experiment: Balanced 50/50 ({n} cases, tol={tol})")
    print("=" * 90)
    print(f"  {'Method':<30} {'Accuracy':>10} {'OK':>6} {'Fail':>5} {'Time':>8} {'Speed':>10}")
    print(f"  {'-' * 72}")

    for m_name, m_fn in methods:
        r = run_test(m_name, m_fn, ds_dir, cases, tol)
        print(f"  {m_name:<30} {r['acc']:>9.1%} {r['ok']:>6} {r['fail']:>5} "
              f"{r['time']:>7.1f}s {r['speed']:>8.1f} i/s")


if __name__ == "__main__":
    main()
