# -*- coding: utf-8 -*-
"""
实验：针对 v4f 失败模式测试多种替代策略。
在 tests/balanced_50_50 上测试，目标：准确率显著高于 50%。

用法:
  python benchmark/experiment.py
  python benchmark/experiment.py --max-cases 500
"""
import json, math, os, sys, time, argparse
import cv2
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import benchmark.algorithms as algo


def _mc_and_npc(background, puzzle):
    """共享的 MC + NPC 预处理，返回 (mx, my, mc, nx, ny, nc)。"""
    bg_g, pz_g = algo._preprocess(background, puzzle)
    mx, my, mc = algo._multi_canny_match(bg_g, pz_g)
    nx, ny, nc = algo._npc_match(bg_g, pz_g)
    return mx, my, mc, nx, ny, nc


def detect_v4f_original(background, puzzle):
    """v4f: MC + NPC + SSD 兜底。"""
    mx, my, mc, nx, ny, nc = _mc_and_npc(background, puzzle)
    if mc >= algo.CONFIDENCE_THRESHOLD or abs(mx - nx) <= algo.AGREEMENT_TOLERANCE:
        return mx, my
    # SSD
    bg_g = cv2.cvtColor(
        cv2.normalize(background, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX),
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float64)
    pz_g = cv2.cvtColor(
        cv2.normalize(puzzle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX),
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float64)
    _, mask = cv2.threshold(pz_g.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
    mb = mask > 0
    ph, pw = pz_g.shape
    bh, bw = bg_g.shape
    if ph > bh or pw > bw:
        return mx, my
    npx = int(np.sum(mb))
    if npx == 0:
        return mx, my
    be2, bx = 1e18, 0
    for x in range(0, bw - pw + 1, 2):
        err = float(np.sum(((bg_g[0:ph, x:x + pw] - pz_g) ** 2)[mb])) / npx
        if err < be2:
            be2, bx = err, x
    for x in range(max(0, bx - 2), min(bw - pw + 1, bx + 3)):
        err = float(np.sum(((bg_g[0:ph, x:x + pw] - pz_g) ** 2)[mb])) / npx
        if err < be2:
            be2, bx = err, x
    if max(0.0, 1.0 - be2 / 10000.0) > 0.80:
        return bx, my
    return mx, my


def detect_no_ssd_npc_fallback(background, puzzle):
    """去掉 SSD，低置信时直接用 NPC。"""
    mx, my, mc, nx, ny, nc = _mc_and_npc(background, puzzle)
    if mc >= algo.CONFIDENCE_THRESHOLD or abs(mx - nx) <= algo.AGREEMENT_TOLERANCE:
        return mx, my
    return nx, ny


def detect_confidence_select(background, puzzle):
    """MC 和 NPC 取置信度更高者 (= v5)。"""
    mx, my, mc, nx, ny, nc = _mc_and_npc(background, puzzle)
    if mc >= algo.CONFIDENCE_THRESHOLD or abs(mx - nx) <= algo.AGREEMENT_TOLERANCE:
        return mx, my
    return (nx, ny) if nc > mc else (mx, my)


def detect_ensemble_vote(background, puzzle):
    """MC + NPC + Sobel 三方法投票。"""
    bg_g, pz_g = algo._preprocess(background, puzzle)
    mx, my, mc = algo._multi_canny_match(bg_g, pz_g)
    nx, ny, nc = algo._npc_match(bg_g, pz_g)

    if mc >= algo.CONFIDENCE_THRESHOLD or abs(mx - nx) <= algo.AGREEMENT_TOLERANCE:
        return mx, my

    # Sobel
    sx = cv2.Sobel(bg_g, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(bg_g, cv2.CV_64F, 0, 1, ksize=3)
    mag_bg = np.uint8(np.clip(np.sqrt(sx**2 + sy**2), 0, 255))
    sxp = cv2.Sobel(pz_g, cv2.CV_64F, 1, 0, ksize=3)
    syp = cv2.Sobel(pz_g, cv2.CV_64F, 0, 1, ksize=3)
    mag_pz = np.uint8(np.clip(np.sqrt(sxp**2 + syp**2), 0, 255))
    r = cv2.matchTemplate(mag_bg, mag_pz, cv2.TM_CCOEFF_NORMED)
    sobel_val, _, _, sobel_loc = cv2.minMaxLoc(r)

    voters = [(mx, my), (nx, ny), (sobel_loc[0], sobel_loc[1])]
    used = [False, False, False]
    best_size, best_med = 0, (mx, my)
    for i in range(3):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, 3):
            if not used[j] and math.hypot(voters[i][0] - voters[j][0],
                                           voters[i][1] - voters[j][1]) <= 5:
                cluster.append(j)
                used[j] = True
        if len(cluster) > best_size:
            best_size = len(cluster)
            xs = [voters[k][0] for k in cluster]
            ys = [voters[k][1] for k in cluster]
            best_med = (int(np.median(xs)), int(np.median(ys)))
    if best_size >= 2:
        return best_med
    confs = [mc, nc, max(0.0, min(1.0, sobel_val))]
    return voters[int(np.argmax(confs))]


def run_test(method_name, method_fn, dataset_dir, cases, tolerance):
    ok = fail = 0
    total_time = 0.0
    for c in tqdm(cases, desc=f"    {method_name:<30}", ncols=95, leave=False,
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"):
        bg = cv2.imread(os.path.join(dataset_dir, c["background"]))
        pz = cv2.imread(os.path.join(dataset_dir, c["puzzle"]))
        if bg is None or pz is None:
            fail += 1
            continue
        sx, sy = c["position"]
        t0 = time.perf_counter()
        rx, ry = method_fn(bg, pz)
        total_time += time.perf_counter() - t0
        if math.hypot(sx - rx, sy - ry) <= tolerance:
            ok += 1
        else:
            fail += 1
    n = ok + fail
    return {"ok": ok, "fail": fail, "total": n, "acc": ok / n if n else 0,
            "time": total_time, "speed": n / total_time if total_time > 0 else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", "-m", type=int, default=None)
    args = parser.parse_args()

    ds_dir = os.path.join(ROOT, "tests", "balanced_50_50")
    with open(os.path.join(ds_dir, "dataset.json"), encoding="utf-8") as f:
        ds = json.load(f)
    tol = ds["error_tolerance"]
    cases = ds["cases"][:args.max_cases] if args.max_cases else ds["cases"]
    n = len(cases)

    methods = [
        ("NPC Baseline", algo.detect_npc),
        ("v4f Original (SSD)", detect_v4f_original),
        ("No SSD + NPC fallback", detect_no_ssd_npc_fallback),
        ("Confidence select (=v5)", detect_confidence_select),
        ("Ensemble vote", detect_ensemble_vote),
    ]

    print("=" * 90)
    print(f"Experiment: {ds.get('name', 'balanced_50_50')} ({n} cases, tol={tol})")
    print("=" * 90)
    print(f"  {'Method':<30} {'Accuracy':>10} {'OK':>6} {'Fail':>5} {'Time':>8} {'Speed':>10}")
    print(f"  {'-' * 72}")

    for m_name, m_fn in methods:
        r = run_test(m_name, m_fn, ds_dir, cases, tol)
        print(f"  {m_name:<30} {r['acc']:>9.1%} {r['ok']:>6} {r['fail']:>5} "
              f"{r['time']:>7.1f}s {r['speed']:>8.1f} i/s")


if __name__ == "__main__":
    main()
