# -*- coding: utf-8 -*-
"""
实验：针对 v4f 失败模式测试多种替代策略。
在 tests/balanced_50_50 上测试，目标：准确率显著高于 50%。
支持多进程并行、图片预加载、GPU 加速。

用法:
  python benchmark/experiment.py
  python benchmark/experiment.py --max-cases 500
  python benchmark/experiment.py --workers 8
  python benchmark/experiment.py --preload --workers 4
"""
import json, math, os, sys, time, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import benchmark.algorithms as algo

# 实验方法注册表
_EXP_METHODS = {}


def _register_exp_methods():
    """注册所有实验方法到全局表。"""
    _EXP_METHODS["NPC Baseline"] = algo.detect_npc
    _EXP_METHODS["v4f Original (SSD)"] = detect_v4f_original
    _EXP_METHODS["No SSD + NPC fallback"] = detect_no_ssd_npc_fallback
    _EXP_METHODS["Confidence select (=v5)"] = detect_confidence_select
    _EXP_METHODS["Ensemble vote"] = detect_ensemble_vote


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


def _worker_run_batch(args):
    """批量样本检测任务，供进程池调用。"""
    batch_cases, dataset_dir, method_key, tolerance = args
    method_fn = _EXP_METHODS[method_key]
    results = []
    for bg_path, pz_path, sx, sy in batch_cases:
        bg = cv2.imread(bg_path)
        pz = cv2.imread(pz_path)
        if bg is None or pz is None:
            results.append((False, 0.0))
            continue
        t0 = time.perf_counter()
        rx, ry = method_fn(bg, pz)
        elapsed = time.perf_counter() - t0
        results.append((math.hypot(sx - rx, sy - ry) <= tolerance, elapsed))
    return results


def _worker_run_batch_preloaded(args):
    """批量样本检测任务（图片已预加载）。"""
    batch_data, method_key, tolerance = args
    method_fn = _EXP_METHODS[method_key]
    results = []
    for bg_enc, pz_enc, sx, sy in batch_data:
        bg = cv2.imdecode(bg_enc, cv2.IMREAD_COLOR)
        pz = cv2.imdecode(pz_enc, cv2.IMREAD_COLOR)
        if bg is None or pz is None:
            results.append((False, 0.0))
            continue
        t0 = time.perf_counter()
        rx, ry = method_fn(bg, pz)
        elapsed = time.perf_counter() - t0
        results.append((math.hypot(sx - rx, sy - ry) <= tolerance, elapsed))
    return results


def preload_images(dataset_dir, cases):
    """预加载所有图片到内存。"""
    preloaded = []
    for c in cases:
        bg_path = os.path.join(dataset_dir, c["background"])
        pz_path = os.path.join(dataset_dir, c["puzzle"])
        bg = cv2.imread(bg_path)
        pz = cv2.imread(pz_path)
        if bg is not None and pz is not None:
            _, bg_enc = cv2.imencode(".png", bg)
            _, pz_enc = cv2.imencode(".png", pz)
            preloaded.append((bg_enc, pz_enc))
        else:
            preloaded.append(None)
    return preloaded


def run_test(method_name, method_fn, dataset_dir, cases, tolerance,
             workers=1, preload=False):
    """运行单个方法，支持多进程并行。"""
    _EXP_METHODS[method_name] = method_fn

    if workers <= 1:
        # 串行模式
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

    # 多进程模式（批量处理以减少 IPC 开销）
    ok = fail = 0
    total_time = 0.0
    n = len(cases)

    # 计算批量大小
    batch_size = max(1, n // workers)
    if batch_size * workers < n:
        batch_size += 1

    if preload:
        preloaded = preload_images(dataset_dir, cases)
        batches = []
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_data = []
            for i in range(batch_start, batch_end):
                if preloaded[i] is not None:
                    bg_enc, pz_enc = preloaded[i]
                    sx, sy = cases[i]["position"]
                    batch_data.append((bg_enc, pz_enc, sx, sy))
            if batch_data:
                batches.append((batch_data, method_name, tolerance))
        worker_fn = _worker_run_batch_preloaded
    else:
        batches = []
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_cases = []
            for i in range(batch_start, batch_end):
                c = cases[i]
                bg_path = os.path.join(dataset_dir, c["background"])
                pz_path = os.path.join(dataset_dir, c["puzzle"])
                sx, sy = c["position"]
                batch_cases.append((bg_path, pz_path, sx, sy))
            if batch_cases:
                batches.append((batch_cases, dataset_dir, method_name, tolerance))
        worker_fn = _worker_run_batch

    wall_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_fn, batch) for batch in batches]

        with tqdm(total=n, desc=f"    {method_name:<30}", ncols=95, leave=False,
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}") as pbar:
            for future in as_completed(futures):
                batch_results = future.result()
                for is_ok, elapsed in batch_results:
                    total_time += elapsed
                    if is_ok:
                        ok += 1
                    else:
                        fail += 1
                pbar.update(len(batch_results))

    wall_time = time.perf_counter() - wall_start
    n = ok + fail
    return {"ok": ok, "fail": fail, "total": n, "acc": ok / n if n else 0,
            "time": wall_time, "speed": n / wall_time if wall_time > 0 else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", "-m", type=int, default=None)
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="Parallel worker processes (default: CPU core count)")
    parser.add_argument("--preload", action="store_true",
                        help="Preload all images into memory before testing")
    args = parser.parse_args()

    if args.workers is None:
        args.workers = min(os.cpu_count() or 4, 8)

    _register_exp_methods()

    print(f"[Config] workers={args.workers}, preload={args.preload}")

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
        r = run_test(m_name, m_fn, ds_dir, cases, tol,
                     workers=args.workers, preload=args.preload)
        print(f"  {m_name:<30} {r['acc']:>9.1%} {r['ok']:>6} {r['fail']:>5} "
              f"{r['time']:>7.1f}s {r['speed']:>8.1f} i/s")


if __name__ == "__main__":
    main()
