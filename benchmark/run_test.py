# -*- coding: utf-8 -*-
"""
统一测试脚本 — 对比 NPC Baseline 与 Optimized v5 在各数据集上的准确率。
支持多进程并行、图片预加载、GPU 加速。

用法:
  python benchmark/run_test.py                          # 测试所有数据集 (自动检测 CPU 核数)
  python benchmark/run_test.py --dataset geetest_test   # 测试指定数据集
  python benchmark/run_test.py --dataset vcode_caltech_30k --tolerance 10
  python benchmark/run_test.py --dataset vcode_caltech_30k --max-cases 500  # 快速验证
  python benchmark/run_test.py --workers 8              # 指定并行进程数
  python benchmark/run_test.py --preload                # 预加载图片到内存
  python benchmark/run_test.py --gpu                    # 使用 GPU 加速
  python benchmark/run_test.py --gpu --workers 4        # GPU + 多进程
"""
import json, math, os, sys, time, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import benchmark.algorithms as algo

# 方法注册表（供多进程 worker 使用）
METHODS = {
    "NPC Baseline": algo.detect_npc,
    "Optimized v4f": algo.detect_v4f,
    "Optimized v5": algo.detect_v5,
    "Optimized v6": algo.detect_v6,
    "NPC GPU": algo.detect_npc_gpu,
    "v4f GPU": algo.detect_v4f_gpu,
    "v5 GPU": algo.detect_v5_gpu,
    "v5 MT": algo.detect_v5_mt,
}


def _worker_run_batch(args):
    """
    批量样本检测任务，供进程池调用。每个 worker 处理一批样本以减少 IPC 开销。
    返回 [(is_ok, elapsed_time), ...]。
    """
    batch_cases, dataset_dir, method_key, tolerance, canny_workers = args
    method_fn = METHODS[method_key]
    results = []
    for bg_path, pz_path, sx, sy in batch_cases:
        bg = cv2.imread(bg_path)
        pz = cv2.imread(pz_path)
        if bg is None or pz is None:
            results.append((False, 0.0))
            continue
        t0 = time.perf_counter()
        if method_key == "v5 MT":
            rx, ry = method_fn(bg, pz, canny_workers=canny_workers)
        else:
            rx, ry = method_fn(bg, pz)
        elapsed = time.perf_counter() - t0
        results.append((math.hypot(sx - rx, sy - ry) <= tolerance, elapsed))
    return results


def _worker_run_batch_preloaded(args):
    """
    批量样本检测任务（图片已预加载），供进程池调用。
    """
    batch_data, method_key, tolerance, canny_workers = args
    method_fn = METHODS[method_key]
    results = []
    for bg_enc, pz_enc, sx, sy in batch_data:
        bg = cv2.imdecode(bg_enc, cv2.IMREAD_COLOR)
        pz = cv2.imdecode(pz_enc, cv2.IMREAD_COLOR)
        if bg is None or pz is None:
            results.append((False, 0.0))
            continue
        t0 = time.perf_counter()
        if method_key == "v5 MT":
            rx, ry = method_fn(bg, pz, canny_workers=canny_workers)
        else:
            rx, ry = method_fn(bg, pz)
        elapsed = time.perf_counter() - t0
        results.append((math.hypot(sx - rx, sy - ry) <= tolerance, elapsed))
    return results


def preload_images(dataset_dir, cases):
    """
    预加载所有图片到内存，返回编码后的 bytes 列表。
    使用 cv2.imencode 避免跨进程序列化大 numpy 数组。
    """
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


def run_single(method_name, method_fn, dataset_dir, cases, tolerance,
               workers=1, preload=False, canny_workers=7):
    """
    运行单个方法。
    workers=1 时使用原始串行逻辑，>1 时使用多进程并行。
    """
    method_key = method_name
    if method_key not in METHODS:
        # 注册自定义方法名（如带后缀的 GPU 版本）
        METHODS[method_key] = method_fn

    if workers <= 1:
        # 串行模式（原始逻辑）
        ok = fail = 0
        total_time = 0.0
        for c in tqdm(cases, desc=f"    {method_name:<20}", ncols=90, leave=False,
                      bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                                 "[{elapsed}<{remaining}, {rate_fmt}]"):
            bg = cv2.imread(os.path.join(dataset_dir, c["background"]))
            pz = cv2.imread(os.path.join(dataset_dir, c["puzzle"]))
            if bg is None or pz is None:
                fail += 1
                continue
            sx, sy = c["position"]
            t0 = time.perf_counter()
            if method_key == "v5 MT":
                rx, ry = method_fn(bg, pz, canny_workers=canny_workers)
            else:
                rx, ry = method_fn(bg, pz)
            total_time += time.perf_counter() - t0
            if math.hypot(sx - rx, sy - ry) <= tolerance:
                ok += 1
            else:
                fail += 1
        n = ok + fail
        return {"ok": ok, "fail": fail, "total": n,
                "acc": ok / n if n else 0, "time": total_time,
                "speed": n / total_time if total_time > 0 else 0}

    # 多进程模式（批量处理以减少 IPC 开销）
    ok = fail = 0
    total_time = 0.0
    n = len(cases)

    # 计算批量大小：每个 worker 至少处理 batch_size 个样本
    batch_size = max(1, n // workers)
    if batch_size * workers < n:
        batch_size += 1

    # 准备批量任务
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
                batches.append((batch_data, method_key, tolerance, canny_workers))
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
                batches.append((batch_cases, dataset_dir, method_key, tolerance, canny_workers))
        worker_fn = _worker_run_batch

    wall_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_fn, batch) for batch in batches]

        with tqdm(total=n, desc=f"    {method_name:<20}", ncols=90, leave=False,
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                             "[{elapsed}<{remaining}, {rate_fmt}]") as pbar:
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
    # 使用墙钟时间计算速度（多进程的实际吞吐量）
    return {"ok": ok, "fail": fail, "total": n,
            "acc": ok / n if n else 0, "time": wall_time,
            "speed": n / wall_time if wall_time > 0 else 0}


def main():
    parser = argparse.ArgumentParser(description="Test captcha detection algorithms")
    parser.add_argument("--dataset", "-d", default=None,
                        help="Dataset name (e.g. geetest_test). Omit to test all.")
    parser.add_argument("--tolerance", "-t", type=int, default=None,
                        help="Override tolerance (default: from dataset.json)")
    parser.add_argument("--max-cases", "-m", type=int, default=None,
                        help="Max cases per dataset (for quick testing)")
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="Parallel worker processes (default: CPU core count)")
    parser.add_argument("--canny-workers", type=int, default=7,
                        help="Thread pool size for Canny matching (default: 7)")
    parser.add_argument("--preload", action="store_true",
                        help="Preload all images into memory before testing")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU acceleration (requires OpenCL/CUDA support)")
    args = parser.parse_args()

    # 自动检测 CPU 核数
    if args.workers is None:
        args.workers = min(os.cpu_count() or 4, 8)

    tests_dir = os.path.join(ROOT, "tests")

    # 根据 --gpu 选择方法集
    if args.gpu:
        if algo._has_gpu():
            methods = [
                ("NPC GPU", algo.detect_npc_gpu),
                ("v4f GPU", algo.detect_v4f_gpu),
                ("v5 GPU", algo.detect_v5_gpu),
            ]
            print(f"[GPU] OpenCL/CUDA 加速已启用")
        else:
            print("[GPU] 警告: 未检测到 GPU 支持，回退到 CPU 模式")
            methods = [
                ("NPC Baseline", algo.detect_npc),
                ("Optimized v4f", algo.detect_v4f),
                ("Optimized v5", algo.detect_v5),
                ("Optimized v6", algo.detect_v6),
            ]
    else:
        methods = [
            ("NPC Baseline", algo.detect_npc),
            ("Optimized v4f", algo.detect_v4f),
            ("Optimized v5", algo.detect_v5),
            ("Optimized v6", algo.detect_v6),
        ]

    print(f"[Config] workers={args.workers}, canny_workers={args.canny_workers}, "
          f"preload={args.preload}, gpu={args.gpu}")

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = sorted(
            d for d in os.listdir(tests_dir)
            if os.path.isdir(os.path.join(tests_dir, d))
            and os.path.exists(os.path.join(tests_dir, d, "dataset.json"))
        )

    totals = {m[0]: {"ok": 0, "total": 0, "time": 0} for m in methods}

    print("=" * 85)
    print("TncodeSolver Algorithm Benchmark")
    print("=" * 85)

    for ds_name in tqdm(datasets, desc="Datasets", ncols=90,
                        bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}|"):
        ds_dir = os.path.join(tests_dir, ds_name)
        json_path = os.path.join(ds_dir, "dataset.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, encoding="utf-8") as f:
            ds = json.load(f)
        tol = args.tolerance or ds.get("error_tolerance", 5)
        cases = ds["cases"][:args.max_cases] if args.max_cases else ds["cases"]
        n = len(cases)

        tqdm.write(f"\n  {ds.get('name', ds_name)} ({n} cases, tol={tol})")
        tqdm.write(f"  {'Method':<20} {'Accuracy':>10} {'OK':>6} {'Fail':>5} {'Time':>8} {'Speed':>10}")
        tqdm.write(f"  {'-' * 62}")

        for m_name, m_fn in methods:
            r = run_single(m_name, m_fn, ds_dir, cases, tol,
                           workers=args.workers, preload=args.preload,
                           canny_workers=args.canny_workers)
            totals[m_name]["ok"] += r["ok"]
            totals[m_name]["total"] += r["total"]
            totals[m_name]["time"] += r["time"]
            tqdm.write(f"  {m_name:<20} {r['acc']:>9.1%} {r['ok']:>6} {r['fail']:>5} "
                       f"{r['time']:>7.1f}s {r['speed']:>8.1f} i/s")

    print("\n" + "=" * 75)
    print("TOTALS")
    print("=" * 75)
    print(f"  {'Method':<20} {'Accuracy':>10} {'Cases':>12} {'Time':>8} {'Speed':>10}")
    print(f"  {'-' * 65}")
    for m_name, t in totals.items():
        acc = t["ok"] / t["total"] if t["total"] else 0
        spd = t["total"] / t["time"] if t["time"] else 0
        print(f"  {m_name:<20} {acc:>9.1%} {t['ok']:>5}/{t['total']:<5} "
              f"{t['time']:>6.1f}s {spd:>8.1f} i/s")


if __name__ == "__main__":
    main()
