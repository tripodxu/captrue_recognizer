# -*- coding: utf-8 -*-
"""
统一测试脚本 — 对比 NPC Baseline 与 Optimized v5 在各数据集上的准确率。

用法:
  python benchmark/run_test.py                          # 测试所有数据集
  python benchmark/run_test.py --dataset geetest_test   # 测试指定数据集
  python benchmark/run_test.py --dataset vcode_caltech_30k --tolerance 10
  python benchmark/run_test.py --dataset vcode_caltech_30k --max-cases 500  # 快速验证
"""
import json, math, os, sys, time, argparse
import cv2
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import benchmark.algorithms as algo


def run_single(method_name, method_fn, dataset_dir, cases, tolerance):
    """运行单个方法，带 tqdm 进度条。"""
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
        rx, ry = method_fn(bg, pz)
        total_time += time.perf_counter() - t0
        if math.hypot(sx - rx, sy - ry) <= tolerance:
            ok += 1
        else:
            fail += 1

    n = ok + fail
    return {
        "ok": ok, "fail": fail, "total": n,
        "acc": ok / n if n else 0,
        "time": total_time,
        "speed": n / total_time if total_time > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Test captcha detection algorithms")
    parser.add_argument("--dataset", "-d", default=None,
                        help="Dataset name (e.g. geetest_test). Omit to test all.")
    parser.add_argument("--tolerance", "-t", type=int, default=None,
                        help="Override tolerance (default: from dataset.json)")
    parser.add_argument("--max-cases", "-m", type=int, default=None,
                        help="Max cases per dataset (for quick testing)")
    args = parser.parse_args()

    tests_dir = os.path.join(ROOT, "tests")
    methods = [
        ("NPC Baseline", algo.detect_npc),
        ("Optimized v4f", algo.detect_v4f),
        ("Optimized v5", algo.detect_v5),
    ]

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
            r = run_single(m_name, m_fn, ds_dir, cases, tol)
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
