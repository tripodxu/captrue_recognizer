# -*- coding: utf-8 -*-
"""
提取失败/成功用例，按 1:1 比例构建平衡测试集。

用法:
  python benchmark/extract_balanced.py
  python benchmark/extract_balanced.py --tolerance 5
  python benchmark/extract_balanced.py --output tests/balanced_50_50
"""
import json, math, os, sys, random, argparse, shutil
import cv2
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import benchmark.algorithms as algo


def main():
    parser = argparse.ArgumentParser(description="Extract balanced fail/pass dataset")
    parser.add_argument("--output", "-o", default="tests/balanced_50_50")
    parser.add_argument("--tolerance", "-t", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    tests_dir = os.path.join(ROOT, "tests")
    output_dir = os.path.join(ROOT, args.output)
    os.makedirs(output_dir, exist_ok=True)

    datasets = sorted(
        d for d in os.listdir(tests_dir)
        if os.path.isdir(os.path.join(tests_dir, d))
        and os.path.exists(os.path.join(tests_dir, d, "dataset.json"))
        and d != "balanced_50_50"
    )

    print("=" * 70)
    print("Extracting balanced fail/pass dataset")
    print("=" * 70)

    all_fail, all_pass = [], []

    for ds_name in datasets:
        ds_dir = os.path.join(tests_dir, ds_name)
        with open(os.path.join(ds_dir, "dataset.json"), encoding="utf-8") as f:
            ds = json.load(f)
        tol = args.tolerance or ds.get("error_tolerance", 5)
        fail_list, pass_list = [], []

        for c in tqdm(ds["cases"], desc=f"  {ds_name:<25}", ncols=85,
                      bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"):
            bg = cv2.imread(os.path.join(ds_dir, c["background"]))
            pz = cv2.imread(os.path.join(ds_dir, c["puzzle"]))
            if bg is None or pz is None:
                continue
            sx, sy = c["position"]
            rx, ry = algo.detect_v5(bg, pz)
            err = math.hypot(sx - rx, sy - ry)
            entry = {
                "background": os.path.join(ds_name, c["background"]),
                "puzzle": os.path.join(ds_name, c["puzzle"]),
                "position": c["position"],
                "source_dataset": ds_name,
                "tolerance": tol,
                "predicted": [rx, ry],
                "error": round(err, 2),
            }
            (fail_list if err > tol else pass_list).append(entry)

        all_fail.extend(fail_list)
        all_pass.extend(pass_list)
        tqdm.write(f"    {ds_name}: {len(pass_list)} pass, {len(fail_list)} fail")

    print(f"\n  Total: {len(all_fail)} fail, {len(all_pass)} pass")

    n_sample = min(len(all_fail), len(all_pass))
    random.shuffle(all_pass)
    selected_pass = all_pass[:n_sample]
    balanced = all_fail + selected_pass
    random.shuffle(balanced)

    print(f"  Selected: {len(all_fail)} fail + {len(selected_pass)} pass = {len(balanced)} total")

    # 复制图片
    print(f"\n  Copying images to {args.output}/ ...")
    cases_out = []
    for i, entry in enumerate(tqdm(balanced, desc="  Copying", ncols=80,
                                    bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                                               "{n_fmt}/{total_fmt}")):
        bg_name = f"BAL_{i:05d}_O.png"
        pz_name = f"BAL_{i:05d}_P.png"
        shutil.copy2(os.path.join(tests_dir, entry["background"]),
                     os.path.join(output_dir, bg_name))
        shutil.copy2(os.path.join(tests_dir, entry["puzzle"]),
                     os.path.join(output_dir, pz_name))
        cases_out.append({
            "background": bg_name, "puzzle": pz_name,
            "position": entry["position"],
            "source_dataset": entry["source_dataset"],
            "is_fail_case": entry in all_fail,
            "predicted": entry["predicted"],
            "error": entry["error"],
        })

    dataset = {
        "name": "Balanced 50/50 (fail+pass)",
        "error_tolerance": 5,
        "total_cases": len(cases_out),
        "fail_cases": len(all_fail),
        "pass_cases": len(selected_pass),
        "source_datasets": datasets,
        "revision_date": "2026-09-04",
        "cases": cases_out,
    }
    with open(os.path.join(output_dir, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved {len(cases_out)} cases to {args.output}/")
    print(f"  Verify: python benchmark/run_test.py --dataset balanced_50_50")


if __name__ == "__main__":
    main()
