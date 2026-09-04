# -*- coding: utf-8 -*-
"""
提取 Optimized v4f 的失败/成功用例，按 1:1 比例构建新的平衡测试集。
理论上 Optimized v4f 在该数据集上准确率应约为 50%。

用法:
  python benchmark/extract_balanced.py
  python benchmark/extract_balanced.py --tolerance 5
  python benchmark/extract_balanced.py --output tests/balanced_50_50
"""
import json, math, os, sys, random, argparse, shutil
import cv2
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import benchmark.npc_baseline as npc


def detect_optimized(background, puzzle):
    """优化算法 v4f。"""
    bg = cv2.normalize(background, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pz = cv2.normalize(puzzle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    bg_g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    pz_g = cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY)
    bv, bl = -1.0, (0, 0)
    for lo, hi in [(30, 100), (50, 150), (80, 180), (100, 200),
                   (120, 240), (150, 250), (180, 300)]:
        be = cv2.Canny(bg_g, lo, hi)
        pe = cv2.Canny(pz_g, lo, hi)
        r = cv2.matchTemplate(be, pe, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(r)
        if mv > bv:
            bv, bl = mv, ml
    xm, ym = bl
    cm = max(0.0, min(1.0, bv))
    bn = cv2.Canny(bg_g, 150, 250)
    pn = cv2.Canny(pz_g, 150, 250)
    nr = cv2.matchTemplate(bn, pn, cv2.TM_CCOEFF_NORMED)
    _, _, _, nl = cv2.minMaxLoc(nr)
    if cm >= 0.35 or abs(xm - nl[0]) <= 5:
        return xm, ym
    bg_f = bg_g.astype(np.float64)
    pz_f = pz_g.astype(np.float64)
    _, mask = cv2.threshold(pz_g, 1, 255, cv2.THRESH_BINARY)
    mb = mask > 0
    ph, pw = pz_f.shape
    bh, bw = bg_f.shape
    if ph > bh or pw > bw:
        return xm, ym
    npx = int(np.sum(mb))
    if npx == 0:
        return xm, ym
    be2, bx = 1e18, 0
    for x in range(0, bw - pw + 1, 2):
        err = float(np.sum(((bg_f[0:ph, x:x + pw] - pz_f) ** 2)[mb])) / npx
        if err < be2:
            be2, bx = err, x
    for x in range(max(0, bx - 2), min(bw - pw + 1, bx + 3)):
        err = float(np.sum(((bg_f[0:ph, x:x + pw] - pz_f) ** 2)[mb])) / npx
        if err < be2:
            be2, bx = err, x
    sc = max(0.0, 1.0 - be2 / 10000.0)
    if sc > 0.80:
        return bx, ym
    return xm, ym


def main():
    parser = argparse.ArgumentParser(description="Extract balanced fail/pass dataset")
    parser.add_argument("--output", "-o", default="tests/balanced_50_50",
                        help="Output directory (default: tests/balanced_50_50)")
    parser.add_argument("--tolerance", "-t", type=int, default=None,
                        help="Override tolerance (default: per-dataset)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    tests_dir = os.path.join(ROOT, "tests")
    output_dir = os.path.join(ROOT, args.output)
    os.makedirs(output_dir, exist_ok=True)

    # 发现数据集
    datasets = sorted([d for d in os.listdir(tests_dir)
                       if os.path.isdir(os.path.join(tests_dir, d))
                       and os.path.exists(os.path.join(tests_dir, d, "dataset.json"))
                       and d != "balanced_50_50"])

    print("=" * 70)
    print("Extracting balanced fail/pass dataset from Optimized v4f results")
    print("=" * 70)

    all_fail = []  # 失败用例 (带源信息)
    all_pass = []  # 成功用例 (带源信息)

    for ds_name in datasets:
        ds_dir = os.path.join(tests_dir, ds_name)
        ds = json.load(open(os.path.join(ds_dir, "dataset.json"), encoding="utf-8"))
        tol = args.tolerance or ds.get("error_tolerance", 5)
        cases = ds["cases"]
        n = len(cases)

        fail_list = []
        pass_list = []

        for c in tqdm(cases, desc=f"  {ds_name:<25}", ncols=85,
                      bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"):
            bg = cv2.imread(os.path.join(ds_dir, c["background"]))
            pz = cv2.imread(os.path.join(ds_dir, c["puzzle"]))
            if bg is None or pz is None:
                continue
            sx, sy = c["position"]
            rx, ry = detect_optimized(bg, pz)
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

            if err <= tol:
                pass_list.append(entry)
            else:
                fail_list.append(entry)

        all_fail.extend(fail_list)
        all_pass.extend(pass_list)
        tqdm.write(f"    {ds_name}: {len(pass_list)} pass, {len(fail_list)} fail")

    total_fail = len(all_fail)
    total_pass = len(all_pass)
    print(f"\n  Total: {total_fail} fail, {total_pass} pass")

    # 按 1:1 比例构建: 取全部失败 + 随机采样等量成功
    n_sample = min(total_fail, total_pass)
    random.shuffle(all_pass)
    selected_pass = all_pass[:n_sample]
    selected_fail = all_fail

    balanced = selected_fail + selected_pass
    random.shuffle(balanced)

    print(f"  Selected: {len(selected_fail)} fail + {len(selected_pass)} pass = {len(balanced)} total")

    # 复制图片到输出目录 (扁平结构, 重命名避免冲突)
    print(f"\n  Copying images to {args.output}/ ...")
    cases_out = []
    for i, entry in enumerate(tqdm(balanced, desc="  Copying", ncols=80,
                                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}")):
        bg_name = f"BAL_{i:05d}_O.png"
        pz_name = f"BAL_{i:05d}_P.png"

        src_bg = os.path.join(tests_dir, entry["background"])
        src_pz = os.path.join(tests_dir, entry["puzzle"])
        dst_bg = os.path.join(output_dir, bg_name)
        dst_pz = os.path.join(output_dir, pz_name)

        shutil.copy2(src_bg, dst_bg)
        shutil.copy2(src_pz, dst_pz)

        cases_out.append({
            "background": bg_name,
            "puzzle": pz_name,
            "position": entry["position"],
            "source_dataset": entry["source_dataset"],
            "is_fail_case": entry in selected_fail,
            "predicted": entry["predicted"],
            "error": entry["error"],
        })

    # 写 dataset.json
    dataset = {
        "name": "Balanced 50/50 (Optimized v4f fail+pass)",
        "description": "Equal fail/pass cases for Optimized v4f. Expected accuracy ~50%.",
        "error_tolerance": 5,
        "total_cases": len(cases_out),
        "fail_cases": len(selected_fail),
        "pass_cases": len(selected_pass),
        "source_datasets": datasets,
        "revision_date": "2026-09-04",
        "cases": cases_out,
    }

    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved {len(cases_out)} cases to {args.output}/")
    print(f"    Fail: {len(selected_fail)}")
    print(f"    Pass: {len(selected_pass)}")
    print(f"\n  Verify:")
    print(f"    python benchmark/run_test.py --dataset balanced_50_50")


if __name__ == "__main__":
    main()
