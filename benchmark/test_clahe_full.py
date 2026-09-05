# -*- coding: utf-8 -*-
"""Sweep CLAHE clipLimit from 2.0 to 10.0 on ALL 11 datasets"""
import json, math, os, sys, time, cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import benchmark.algorithms as algo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.environ['OPENCV_NUM_THREADS'] = '1'

EXTENDED = [
    (20, 80), (30, 100), (50, 150), (80, 180), (100, 200),
    (120, 240), (150, 250), (180, 300), (200, 350),
]

def _run_one(args):
    bg_path, pz_path, sx, sy, clip, tol = args
    bg = cv2.imread(bg_path)
    pz = cv2.imread(pz_path)
    if bg is None or pz is None:
        return False
    bg = cv2.normalize(bg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pz = cv2.normalize(pz, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    bg_g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    pz_g = cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY)
    if clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        bg_g = clahe.apply(bg_g)
        pz_g = clahe.apply(pz_g)
    bv, bl = -1.0, (0, 0)
    for lo, hi in EXTENDED:
        r = cv2.matchTemplate(cv2.Canny(bg_g, lo, hi), cv2.Canny(pz_g, lo, hi), cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(r)
        if mv > bv: bv, bl = mv, ml
    mc = max(0.0, min(1.0, bv))
    r2 = cv2.matchTemplate(cv2.Canny(bg_g, 150, 250), cv2.Canny(pz_g, 150, 250), cv2.TM_CCOEFF_NORMED)
    nv, _, _, nl = cv2.minMaxLoc(r2)
    nc = max(0.0, min(1.0, nv))
    if mc >= 0.35 or abs(bl[0] - nl[0]) <= 5:
        rx, ry = bl[0], bl[1]
    elif nc > mc:
        rx, ry = nl[0], nl[1]
    else:
        rx, ry = bl[0], bl[1]
    return math.hypot(sx - rx, sy - ry) <= tol

def test_clip(clip, dataset_dir, cases, tol, workers=8):
    tasks = [(os.path.join(dataset_dir, c['background']),
              os.path.join(dataset_dir, c['puzzle']),
              c['position'][0], c['position'][1], clip, tol) for c in cases]
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_one, t): i for i, t in enumerate(tasks)}
        for f in tqdm(as_completed(futures), total=len(tasks),
                       desc=f'  clip={clip:<4}', ncols=70, leave=False):
            if f.result():
                ok += 1
    return ok, len(cases)

# Load all datasets
tests_dir = 'tests'
datasets = sorted(d for d in os.listdir(tests_dir)
                  if os.path.isdir(os.path.join(tests_dir, d))
                  and os.path.exists(os.path.join(tests_dir, d, 'dataset.json')))

all_ds = {}
for ds_name in datasets:
    ds_dir = os.path.join(tests_dir, ds_name)
    with open(os.path.join(ds_dir, 'dataset.json'), encoding='utf-8') as f:
        ds = json.load(f)
    all_ds[ds_name] = (ds_dir, ds['cases'], ds.get('error_tolerance', 5))

clips = [0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

print('=' * 130)
print('CLAHE clipLimit Sweep (extended 9 thresholds) on ALL 11 datasets')
print('=' * 130)

# Header
ds_names = list(all_ds.keys())
header = f"{'clip':<6}"
for name in ds_names:
    header += f" {name[:12]:>12}"
header += f" {'TOTAL':>10}"
print(header)
print('-' * len(header))

for clip in clips:
    line = f"{clip:<6}"
    total_ok = 0
    total_n = 0
    for ds_name, (ds_dir, cases, tol) in all_ds.items():
        ok, n = test_clip(clip, ds_dir, cases, tol)
        total_ok += ok
        total_n += n
        line += f" {ok/n:>11.1%}"
    line += f" {total_ok/total_n:>9.1%}"
    print(line)

print('=' * 130)
