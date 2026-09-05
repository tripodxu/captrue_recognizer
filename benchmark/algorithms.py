# -*- coding: utf-8 -*-
"""
核心检测算法 — 供 benchmark 脚本和 tncode_solver.py 共用。
包含 NPC Baseline、v4f 和 v5 的纯函数实现。
支持多线程加速和 GPU 加速。
"""
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor


# 常量
CANNY_THRESHOLDS = [
    (30, 100), (50, 150), (80, 180), (100, 200),
    (120, 240), (150, 250), (180, 300),
]
CANNY_THRESHOLDS_V6 = [
    (20, 80), (30, 100), (50, 150), (80, 180), (100, 200),
    (120, 240), (150, 250), (180, 300), (200, 350),
]
NPC_LOW, NPC_HIGH = 150, 250
CONFIDENCE_THRESHOLD = 0.35
AGREEMENT_TOLERANCE = 5
SSD_CONFIDENCE_THRESHOLD = 0.80
CLAHE_CLIP_V6 = 8.0


def _preprocess(background, puzzle):
    """归一化 + 灰度转换。"""
    bg = cv2.normalize(background, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pz = cv2.normalize(puzzle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY)


def _preprocess_clahe(background, puzzle, clip=CLAHE_CLIP_V6):
    """归一化 + CLAHE 自适应直方图均衡化 + 灰度转换。"""
    bg = cv2.normalize(background, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pz = cv2.normalize(puzzle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    pz_gray = cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    bg_gray = clahe.apply(bg_gray)
    pz_gray = clahe.apply(pz_gray)
    return bg_gray, pz_gray


def _multi_canny_match(bg_gray, pz_gray, thresholds=None):
    """多阈值 Canny 模板匹配，返回 (x, y, confidence)。"""
    if thresholds is None:
        thresholds = CANNY_THRESHOLDS
    best_val, best_loc = -1.0, (0, 0)
    for lo, hi in thresholds:
        bg_e = cv2.Canny(bg_gray, lo, hi)
        pz_e = cv2.Canny(pz_gray, lo, hi)
        result = cv2.matchTemplate(bg_e, pz_e, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_val:
            best_val, best_loc = max_val, max_loc
    return best_loc[0], best_loc[1], max(0.0, min(1.0, best_val))


def _canny_match_one(args):
    """单组阈值的 Canny + matchTemplate，供线程池调用。OpenCV C++ 释放 GIL。"""
    lo, hi, bg_gray, pz_gray = args
    bg_e = cv2.Canny(bg_gray, lo, hi)
    pz_e = cv2.Canny(pz_gray, lo, hi)
    result = cv2.matchTemplate(bg_e, pz_e, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_val, max_loc


def _multi_canny_match_mt(bg_gray, pz_gray, max_workers=7):
    """多阈值 Canny 模板匹配（多线程版），返回 (x, y, confidence)。"""
    tasks = [(lo, hi, bg_gray, pz_gray) for lo, hi in CANNY_THRESHOLDS]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_canny_match_one, tasks))
    best_val, best_loc = max(results, key=lambda r: r[0])
    return best_loc[0], best_loc[1], max(0.0, min(1.0, best_val))


# --- GPU 加速版 ---

def _has_gpu():
    """检测是否有可用的 GPU 加速后端。"""
    try:
        # 检查 OpenCL 支持
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            return True
    except Exception:
        pass
    try:
        # 检查 CUDA 支持
        cv2.cuda.getCudaEnabledDeviceCount()
        return True
    except Exception:
        pass
    return False


def _preprocess_gpu(background, puzzle):
    """归一化 + 灰度转换（GPU 版，使用 UMat）。"""
    bg = cv2.normalize(background, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pz = cv2.normalize(puzzle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    bg_umat = cv2.UMat(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY))
    pz_umat = cv2.UMat(cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY))
    return bg_umat, pz_umat


def _multi_canny_match_gpu(bg_umat, pz_umat):
    """多阈值 Canny 模板匹配（GPU 版，使用 UMat），返回 (x, y, confidence)。"""
    best_val, best_loc = -1.0, (0, 0)
    for lo, hi in CANNY_THRESHOLDS:
        bg_e = cv2.Canny(bg_umat, lo, hi)
        pz_e = cv2.Canny(pz_umat, lo, hi)
        result = cv2.matchTemplate(bg_e, pz_e, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_val:
            best_val, best_loc = max_val, max_loc
    return best_loc[0], best_loc[1], max(0.0, min(1.0, best_val))


def _npc_match_gpu(bg_umat, pz_umat):
    """NPC 单阈值匹配（GPU 版），返回 (x, y, confidence)。"""
    bg_e = cv2.Canny(bg_umat, NPC_LOW, NPC_HIGH)
    pz_e = cv2.Canny(pz_umat, NPC_LOW, NPC_HIGH)
    result = cv2.matchTemplate(bg_e, pz_e, cv2.TM_CCOEFF_NORMED)
    max_val, _, _, max_loc = cv2.minMaxLoc(result)
    return max_loc[0], max_loc[1], max(0.0, min(1.0, max_val))


def _npc_match(bg_gray, pz_gray):
    """NPC 单阈值匹配，返回 (x, y, confidence)。"""
    bg_e = cv2.Canny(bg_gray, NPC_LOW, NPC_HIGH)
    pz_e = cv2.Canny(pz_gray, NPC_LOW, NPC_HIGH)
    result = cv2.matchTemplate(bg_e, pz_e, cv2.TM_CCOEFF_NORMED)
    max_val, _, _, max_loc = cv2.minMaxLoc(result)
    return max_loc[0], max_loc[1], max(0.0, min(1.0, max_val))


def _ssd_fallback(bg_gray, pz_gray, mc_x, mc_y):
    """SSD 像素匹配兜底 (v4f 使用)。返回 (x, y)。"""
    bg_f = bg_gray.astype(np.float64)
    pz_f = pz_gray.astype(np.float64)
    _, mask = cv2.threshold(pz_gray, 1, 255, cv2.THRESH_BINARY)
    mb = mask > 0
    ph, pw = pz_f.shape
    bh, bw = bg_f.shape
    if ph > bh or pw > bw:
        return mc_x, mc_y
    npx = int(np.sum(mb))
    if npx == 0:
        return mc_x, mc_y
    be2, bx = 1e18, 0
    for x in range(0, bw - pw + 1, 2):
        err = float(np.sum(((bg_f[0:ph, x:x + pw] - pz_f) ** 2)[mb])) / npx
        if err < be2:
            be2, bx = err, x
    for x in range(max(0, bx - 2), min(bw - pw + 1, bx + 3)):
        err = float(np.sum(((bg_f[0:ph, x:x + pw] - pz_f) ** 2)[mb])) / npx
        if err < be2:
            be2, bx = err, x
    if max(0.0, 1.0 - be2 / 10000.0) > SSD_CONFIDENCE_THRESHOLD:
        return bx, mc_y
    return mc_x, mc_y


def detect_npc(background, puzzle):
    """
    NPC Baseline: Normalize → Canny(150,250) → matchTemplate(CCOEFF_NORMED)。
    返回 (x, y)。
    """
    bg_gray, pz_gray = _preprocess(background, puzzle)
    x, y, _ = _npc_match(bg_gray, pz_gray)
    return x, y


def detect_v4f(background, puzzle):
    """
    v4f: 多阈值 Canny + NPC 一致性校验 + SSD 像素匹配兜底。
    1. 多阈值 Canny (7 组) 取最高 CCOEFF_NORMED → MC
    2. NPC 单阈值 (150,250) → NPC
    3. MC conf >= 0.35 或 |MC.x - NPC.x| <= 5 → MC
    4. 否则 SSD 兜底 (conf > 0.80 时采纳)
    5. 兜底返回 MC
    返回 (x, y)。
    """
    bg_gray, pz_gray = _preprocess(background, puzzle)

    mx, my, mc = _multi_canny_match(bg_gray, pz_gray)
    nx, ny, nc = _npc_match(bg_gray, pz_gray)

    if mc >= CONFIDENCE_THRESHOLD or abs(mx - nx) <= AGREEMENT_TOLERANCE:
        return mx, my

    # SSD 兜底
    return _ssd_fallback(bg_gray, pz_gray, mx, my)


def detect_v5(background, puzzle):
    """
    v5: 多阈值 Canny + NPC 置信度择优 (无 SSD)。
    1. 多阈值 Canny (7 组) 取最高 CCOEFF_NORMED → MC
    2. NPC 单阈值 (150,250) → NPC
    3. MC conf >= 0.35 或 |MC.x - NPC.x| <= 5 → MC
    4. 否则取 MC/NPC 中置信度更高者
    返回 (x, y)。
    """
    bg_gray, pz_gray = _preprocess(background, puzzle)

    mx, my, mc = _multi_canny_match(bg_gray, pz_gray)
    nx, ny, nc = _npc_match(bg_gray, pz_gray)

    if mc >= CONFIDENCE_THRESHOLD or abs(mx - nx) <= AGREEMENT_TOLERANCE:
        return mx, my

    if nc > mc:
        return nx, ny
    return mx, my


def detect_v5_clahe(background, puzzle):
    """
    v5 + CLAHE 预处理：自适应直方图均衡化增强暗背景。
    返回 (x, y)。
    """
    bg_gray, pz_gray = _preprocess_clahe(background, puzzle)

    mx, my, mc = _multi_canny_match(bg_gray, pz_gray)
    nx, ny, nc = _npc_match(bg_gray, pz_gray)

    if mc >= CONFIDENCE_THRESHOLD or abs(mx - nx) <= AGREEMENT_TOLERANCE:
        return mx, my

    if nc > mc:
        return nx, ny
    return mx, my


def detect_v6(background, puzzle):
    """
    v6: CLAHE(8.0) + 扩展9组Canny阈值 + NPC 置信度择优。
    改进点:
    1. CLAHE clipLimit=8.0 增强局部对比度（改善暗背景）
    2. 9组阈值覆盖更广的边缘条件
    返回 (x, y)。
    """
    bg_gray, pz_gray = _preprocess_clahe(background, puzzle)

    mx, my, mc = _multi_canny_match(bg_gray, pz_gray, thresholds=CANNY_THRESHOLDS_V6)
    nx, ny, nc = _npc_match(bg_gray, pz_gray)

    if mc >= CONFIDENCE_THRESHOLD or abs(mx - nx) <= AGREEMENT_TOLERANCE:
        return mx, my

    if nc > mc:
        return nx, ny
    return mx, my


def detect_v6b(background, puzzle):
    """
    v6b: 双路径共识策略 — 同时运行 v5 (无CLAHE) 和 v6 (CLAHE+扩展阈值)。
    1. 两路径结果一致 (差<=5px) → 直接采用
    2. 一方与 NPC 一致 → 优先采用
    3. 否则取置信度更高者
    解决 v6 在 GeeTest/Tricky 上的回归，同时保留 Caltech/Balanced 提升。
    返回 (x, y)。
    """
    # Path 1: v5 (原阈值, 无CLAHE)
    bg_g1, pz_g1 = _preprocess(background, puzzle)
    mx1, my1, mc1 = _multi_canny_match(bg_g1, pz_g1, thresholds=CANNY_THRESHOLDS)

    # Path 2: v6 (CLAHE + 扩展阈值)
    bg_g2, pz_g2 = _preprocess_clahe(background, puzzle)
    mx2, my2, mc2 = _multi_canny_match(bg_g2, pz_g2, thresholds=CANNY_THRESHOLDS_V6)

    # NPC (共享)
    nx, ny, nc = _npc_match(bg_g1, pz_g1)

    # 两路径一致 → 直接采用
    if abs(mx1 - mx2) <= AGREEMENT_TOLERANCE:
        return mx1, my1

    # 一方与 NPC 一致 → 优先
    v5_agrees = abs(mx1 - nx) <= AGREEMENT_TOLERANCE
    v6_agrees = abs(mx2 - nx) <= AGREEMENT_TOLERANCE
    if v5_agrees and not v6_agrees:
        return mx1, my1
    if v6_agrees and not v5_agrees:
        return mx2, my2

    # 否则取置信度更高者
    if mc2 > mc1:
        return mx2, my2
    return mx1, my1


# --- 加速版 detect 函数 ---

def detect_v5_mt(background, puzzle, canny_workers=7):
    """
    v5 多线程加速版：多阈值 Canny 用线程池并行。
    返回 (x, y)。
    """
    bg_gray, pz_gray = _preprocess(background, puzzle)
    mx, my, mc = _multi_canny_match_mt(bg_gray, pz_gray, max_workers=canny_workers)
    nx, ny, nc = _npc_match(bg_gray, pz_gray)

    if mc >= CONFIDENCE_THRESHOLD or abs(mx - nx) <= AGREEMENT_TOLERANCE:
        return mx, my
    if nc > mc:
        return nx, ny
    return mx, my


def detect_v5_gpu(background, puzzle):
    """
    v5 GPU 加速版：使用 UMat 将 Canny/matchTemplate 卸载到 GPU。
    返回 (x, y)。
    """
    bg_umat, pz_umat = _preprocess_gpu(background, puzzle)
    mx, my, mc = _multi_canny_match_gpu(bg_umat, pz_umat)
    nx, ny, nc = _npc_match_gpu(bg_umat, pz_umat)

    if mc >= CONFIDENCE_THRESHOLD or abs(mx - nx) <= AGREEMENT_TOLERANCE:
        return mx, my
    if nc > mc:
        return nx, ny
    return mx, my


def detect_npc_gpu(background, puzzle):
    """NPC Baseline GPU 版。返回 (x, y)。"""
    bg_umat, pz_umat = _preprocess_gpu(background, puzzle)
    x, y, _ = _npc_match_gpu(bg_umat, pz_umat)
    return x, y


def detect_v4f_gpu(background, puzzle):
    """v4f GPU 版（SSD 部分仍在 CPU）。返回 (x, y)。"""
    bg_umat, pz_umat = _preprocess_gpu(background, puzzle)
    mx, my, mc = _multi_canny_match_gpu(bg_umat, pz_umat)
    nx, ny, nc = _npc_match_gpu(bg_umat, pz_umat)

    if mc >= CONFIDENCE_THRESHOLD or abs(mx - nx) <= AGREEMENT_TOLERANCE:
        return mx, my

    # SSD 兜底（需要 numpy 数组，从 UMat 取回）
    bg_gray = bg_umat.get() if hasattr(bg_umat, 'get') else bg_umat
    pz_gray = pz_umat.get() if hasattr(pz_umat, 'get') else pz_umat
    return _ssd_fallback(bg_gray, pz_gray, mx, my)
