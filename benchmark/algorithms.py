# -*- coding: utf-8 -*-
"""
核心检测算法 — 供 benchmark 脚本和 tncode_solver.py 共用。
包含 NPC Baseline、v4f 和 v5 的纯函数实现。
"""
import cv2
import numpy as np


# 常量
CANNY_THRESHOLDS = [
    (30, 100), (50, 150), (80, 180), (100, 200),
    (120, 240), (150, 250), (180, 300),
]
NPC_LOW, NPC_HIGH = 150, 250
CONFIDENCE_THRESHOLD = 0.35
AGREEMENT_TOLERANCE = 5
SSD_CONFIDENCE_THRESHOLD = 0.80


def _preprocess(background, puzzle):
    """归一化 + 灰度转换。"""
    bg = cv2.normalize(background, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pz = cv2.normalize(puzzle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY)


def _multi_canny_match(bg_gray, pz_gray):
    """多阈值 Canny 模板匹配，返回 (x, y, confidence)。"""
    best_val, best_loc = -1.0, (0, 0)
    for lo, hi in CANNY_THRESHOLDS:
        bg_e = cv2.Canny(bg_gray, lo, hi)
        pz_e = cv2.Canny(pz_gray, lo, hi)
        result = cv2.matchTemplate(bg_e, pz_e, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_val:
            best_val, best_loc = max_val, max_loc
    return best_loc[0], best_loc[1], max(0.0, min(1.0, best_val))


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
