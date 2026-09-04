# -*- coding: utf-8 -*-
"""
NPC (No-Puzzle-Captcha) baseline 算法。
作为基准对比使用的独立实现。

算法: Normalize → Canny(150,250) → matchTemplate(TM_CCOEFF_NORMED)
来源: https://github.com/isHarryh/No-Puzzle-Captcha
"""
import cv2
import numpy as np


def detect(background, puzzle):
    """
    NPC baseline 检测。
    Args:
        background: BGR 背景图 (numpy array)
        puzzle: BGR 拼图切片 (numpy array)
    Returns:
        (x, y): 检测到的拼图缺口位置
    """
    bg = cv2.normalize(background, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pz = cv2.normalize(puzzle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    pz_gray = cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY)
    bg_edge = cv2.Canny(bg_gray, 150, 250)
    pz_edge = cv2.Canny(pz_gray, 150, 250)
    result = cv2.matchTemplate(bg_edge, pz_edge, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    return max_loc[0], max_loc[1]
