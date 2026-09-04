# -*- coding: utf-8 -*-
"""
vue-puzzle-vcode 风格的拼图验证码批量生成器。

复刻 vue-puzzle-vcode (https://github.com/javaLuo/vue-puzzle-vcode) 的拼图形状算法，
支持从任意图片目录批量生成测试数据集。

生成的格式与 No-Puzzle-Captcha 测试集兼容 (dataset.json + 背景图 + 拼图切片图)。
"""
import json, os, math, random, argparse, glob
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ============================================================================
#  拼图形状算法 — 复刻 vue-puzzle-vcode 的 paintBrick
# ============================================================================

def make_jigsaw_mask(puzzle_size, scale=1.0):
    """
    生成 vue-puzzle-vcode 风格的 jigsaw 拼图形状 mask。

    源自 paintBrick() 方法:
    - 3x3 的 moveL 网格 (moveL = ceil(15 * scale))
    - 顶部有一个凸起的圆形 knob (arcTo)
    - 右侧有一个凸起的圆形 knob (arcTo)
    - 左侧有一个凹陷的 indent (arcTo)
    - 底部是直线

    Args:
        puzzle_size: 拼图块的基础尺寸 (对应 puzzleBaseSize)
        scale: 缩放因子 (对应 puzzleScale)

    Returns:
        mask: 二值 mask (uint8, 0/255)
        moveL: 基础移动距离
    """
    moveL = math.ceil(15 * scale)
    piece_size = moveL * 3  # 拼图块总尺寸 = 3 * moveL

    # 计算 knob 半径
    knob_r = moveL // 2

    # mask 需要额外空间容纳 knob (顶部、右侧、左侧各扩展 knob_r)
    padding = knob_r + 5
    mask_h = piece_size + 2 * padding
    mask_w = piece_size + 2 * padding

    # 原点偏移
    ox, oy = padding, padding

    # 生成多边形点集 — 复刻 paintBrick 的绘制顺序
    points = []

    # 起点 (ox, oy)
    points.append((ox, oy))

    # === 上边缘 ===
    # 第1段: 直线向右 moveL
    points.append((ox + moveL, oy))

    # 顶部 knob: 两个 arcTo 形成半圆凸起
    # arcTo(cp1x, cp1y, cp2x, cp2y, r) 形成一个向上凸起的半圆
    # 复刻: arcTo(pinX+moveL, pinY-moveL/2, pinX+moveL+moveL/2, pinY-moveL/2, moveL/2)
    # 这是一个向上的半圆, 圆心在 (ox+moveL+moveL/2, oy-moveL/2), 半径=moveL/2
    n_arc = 20
    cx1 = ox + moveL + moveL // 2
    cy1 = oy - moveL // 2
    r1 = moveL // 2
    # 半圆: 从270° (底部) 到90° (顶部) 再到底部, 实际是从右到左的上半圆
    for i in range(n_arc + 1):
        angle = math.pi * 1.5 + math.pi * i / n_arc  # 270° → 450° (=90°), 即从底到顶到底
        px = cx1 + r1 * math.cos(angle)
        py = cy1 + r1 * math.sin(angle)
        points.append((int(round(px)), int(round(py))))

    # 第3段: 直线向右 moveL (到达右上角)
    points.append((ox + 3 * moveL, oy))

    # === 右边缘 ===
    # 第1段: 直线向下 moveL
    points.append((ox + 3 * moveL, oy + moveL))

    # 右侧 knob: 两个 arcTo 形成向右凸起的半圆
    cx2 = ox + 3 * moveL + moveL // 2
    cy2 = oy + moveL + moveL // 2
    r2 = moveL // 2
    for i in range(n_arc + 1):
        angle = math.pi + math.pi * i / n_arc  # 180° → 360°, 从左到右的半圆
        px = cx2 + r2 * math.cos(angle)
        py = cy2 + r2 * math.sin(angle)
        points.append((int(round(px)), int(round(py))))

    # 第3段: 直线向下 moveL (到达右下角)
    points.append((ox + 3 * moveL, oy + 3 * moveL))

    # === 下边缘 ===
    # 直线向左 3*moveL (到达左下角)
    points.append((ox, oy + 3 * moveL))

    # === 左边缘 ===
    # 第1段: 直线向上 moveL
    points.append((ox, oy + 2 * moveL))

    # 左侧 indent: 两个 arcTo 形成向左凹陷的半圆 (方向与 knob 相反)
    cx3 = ox + moveL // 2
    cy3 = oy + moveL + moveL // 2
    r3 = moveL // 2
    for i in range(n_arc + 1):
        angle = 0 + math.pi * i / n_arc  # 0° → 180°, 从右到左的半圆 (凹陷方向)
        px = cx3 + r3 * math.cos(angle)
        py = cy3 + r3 * math.sin(angle)
        points.append((int(round(px)), int(round(py))))

    # 第3段: 直线向上 moveL (回到起点)
    points.append((ox, oy + moveL))

    # 绘制填充多边形
    pts = np.array(points, dtype=np.int32)
    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    return mask, moveL, (ox, oy)


# ============================================================================
#  图片预处理
# ============================================================================

def cover_resize(img, target_w, target_h):
    """
    以 cover 模式缩放图片到目标尺寸 (与 vue-puzzle-vcode 的 makeImgSize 一致)。
    """
    h, w = img.shape[:2]
    img_scale = w / h
    target_scale = target_w / target_h

    if img_scale > target_scale:
        # 图片更宽: 高度填满, 宽度裁剪
        new_h = target_h
        new_w = int(img_scale * new_h)
    else:
        # 图片更高: 宽度填满, 高度裁剪
        new_w = target_w
        new_h = int(new_w / img_scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 居中裁剪
    y_off = (new_h - target_h) // 2
    x_off = (new_w - target_w) // 2
    cropped = resized[y_off:y_off + target_h, x_off:x_off + target_w]

    return cropped


# ============================================================================
#  生成一个测试用例
# ============================================================================

def generate_vcode_case(img_path, canvas_w=310, canvas_h=160, puzzle_scale=1.0):
    """
    生成一个 vue-puzzle-vcode 风格的拼图验证码测试用例。

    Args:
        img_path: 背景图片路径
        canvas_w: canvas 宽度 (默认 310, 与 vue-puzzle-vcode 一致)
        canvas_h: canvas 高度 (默认 160, 与 vue-puzzle-vcode 一致)
        puzzle_scale: 拼图块缩放 (默认 1.0)

    Returns:
        (bg_with_gap, puzzle_piece_bgra, [pinX, pinY]) or None
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    # cover resize 到 canvas 尺寸
    bg = cover_resize(img, canvas_w, canvas_h)

    # 生成 jigsaw mask
    puzzle_base_size = round(max(min(puzzle_scale, 2), 0.2) * 52.5 + 6)
    mask, moveL, (ox, oy) = make_jigsaw_mask(puzzle_base_size, puzzle_scale)
    mask_h, mask_w = mask.shape

    # 随机位置 (与 vue-puzzle-vcode 一致的边界)
    # pinX: puzzleBaseSize 到 canvasWidth - puzzleBaseSize - 20
    min_x = puzzle_base_size
    max_x = canvas_w - puzzle_base_size - 20
    if max_x <= min_x:
        return None
    pinX = random.randint(min_x, max_x)

    # pinY: 20 到 canvasHeight - puzzleBaseSize - 20
    min_y = 20
    max_y = canvas_h - puzzle_base_size - 20
    if max_y <= min_y:
        return None
    pinY = random.randint(min_y, max_y)

    # 检查 mask 是否超出背景边界
    mask_x0 = pinX - ox
    mask_y0 = pinY - oy
    if mask_x0 < 0 or mask_y0 < 0:
        return None
    if mask_x0 + mask_w > canvas_w or mask_y0 + mask_h > canvas_h:
        return None

    # 提取拼图切片 — 保存为 BGR (3通道), 非拼图区域填黑色
    # 这样 cv2.imread 可以正确读取, 且模板匹配可以利用黑色边界
    piece_bgr = np.zeros((mask_h, mask_w, 3), np.uint8)
    roi = bg[mask_y0:mask_y0 + mask_h, mask_x0:mask_x0 + mask_w]
    for c in range(3):
        piece_bgr[:, :, c] = cv2.bitwise_and(roi[:, :, c], mask)

    # 创建带缺口的背景 (暗化缺口区域，保留原始内容的边缘结构)
    bg_gap = bg.copy()
    roi_gap = bg_gap[mask_y0:mask_y0 + mask_h, mask_x0:mask_x0 + mask_w]
    darkened = np.clip(roi_gap.astype(float) * 0.3, 0, 255).astype(np.uint8)
    bg_gap[mask_y0:mask_y0 + mask_h, mask_x0:mask_x0 + mask_w] = np.where(
        mask[:, :, None] > 0,
        darkened,
        roi_gap
    )

    return bg_gap, piece_bgr, [pinX, pinY], [mask_x0, mask_y0], mask


# ============================================================================
#  批量生成数据集
# ============================================================================

def find_images(directory, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    """递归查找目录下所有图片文件。"""
    images = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(extensions):
                images.append(os.path.join(root, f))
    return images


def generate_dataset(image_dir, output_dir, name, n_cases,
                     canvas_w=310, canvas_h=160, puzzle_scale=1.0,
                     image_format="png", seed=None):
    """
    从图片目录批量生成 vue-puzzle-vcode 风格的拼图验证码数据集。

    Args:
        image_dir: 输入图片目录
        output_dir: 输出目录
        name: 数据集名称
        n_cases: 要生成的样例数
        canvas_w, canvas_h: canvas 尺寸
        puzzle_scale: 拼图块缩放
        image_format: 输出图片格式 ("png" or "jpg")
        seed: 随机种子
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # 查找图片
    images = find_images(image_dir)
    if not images:
        print(f"  ERROR: No images found in {image_dir}")
        return None

    print(f"  Found {len(images)} images in {image_dir}")

    cases = []
    generated = 0
    max_attempts = n_cases * 3

    pbar = tqdm(total=n_cases, desc=f"  Generating", ncols=80,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for attempt in range(max_attempts):
        if generated >= n_cases:
            break

        img_path = random.choice(images)

        result = generate_vcode_case(img_path, canvas_w, canvas_h, puzzle_scale)
        if result is None:
            continue

        bg_img, pz_img, pinXY, gap_pos, mask = result

        bg_name = f"IMG_{generated:04d}_O.{image_format}"
        pz_name = f"IMG_{generated:04d}_P.{image_format}"

        cv2.imwrite(os.path.join(output_dir, bg_name), bg_img)
        cv2.imwrite(os.path.join(output_dir, pz_name), pz_img)

        cases.append({
            "background": bg_name,
            "puzzle": pz_name,
            "position": gap_pos,
            "pinXY": pinXY,
            "source": os.path.basename(img_path),
        })
        generated += 1
        pbar.update(1)

    pbar.close()

    dataset = {
        "name": name,
        "error_tolerance": 10,  # vue-puzzle-vcode 默认 range=10
        "canvas": {"width": canvas_w, "height": canvas_h},
        "puzzle_scale": puzzle_scale,
        "source": "vue-puzzle-vcode style generator",
        "revision_date": "2026-09-04",
        "cases": cases,
    }

    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"  Generated {generated}/{n_cases} cases → {output_dir}")
    return dataset


# ============================================================================
#  命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="vue-puzzle-vcode style puzzle captcha dataset generator"
    )
    parser.add_argument("--images", "-i", required=True,
                        help="Input image directory (e.g. Caltech-256 path)")
    parser.add_argument("--output", "-o", default="tests/vcode_real",
                        help="Output directory (default: tests/vcode_real)")
    parser.add_argument("--name", "-n", default="VuePuzzleVCode Real Images",
                        help="Dataset name")
    parser.add_argument("--count", "-c", type=int, default=1000,
                        help="Number of cases to generate (default: 1000)")
    parser.add_argument("--width", type=int, default=310,
                        help="Canvas width (default: 310)")
    parser.add_argument("--height", type=int, default=160,
                        help="Canvas height (default: 160)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Puzzle scale (default: 1.0)")
    parser.add_argument("--format", choices=["png", "jpg"], default="png",
                        help="Output image format (default: png)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    generate_dataset(
        image_dir=args.images,
        output_dir=args.output,
        name=args.name,
        n_cases=args.count,
        canvas_w=args.width,
        canvas_h=args.height,
        puzzle_scale=args.scale,
        image_format=args.format,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
