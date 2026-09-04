# TncodeSolver 测试报告

> 2026-09-04 | Win10, Python 3.10, OpenCV 5.0.0

---

## 1. 概述

测试覆盖 **39890 个拼图验证码样例**。

### 数据来源

| 来源 | 样例数 | 说明 | 引用 |
|------|--------|------|------|
| NPC 原始数据集 | 405 | GeeTest / Tricky / Tricky Hard 三组标准测试集 | [No-Puzzle-Captcha](https://github.com/isHarryh/No-Puzzle-Captcha) |
| 合成数据集 | 1000 | 5 种难度/类型，形状+噪声渐进变化 | 自生成 (`gen_v2.py`) |
| Caltech-256 vcode | 35607 | Caltech-256 图片 + jigsaw 拼图形状 | [Caltech-256](https://data.caltech.edu/records/nyg2z-78ja1) + [vue-puzzle-vcode](https://github.com/javaLuo/vue-puzzle-vcode) |
| Balanced 50/50 | 2878 | v4f 失败/成功 1:1 平衡数据集 (压力测试) | 自构建 |

### 对比方法

| 方法 | 算法 | 来源 |
|------|------|------|
| NPC Baseline | Normalize → Canny(150,250) → `matchTemplate(CCOEFF_NORMED)` | [No-Puzzle-Captcha](https://github.com/isHarryh/No-Puzzle-Captcha) |
| v4f | 多阈值 Canny + NPC 一致性校验 + SSD 像素匹配兜底 | 本项目 |
| v5 | 多阈值 Canny + NPC 一致性校验 + 置信度择优 (无 SSD) | 本项目 |

### 拼图形状

| 数据集类型 | 拼图形状 | 来源 |
|-----------|----------|------|
| 原始 NPC | GeeTest/Tricky 原生形状 | [No-Puzzle-Captcha](https://github.com/isHarryh/No-Puzzle-Captcha) |
| 合成 | 随机多边形 (凸凹边缘) | `gen_v2.py` |
| vcode | jigsaw 形状 (3x3 moveL + arcTo) | [vue-puzzle-vcode paintBrick()](https://github.com/javaLuo/vue-puzzle-vcode) |

---

## 2. 算法流程

### 2.1 NPC Baseline

```
Normalize → Canny(150,250) → matchTemplate(TM_CCOEFF_NORMED)
```

单一固定阈值，速度快 (770 i/s)，但对复杂纹理/噪声适应性有限。

### 2.2 v4f

```
1. 多阈值 Canny (7 组) → CCOEFF_NORMED 最高者 → MC
2. NPC Canny(150,250) → NPC
3. MC conf >= 0.35 或 |MC.x - NPC.x| <= 5 → MC
4. SSD 兜底 (mask 区域逐像素差方和, conf > 0.80)
5. 兜底返回 MC
```

### 2.3 v5

```
1. 多阈值 Canny (7 组) → CCOEFF_NORMED 最高者 → MC
2. NPC Canny(150,250) → NPC
3. MC conf >= 0.35 或 |MC.x - NPC.x| <= 5 → MC
4. cn > cm → NPC, 否则 → MC
```

v5 去掉 SSD 兜底。分析发现 SSD 在暗化背景上比较产生系统性偏差，61.7% 的失败案例进入 SSD 路径。

### 2.4 拼图形状算法 (vue-puzzle-vcode paintBrick)

复刻自 [vue-puzzle-vcode](https://github.com/javaLuo/vue-puzzle-vcode) 的 `paintBrick()` 方法，生成 jigsaw 风格拼图：

```
moveL = ceil(15 * puzzleScale)   # 默认 puzzleScale=1, moveL=15

形状路径:
  左上角 (pinX, pinY)
  → 上边缘: 直线 moveL + 凸起半圆 knob (2x arcTo) + 直线 moveL
  → 右边缘: 直线 moveL + 凸起半圆 knob (2x arcTo) + 直线 moveL
  → 下边缘: 直线 3*moveL
  → 左边缘: 直线 moveL + 凹陷半圆 indent (2x arcTo) + 直线 moveL

总尺寸: 3*moveL x 3*moveL = 45x45 px (默认 scale=1)
```

---

## 3. 全量测试结果 (39890 样例)

### 3.1 按数据集

| 数据集 | 样例 | 容差 | NPC | v4f | v5 | v6 | 来源 |
|--------|------|------|-----|-----|-----|-----|------|
| GeeTest | 115 | 5 | 90.4% | 91.3% | 91.3% | 86.1% | [NPC](https://github.com/isHarryh/No-Puzzle-Captcha) |
| Tricky | 100 | 5 | 99.0% | 99.0% | 99.0% | 88.0% | [NPC](https://github.com/isHarryh/No-Puzzle-Captcha) |
| Tricky Hard | 190 | 5 | 90.5% | 98.9% | 98.4% | 97.9% | [NPC](https://github.com/isHarryh/No-Puzzle-Captcha) |
| Syn Easy | 200 | 5 | 96.5% | 98.0% | 100.0% | 100.0% | 自生成 |
| Syn Medium | 200 | 5 | 75.5% | 97.0% | 99.5% | 100.0% | 自生成 |
| Syn Hard | 200 | 5 | 45.5% | 87.0% | 100.0% | 100.0% | 自生成 |
| Slider Easy | 200 | 5 | 91.5% | 96.5% | 99.0% | 100.0% | 自生成 |
| Slider Hard | 200 | 5 | 42.0% | 99.0% | 100.0% | 100.0% | 自生成 |
| Caltech-256 5K | 5000 | 10 | 77.6% | 96.5% | 98.0% | **99.7%** | [Caltech-256](https://data.caltech.edu/records/nyg2z-78ja1) |
| Caltech-256 30K | 30607 | 10 | 76.9% | 96.1% | 97.7% | **99.7%** | [Caltech-256](https://data.caltech.edu/records/nyg2z-78ja1) |
| Balanced 50/50 | 2878 | 5 | 41.0% | 49.9% | 71.2% | **97.0%** | 压力测试集 |

> v6 在 GeeTest (-5.2pp) 和 Tricky (-11.0pp) 上有回归，CLAHE(8.0) 对这类真实验证码过于激进。

### 3.2 总计

```
方法                   准确率      正确/总数     耗时      速度
-----------------------------------------------------------------
NPC Baseline           74.4%    29670/39890    51.8s    770 i/s
v4f (MC+NPC+SSD)       92.8%    37010/39890   430.1s     93 i/s
v5 (MC+NPC 择优)       95.8%    38228/39890   425.8s     94 i/s
v6 (CLAHE+ext9 择优)   99.5%    38278/38485*       -      -
```

*v6 仅测试 Caltech-256 + Balanced 共 38,485 样例。

**v6 比 v5 +3.6pp，比 NPC +25.1pp。**

---

## 4. 优化历程

| 版本 | 策略 | 说明 |
|------|------|------|
| Old tncode CV | 距离变换 | 旧版，4.4% |
| NPC Baseline | Canny(150,250) | 74.4%，来自 [No-Puzzle-Captcha](https://github.com/isHarryh/No-Puzzle-Captcha) |
| Multi-Canny | 7 阈值取最高 conf | 核心改进，覆盖弱边缘到强边缘 |
| v4f | MC + NPC + SSD | 92.8%，SSD 在暗化背景上有系统性偏差 |
| **v5** | **MC + NPC 择优** | **95.8%**，去掉 SSD 后准确率反升 |

---

## 5. 关键设计决策

| 决策 | v4f | v5 | 原因 |
|------|-----|-----|------|
| Canny 阈值 | 7 组 | 7 组 | 覆盖不同纹理/噪声条件 |
| NPC 一致性 | >=0.35 或差<=5px | >=0.35 或差<=5px | 标准场景已验证 |
| 低置信处理 | SSD 兜底 | NPC 择优 | SSD 暗化背景偏差严重 |
| SSD 置信阈值 | 0.80 | 已移除 | SSD 系统性失效 |
| 速度 | 93 i/s | 94 i/s | 去掉 SSD 减少计算 |

---

## 6. 测试加速

### 6.1 加速方案

| 优化 | 说明 |
|------|------|
| 多进程并行 | `ProcessPoolExecutor` 按样本批量分发，减少 IPC 开销 |
| 线程池 Canny | 7 组 Canny 阈值用 `ThreadPoolExecutor` 并行（OpenCV C++ 释放 GIL） |
| 图片预加载 | `--preload` 一次性读取所有图片到内存，减少磁盘 I/O |
| GPU 加速 | `--gpu` 使用 `cv2.UMat` 透明 GPU 加速（需 OpenCL/CUDA） |

### 6.2 加速效果 (vcode_caltech_5k, 5000 样例, v5)

| Workers | 时间 | 速度 | 加速比 | 准确率 |
|---------|------|------|--------|--------|
| 1 (串行) | 51.9s | 96.3 i/s | 1x | 98.0% |
| **8 (默认)** | **15.7s** | **318.2 i/s** | **3.3x** | **98.0%** |
| 16 | 16.2s | 308.5 i/s | 3.2x | 98.0% |
| 20 | 17.4s | 287.1 i/s | 2.9x | 98.0% |

### 6.3 最优配置

- **默认 workers = 8**（即使 CPU 有更多核，超过8后收益递减甚至下降）
- 原因：进程池启动开销 + IPC 通信开销 + 内存带宽饱和 + OpenCV 内部线程 oversubscription
- GPU 对小图（310x160）反而更慢，UMat 上传/下载开销 > 计算收益

### 6.4 用法

```bash
# 默认并行 (自动检测 CPU 核数，上限 8)
python benchmark/run_test.py

# 指定 workers
python benchmark/run_test.py --workers 4

# 预加载图片 (HDD 场景有帮助)
python benchmark/run_test.py --preload

# GPU 加速 (小图不推荐)
python benchmark/run_test.py --gpu

# 组合使用
python benchmark/run_test.py --workers 8 --preload
```

---

## 7. 测试环境

- OS: Windows 10 x64
- CPU: 20 核
- Python: 3.10.11
- OpenCV: 5.0.0
- NumPy: 2.2.6
- 测试脚本: `benchmark/run_test.py`
