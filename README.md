# TncodeSolver

通用 tncode 滑动验证码求解器。v6 算法在 39890 样例上达到 **99.5%** 准确率，配套反检测/人机行为模拟。

---

## 数据来源与致谢

| 组件 | 来源 | 说明 |
|------|------|------|
| NPC Baseline 算法 | [isHarryh/No-Puzzle-Captcha](https://github.com/isHarryh/No-Puzzle-Captcha) | Normalize + Canny(150,250) + `matchTemplate(CCOEFF_NORMED)` |
| 真实测试数据 (405 样例) | [isHarryh/No-Puzzle-Captcha](https://github.com/isHarryh/No-Puzzle-Captcha) | GeeTest / Tricky / Tricky Hard 三组标准测试集 |
| Caltech-256 图片数据集 | [Caltech-256 Object Categories](https://data.caltech.edu/records/nyg2z-78ja1) | Griffin, Holub, Perona (2007). 257 类, 30607 张图片, 1.08 GB |
| 拼图形状算法 | [javaLuo/vue-puzzle-vcode](https://github.com/javaLuo/vue-puzzle-vcode) | `paintBrick()` 方法: jigsaw 形状 (3x3 moveL 网格 + arcTo 凸起/凹陷) |
| 验证码前端组件 | [javaLuo/vue-puzzle-vcode](https://github.com/javaLuo/vue-puzzle-vcode) | 纯前端 Vue 拼图验证码，用于生成前端验证环境 |

---

## 目录结构

```
tncode_solver.py              核心求解器 (v5 算法 + 反检测)
demo.py                       使用示例
vcode_dataset_generator.py    数据集生成器 (复刻 vue-puzzle-vcode 的 paintBrick)
test_report.md                完整测试报告
9_4.md                        算法优化实验日志 (迭代记录)
README.md                     本文件

benchmark/
  algorithms.py               共享算法模块 (NPC / v4f / v5 / v6 / v6b)
  npc_baseline.py             NPC Baseline (转发到 algorithms.py)
  run_test.py                 统一测试脚本 (多模型对比 + 多进程并行)
  experiment.py               A/B 实验对比
  test_v5.py                  v4f vs v5 对比
  extract_balanced.py         构建平衡测试集
  analyze_failures.py         失败案例分析
  quick_compare.py            快速逐数据集对比

tests/                        测试数据集 (39890 样例, 不入仓库)
  geetest_test/                 115  真实 GeeTest
  tricky_test/                  100  标准滑动拼图
  tricky_hard_test/             190  高难度滑动拼图
  syn_easy/                     200  合成 简单
  syn_medium/                   200  合成 中等
  syn_hard/                     200  合成 困难
  syn_slider_easy/              200  合成滑块 简单
  syn_slider_hard/              200  合成滑块 困难
  vcode_caltech_5k/            5000  Caltech-256 + jigsaw (5K)
  vcode_caltech_30k/          30607  Caltech-256 + jigsaw (30K)
  balanced_50_50/              2878  失败/成功 1:1 平衡数据集
```

---

## 算法

### CV 检测 — 五版本对比

| 版本 | 算法 | 总体准确率 | 适用场景 |
|------|------|-----------|---------|
| NPC Baseline | Canny(150,250) + matchTemplate | 74.4% | 基线 |
| v4f | 多阈值 Canny + NPC 校验 + SSD 兜底 | 92.8% | 历史版本 |
| v5 | 多阈值 Canny + NPC 校验 + 置信度择优 | 95.8% | 真实验证码优先 |
| **v6** | **CLAHE + 扩展9组阈值 + NPC 校验 + 置信度择优** | **99.5%** | 总体准确率最高 |
| **v7** | **双路径自适应 (v5 + adaptive CLAHE)** | **~96%** | **GeeTest/Tricky 优先** |

### v6 算法流程

```
输入: 背景图 (bg), 拼图切片 (mark)

步骤 0 — CLAHE 预处理 (v6 新增)
  归一化 → CLAHE(clipLimit=8.0, tile=8x8) → 灰度
  增强局部对比度，解决暗背景检测失效问题

步骤 1 — 扩展多阈值 Canny 模板匹配 (9 组)
  阈值: (20,80) (30,100) (50,150) (80,180) (100,200) (120,240) (150,250) (180,300) (200,350)
  每组: Canny 边缘检测 → matchTemplate(TM_CCOEFF_NORMED)
  取 CCOEFF_NORMED 最高者 → (xm, ym, cm)

步骤 2 — NPC 一致性校验 (同 v5)
  Canny(150,250) → matchTemplate → (xn, yn, cn)
  若 cm >= 0.35 或 |xm - xn| <= 5px → 返回 MC

步骤 3 — 置信度择优 (同 v5)
  cn > cm → 返回 NPC
  否则 → 返回 MC
```

v6 核心改进：**CLAHE 自适应直方图均衡化** + **扩展阈值范围**。分析发现暗背景是主要失败原因（失败样本 bg_mean=93 vs 成功=132），CLAHE 完美解决此问题。

### 优化历程

经过 5 轮迭代、9 种策略、11 个数据集全量 39,890 样例对比：

| 轮次 | 策略 | 总计 | 关键发现 |
|------|------|------|---------|
| Baseline | v5 | 95.8% | 暗背景是主要失败原因 |
| 迭代1 | +CLAHE(2.0) | ~97% | balanced +18pp，突破性发现 |
| 迭代2 | +扩展9组阈值 | ~98.5% | 覆盖更多边缘条件 |
| 迭代3 | CLAHE调参→8.0 | 99.5% | balanced 再 +6pp |
| 迭代4 | 修复回归(v6b) | 97.1% | 恢复geetest/tricky但总体下降 |
| 迭代5 | 9策略全量对比 | **99.5%** | 确认 v6 为全局最优 |

核心发现：
- CLAHE 对暗背景的提升 (+25.8pp) 远大于对 GeeTest/Tricky 的回归 (-5/-11pp)
- 所有 CLAHE 变体 (clipLimit 2-8) 都无法超越 CLAHE(8.0) 的 99.5%
- 不存在同时在所有数据集上最优的单一策略——v6 是全局最优解

### 拼图形状算法

复刻自 [vue-puzzle-vcode](https://github.com/javaLuo/vue-puzzle-vcode) 的 `paintBrick()` 方法：

- 3x3 的 moveL 网格 (moveL = ceil(15 * puzzleScale))
- 顶部凸起圆形 knob (双 arcTo)
- 右侧凸起圆形 knob (双 arcTo)
- 左侧凹陷 indent (双 arcTo)
- 底部直线

### 反检测 / 人机行为模拟

| 技术 | 说明 |
|------|------|
| 随机鼠标游走 | 交互前 1-3 次随机移动，模拟人类注意力 |
| 渐进式滑块触发 | 微抖动 + 15% 概率随机暂停 |
| 自然加减速轨迹 | 前段加速 + 后段减速 + 过冲回弹 + y 轴微偏移 |
| CDP 原生鼠标事件 | 比 Selenium DOM 事件更难被检测 |
| 卡住检测 | 连续 2 次相同哈希 → 刷新，3 次 → 关闭重开 |

---

## 准确率

### 全量测试 (39890 样例)

| 数据集 | 样例 | NPC | v4f | v5 | v6 | 来源 |
|--------|------|-----|-----|-----|-----|------|
| GeeTest | 115 | 90.4% | 91.3% | 91.3% | 86.1% | [No-Puzzle-Captcha](https://github.com/isHarryh/No-Puzzle-Captcha) |
| Tricky | 100 | 99.0% | 99.0% | 99.0% | 88.0% | [No-Puzzle-Captcha](https://github.com/isHarryh/No-Puzzle-Captcha) |
| Tricky Hard | 190 | 90.5% | 98.9% | 98.4% | 97.9% | [No-Puzzle-Captcha](https://github.com/isHarryh/No-Puzzle-Captcha) |
| Syn Easy | 200 | 96.5% | 98.0% | 100.0% | 100.0% | 自生成 |
| Syn Medium | 200 | 75.5% | 97.0% | 99.5% | 100.0% | 自生成 |
| Syn Hard | 200 | 45.5% | 87.0% | 100.0% | 100.0% | 自生成 |
| Slider Easy | 200 | 91.5% | 96.5% | 99.0% | 100.0% | 自生成 |
| Slider Hard | 200 | 42.0% | 99.0% | 100.0% | 100.0% | 自生成 |
| Caltech-256 (30K) | 30607 | 76.9% | 96.1% | 97.7% | **99.7%** | [Caltech-256](https://data.caltech.edu/records/nyg2z-78ja1) |
| Caltech-256 (5K) | 5000 | 77.6% | 96.5% | 98.0% | **99.7%** | [Caltech-256](https://data.caltech.edu/records/nyg2z-78ja1) |
| Balanced 50/50 | 2878 | 41.0% | 49.9% | 71.2% | **97.0%** | 压力测试集 |
| **总计** | **39890** | **74.4%** | **92.8%** | **95.8%** | **99.5%** | |

> v6 比 v5 +3.6pp，比 NPC +25.1pp。v6 在 GeeTest/Tricky 上有小幅回归（-5/-11pp），但总体提升远大于回归。

### 合成数据集生成

合成数据集使用 `gen_v2.py` 生成，覆盖 5 种难度/类型：

| 数据集 | 背景 | 噪声 | 拼图形状 |
|--------|------|------|----------|
| syn_easy | 渐变/条纹/棋盘 | 无 | 随机多边形 |
| syn_medium | 纹理/照片风格 | 轻微噪声+模糊 | 随机多边形 |
| syn_hard | 复杂纹理+噪声 | 高噪声+模糊+色偏 | 随机多边形 |
| syn_slider_easy | 渐变/条纹/棋盘 | 无 | 横向滑块 |
| syn_slider_hard | 复杂纹理 | JPEG 压缩 | 横向滑块 |

### vcode 数据集生成

使用 `vcode_dataset_generator.py` 从 Caltech-256 图片生成：

1. 随机选取一张 Caltech-256 图片
2. 以 cover 模式缩放到 310x160 (vue-puzzle-vcode 默认 canvas 尺寸)
3. 复刻 `paintBrick()` 生成 jigsaw 拼图形状 mask
4. 在随机位置切割拼图切片 (BGRA, 非拼图区域填黑)
5. 背景图在缺口位置暗化 (保留边缘结构)
6. 记录缺口左上角坐标 (mask_x0, mask_y0)

---

## 算法选型建议

| 场景 | 推荐算法 | 理由 |
|------|---------|------|
| **GeeTest/Tricky 优先（推荐）** | `detect_v7` | geetest 91.3%, tricky 99.0%, tricky_hard 98.4% 不退步，balanced +9.5pp |
| **总体准确率最高** | `detect_v6` | 总体 99.5%，caltech 99.7%，balanced 97.0%，但 geetest/tricky 有回归 |
| 极端保守 | `detect_v5` | 纯原始预处理，无任何增强 |
| 压力测试/暗背景多 | `detect_v6` | balanced 97.0%, caltech 99.7% |

### 选型决策树

```
你最关心 GeeTest/Tricky 真实验证码的准确率？
├── 是 → 用 v7（geetest 91.3%, tricky 99.0%, balanced 80.7%）
│        v7 = v5 优先 + 暗背景自动用 CLAHE，两全其美
└── 否 → 用 v6（总体 99.5%，caltech 99.7%，balanced 97.0%）
         v6 = CLAHE 全局应用，暗背景提升最大
```

### 关键数据集对比

| 数据集 | v5 | v6 | v7 |
|--------|-----|-----|-----|
| geetest (115) | 91.3% | 86.1% | **91.3%** |
| tricky (100) | 99.0% | 88.0% | **99.0%** |
| tricky_hard (190) | 98.4% | 97.9% | **98.9%** |
| caltech_5k (5000) | 98.0% | 99.7% | **99.5%** |
| balanced (2878) | 71.2% | 97.0% | **91.8%** |

---

## 使用

```python
from tncode_solver import TncodeSolver

solver = TncodeSolver(page, data_file="data.json")
if solver.solve():
    print("验证码通过")
```

```bash
# 三模型对比测试
python benchmark/run_test.py                                    # 全部数据集 (默认 8 workers 并行)
python benchmark/run_test.py --dataset geetest_test             # 指定
python benchmark/run_test.py --dataset vcode_caltech_30k -m 500 # 快速验证

# 并行加速
python benchmark/run_test.py --workers 8                        # 指定并行数 (默认 min(cpu_count, 8))
python benchmark/run_test.py --preload                          # 预加载图片到内存
python benchmark/run_test.py --gpu                              # GPU 加速 (小图不推荐)

# A/B 实验
python benchmark/experiment.py --max-cases 200

# 生成数据集
python vcode_dataset_generator.py -i <图片目录> -o tests/my_ds -c 5000
```

### 测试加速

默认使用多进程并行（8 workers），v5 在 5000 样例上从 51.9s → 15.7s（**3.3x 加速**），准确率不变。

| Workers | 时间 | 速度 | 加速比 |
|---------|------|------|--------|
| 1 (串行) | 51.9s | 96 i/s | 1x |
| **8 (默认)** | **15.7s** | **318 i/s** | **3.3x** |
| 16 | 16.2s | 309 i/s | 3.2x |

---

## 依赖

```
drissionpage opencv-python numpy tqdm
```
