# TncodeSolver

通用 tncode 滑动验证码求解器。v5 算法在 39890 样例上达到 **95.8%** 准确率，配套反检测/人机行为模拟。

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
README.md                     本文件

benchmark/
  algorithms.py               共享算法模块 (NPC / v4f / v5 纯函数)
  npc_baseline.py             NPC Baseline (转发到 algorithms.py)
  run_test.py                 统一测试脚本 (三模型对比)
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

### CV 检测 — 三版本对比

| 版本 | 算法 | 总体准确率 |
|------|------|-----------|
| NPC Baseline | Canny(150,250) + matchTemplate | 74.4% |
| v4f | 多阈值 Canny + NPC 校验 + SSD 兜底 | 92.8% |
| **v5** | **多阈值 Canny + NPC 校验 + 置信度择优** | **95.8%** |

### v5 算法流程

```
输入: 背景图 (bg), 拼图切片 (mark)

步骤 1 — 多阈值 Canny 模板匹配 (7 组)
  阈值: (30,100) (50,150) (80,180) (100,200) (120,240) (150,250) (180,300)
  每组: Canny 边缘检测 → matchTemplate(TM_CCOEFF_NORMED)
  取 CCOEFF_NORMED 最高者 → (xm, ym, cm)

步骤 2 — NPC 一致性校验
  Canny(150,250) → matchTemplate → (xn, yn, cn)
  若 cm >= 0.35 或 |xm - xn| <= 5px → 返回 MC

步骤 3 — 置信度择优
  cn > cm → 返回 NPC
  否则 → 返回 MC
```

v5 核心改进：**去掉 SSD 兜底**。分析发现 SSD 在暗化背景上比较产生系统性偏差，61.7% 的失败案例进入 SSD 路径。去掉 SSD 后改用 NPC 置信度择优，准确率从 92.8% 提升到 95.8%。

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

| 数据集 | 样例 | NPC | v4f | v5 | 来源 |
|--------|------|-----|-----|-----|------|
| GeeTest | 115 | 90.4% | 91.3% | 91.3% | [No-Puzzle-Captcha](https://github.com/isHarryh/No-Puzzle-Captcha) |
| Tricky | 100 | 99.0% | 99.0% | 99.0% | [No-Puzzle-Captcha](https://github.com/isHarryh/No-Puzzle-Captcha) |
| Tricky Hard | 190 | 90.5% | 98.9% | 98.4% | [No-Puzzle-Captcha](https://github.com/isHarryh/No-Puzzle-Captcha) |
| Syn Easy | 200 | 96.5% | 98.0% | 100.0% | 自生成 |
| Syn Medium | 200 | 75.5% | 97.0% | 99.5% | 自生成 |
| Syn Hard | 200 | 45.5% | 87.0% | 100.0% | 自生成 |
| Slider Easy | 200 | 91.5% | 96.5% | 99.0% | 自生成 |
| Slider Hard | 200 | 42.0% | 99.0% | 100.0% | 自生成 |
| Caltech-256 (30K) | 30607 | 76.9% | 96.1% | 97.7% | [Caltech-256](https://data.caltech.edu/records/nyg2z-78ja1) |
| Caltech-256 (5K) | 5000 | 77.6% | 96.5% | 98.0% | [Caltech-256](https://data.caltech.edu/records/nyg2z-78ja1) |
| Balanced 50/50 | 2878 | 41.0% | 49.9% | 71.2% | 压力测试集 |
| **总计** | **39890** | **74.4%** | **92.8%** | **95.8%** | |

> v5 比 NPC +21.4pp，比 v4f +3.0pp。

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

## 使用

```python
from tncode_solver import TncodeSolver

solver = TncodeSolver(page, data_file="data.json")
if solver.solve():
    print("验证码通过")
```

```bash
# 三模型对比测试
python benchmark/run_test.py                                    # 全部数据集
python benchmark/run_test.py --dataset geetest_test             # 指定
python benchmark/run_test.py --dataset vcode_caltech_30k -m 500 # 快速验证

# A/B 实验
python benchmark/experiment.py --max-cases 200

# 生成数据集
python vcode_dataset_generator.py -i <图片目录> -o tests/my_ds -c 5000
```

---

## 依赖

```
drissionpage opencv-python numpy tqdm
```
