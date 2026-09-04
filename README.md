# TncodeSolver

通用 tncode 滑动验证码求解器。v5 算法在 39890 样例上达到 **95.8%** 准确率，配套反检测/人机行为模拟。

## 目录结构

```
tncode_solver.py              核心求解器 (v5 算法 + 反检测)
demo.py                       使用示例
vcode_dataset_generator.py    数据集生成器 (复刻 vue-puzzle-vcode)
test_report.md                完整测试报告
README.md                     本文件

benchmark/
  algorithms.py               共享算法模块 (NPC + v5 纯函数)
  npc_baseline.py             NPC Baseline (转发到 algorithms.py)
  run_test.py                 统一测试脚本
  experiment.py               A/B 实验对比
  test_v5.py                  v4f vs v5 对比
  extract_balanced.py         构建平衡测试集
  analyze_failures.py         失败案例分析

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

## 算法

### CV 检测 (v5)

```
多阈值 Canny (7 组) → NPC 一致性校验 → 置信度择优
```

1. 对 7 组 Canny 阈值分别做 `matchTemplate(CCOEFF_NORMED)`，取最高置信度 → MC
2. 用 NPC 阈值 (150,250) 单独匹配 → NPC
3. MC conf >= 0.35 或两者 x 差距 <= 5px → 采纳 MC
4. 否则取 MC 和 NPC 中置信度更高者

### 历代算法

| 版本 | 策略 | 总体准确率 |
|------|------|-----------|
| NPC Baseline | Canny(150,250) + matchTemplate | 74.4% |
| v4f | MC + NPC + SSD 兜底 | 96.5% |
| **v5** | **MC + NPC 置信度择优** | **95.8%** |

v5 在合成数据上 99.7%（v4f 95.5%），平衡数据集上 68.2%（v4f 49.9%）。

### 反检测 / 人机行为模拟

- 随机鼠标游走 (1-3 次)
- 渐进式滑块 (微抖动 + 15% 随机暂停)
- 自然加减速 + 过冲回弹 + y 轴偏移
- CDP 原生鼠标事件
- 卡住检测: 连续 2 次相同哈希 → 刷新，3 次 → 关闭重开

## 准确率

### 按数据集 (NPC / v4f / v5)

| 数据集 | 样例 | NPC | v4f | v5 | v4f→v5 |
|--------|------|-----|-----|-----|--------|
| GeeTest | 115 | 90.4% | 91.3% | 91.3% | 0 |
| Tricky | 100 | 99.0% | 99.0% | 99.0% | 0 |
| Tricky Hard | 190 | 90.5% | 98.9% | 98.4% | -0.5 |
| Syn Easy | 200 | 96.5% | 98.0% | 100.0% | +2.0 |
| Syn Medium | 200 | 75.5% | 97.0% | 99.5% | +2.5 |
| Syn Hard | 200 | 45.5% | 87.0% | 100.0% | +13.0 |
| Slider Easy | 200 | 91.5% | 96.5% | 99.0% | +2.5 |
| Slider Hard | 200 | 42.0% | 99.0% | 100.0% | +1.0 |
| Caltech-256 (30K) | 30607 | 76.9% | 96.1% | 96.1% | 0 |
| Caltech-256 (5K) | 5000 | 77.6% | 96.5% | 98.0% | +1.5 |
| Balanced 50/50 | 2878 | 41.0% | 49.9% | 68.2% | +18.3 |
| **总计** | **39890** | **74.4%** | **96.5%** | **95.8%** | — |

### 按类别

| 类别 | 样例 | NPC | v4f | v5 |
|------|------|-----|-----|-----|
| 原始 | 405 | 92.6% | 96.8% | **97.5%** |
| 合成 | 1000 | 70.2% | 95.5% | **99.7%** |
| Caltech-256 | 35607 | 77.0% | 96.2% | **96.3%** |
| Balanced | 2878 | 41.0% | 49.9% | **68.2%** |

## 使用

```python
from tncode_solver import TncodeSolver

solver = TncodeSolver(page, data_file="data.json")
if solver.solve():
    print("验证码通过")
```

```bash
# 测试
python benchmark/run_test.py                                    # 全部
python benchmark/run_test.py --dataset geetest_test             # 指定
python benchmark/run_test.py --dataset vcode_caltech_30k -m 500 # 快速验证

# 实验对比
python benchmark/experiment.py --max-cases 200
python benchmark/test_v5.py

# 生成数据集
python vcode_dataset_generator.py -i <图片目录> -o tests/my_ds -c 5000
```

## 依赖

```
drissionpage opencv-python numpy tqdm
```
