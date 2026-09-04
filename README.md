# TncodeSolver

通用 tncode 滑动验证码求解器。优化 CV 检测算法 (v5) 在 39890 样例上达到 **95.8%** 准确率，配套完整反检测/人机行为模拟。

## 目录结构

```
tncode_solver.py              核心求解器 (v5 算法 + 反检测)
demo.py                       使用示例
vcode_dataset_generator.py    数据集生成器
test_report.md                测试报告
README.md                     本文件

benchmark/
  npc_baseline.py             NPC Baseline (对比基准)
  run_test.py                 统一测试脚本

tests/                        测试数据集 (39890 样例)
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
  balanced_50_50/              2878  v4f 失败/成功 1:1 平衡数据集
```

## 算法

### CV 检测 (v5)

```
多阈值 Canny 模板匹配 → NPC 一致性校验 → 置信度择优
```

- **多阈值 Canny**: 7 组阈值 (30/100 ~ 180/300)，取 `TM_CCOEFF_NORMED` 最高者
- **NPC 校验**: 置信度 >= 0.35 或与 NPC 结果差距 <= 5px 时采纳 MC
- **置信度择优**: 低置信度时取 MC 和 NPC 中置信度更高者

### 历代算法

| 版本 | 算法 |
|------|------|
| NPC Baseline | Normalize → Canny(150,250) → `matchTemplate(CCOEFF_NORMED)` |
| v4f | 多阈值 Canny + NPC 一致性校验 + SSD 像素匹配兜底 |
| v5 | 多阈值 Canny + NPC 一致性校验 + 置信度择优（去掉 SSD） |

### 反检测 / 人机行为模拟

- 随机鼠标游走 (交互前 1-3 次)
- 渐进式滑块触发 (含微抖动 + 15% 概率随机暂停)
- 自然加减速轨迹 + 过冲回弹 + y 轴微偏移
- CDP 原生鼠标事件 (比 Selenium DOM 事件更难检测)
- 卡住检测: 连续 2 次相同哈希 → 刷新，3 次 → 关闭重开

## 准确率

### 按数据集

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

### 总计

| 方法 | 准确率 | 正确/总数 | 耗时 | 速度 |
|------|--------|-----------|------|------|
| NPC Baseline | 74.4% | 29670/39890 | 55.0s | 725 i/s |
| v4f (MC+NPC+SSD) | 96.5% | 38465/39890 | 400s | ~100 i/s |
| **v5 (MC+NPC 择优)** | **95.8%** | **38228/39890** | **370.7s** | **108 i/s** |

### 按类别

| 类别 | 样例 | NPC | v4f | v5 |
|------|------|-----|-----|-----|
| 原始数据集 | 405 | 92.6% | 96.8% | **97.5%** |
| 合成数据集 | 1000 | 70.2% | 95.5% | **99.7%** |
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
# 运行测试
python benchmark/run_test.py                                  # 全部数据集
python benchmark/run_test.py --dataset geetest_test           # 指定数据集
python benchmark/run_test.py --dataset vcode_caltech_30k --max-cases 1000  # 限制样例数

# 生成新数据集
python vcode_dataset_generator.py -i <图片目录> -o tests/my_ds -c 5000
```

## 依赖

```
drissionpage opencv-python numpy
```
