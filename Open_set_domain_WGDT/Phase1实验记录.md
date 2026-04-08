# Phase 1 实验记录

## 1. 记录说明

本文件用于持续记录 `WGDT + OCCR` 第一阶段实验结果。

当前固定前提：

- 任务：`HSI Open-Set Domain Adaptation`
- 数据集：`PaviaU_7gt -> PaviaC_OS`
- 开发种子：`seed=7`
- 总轮数：`epochs=150`

---

## 2. 已完成实验

| 编号 | 配置名 | 关键参数 | OA | OA_known | Unknown | HOS | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| E1 | Baseline WGDT | `lambda_occr=0, lambda_pl=0` | 0.8123 | 0.6780 | 0.9357 | 0.7770 | 强基线 |
| E2 | 激进 OCCR | `warmup=15, delta_low=0.05, delta_high=0.05, delta_pseudo=0.10, push_margin=0.20, lambda_occr=0.10, lambda_pl=0.0` | 0.4386 | 0.6864 | 0.2109 | 0.3221 | 明显破坏 unknown |
| E3 | 激进 OCCR + PL | `warmup=15, delta_low=0.05, delta_high=0.05, delta_pseudo=0.10, push_margin=0.20, lambda_occr=0.10, lambda_pl=0.10` | 0.4475 | 0.6680 | 0.2450 | 0.3601 | 略优于 E2，但仍远差于 E1 |
| E4 | 保守 OCCR | `warmup=60, delta_low=0.15, delta_high=0.10, delta_pseudo=0.20, push_margin=0.30, lambda_occr=0.02, lambda_pl=0.0` | 0.8552 | 0.8106 | 0.8962 | 0.8390 | 当前最佳 |
| E5 | 保守 OCCR + PL | `warmup=60, delta_low=0.15, delta_high=0.10, delta_pseudo=0.20, push_margin=0.30, lambda_occr=0.02, lambda_pl=0.05` | 0.8370 | 0.8070 | 0.8646 | 0.8225 | 优于 E1，但弱于 E4 |

---

## 3. 当前阶段结论

### 3.1 关于 OCCR

- `OCCR` 本身有潜力，但必须保守启用；
- 激进版本会明显把 unknown 样本拉回 known 区域；
- 保守版本可以同时提升 `OA_known` 与 `HOS`。

### 3.2 关于 late pseudo-label

- 当前在 `Pavia, seed=7` 上没有带来稳定增益；
- 在保守 OCCR 上，加入 pseudo-label 反而略有回落；
- 后续应将其视为可选增强项，而非默认开启模块。

### 3.3 当前推荐默认配置

```text
warmup_epochs = 60
pseudo_last_epochs = 20
delta_low = 0.20
delta_high = 0.10
delta_pseudo = 0.30
push_margin = 0.30
lambda_push = 1.0
lambda_occr = 0.02
lambda_pl = 0.0
```

---

## 4. 下一步实验建议

建议按以下顺序继续：

1. 基于已完成的 `Pavia` 十种子结果，优先总结方法得失；
2. 在保守 OCCR 配置下，优先做稳健性优化；
3. 若稳定性改善后仍优于 baseline，再扩展到 `Houston18_OS`；
4. 若多数据集仍成立，再评估是否重新引入 `late pseudo-label`；
5. 只有在主线稳定后，再考虑 `Mamba / EDL / UOT` 扩展线。

---

## 5. 十种子汇总结果

### 5.1 Baseline 十种子汇总

| seed | oa | oa_known | unknown | hos |
|---|---:|---:|---:|---:|
| 7 | 0.8516 | 0.7790 | 0.9183 | 0.8282 |
| 43 | 0.7409 | 0.6762 | 0.8003 | 0.7284 |
| 45 | 0.7903 | 0.6919 | 0.8808 | 0.7659 |
| 46 | 0.8535 | 0.7516 | 0.9471 | 0.8247 |
| 66 | 0.7112 | 0.6632 | 0.7553 | 0.6979 |
| 67 | 0.8281 | 0.7107 | 0.9360 | 0.8079 |
| 77 | 0.7948 | 0.6420 | 0.9352 | 0.7465 |
| 81 | 0.8462 | 0.7348 | 0.9485 | 0.8119 |
| 88 | 0.7828 | 0.6096 | 0.9420 | 0.7389 |
| 98 | 0.8441 | 0.7643 | 0.9175 | 0.8220 |
| Mean±Std | 0.8043 ± 0.0494 | 0.7023 ± 0.0556 | 0.8981 ± 0.0673 | 0.7772 ± 0.0474 |

### 5.2 保守 OCCR 十种子汇总

| seed | oa | oa_known | unknown | hos |
|---|---:|---:|---:|---:|
| 7 | 0.8846 | 0.8270 | 0.9375 | 0.8751 |
| 43 | 0.7948 | 0.7807 | 0.8077 | 0.7898 |
| 45 | 0.8539 | 0.7667 | 0.9341 | 0.8040 |
| 46 | 0.8513 | 0.8010 | 0.8974 | 0.8306 |
| 66 | 0.7015 | 0.7268 | 0.6783 | 0.6984 |
| 67 | 0.8950 | 0.8309 | 0.9540 | 0.8678 |
| 77 | 0.6279 | 0.7035 | 0.5585 | 0.6109 |
| 81 | 0.8707 | 0.8120 | 0.9247 | 0.8506 |
| 88 | 0.7147 | 0.6871 | 0.7401 | 0.6985 |
| 98 | 0.8894 | 0.8259 | 0.9476 | 0.8609 |
| Mean±Std | 0.8084 ± 0.0946 | 0.7762 ± 0.0535 | 0.8380 ± 0.1373 | 0.7887 ± 0.0899 |

### 5.3 十种子对比结论

- `HOS`：`0.7772 ± 0.0474 -> 0.7887 ± 0.0899`，保守 OCCR 在平均意义上优于 baseline；
- `OA_known`：`0.7023 ± 0.0556 -> 0.7762 ± 0.0535`，已知类识别能力提升非常明显；
- `Unknown`：`0.8981 ± 0.0673 -> 0.8380 ± 0.1373`，未知类拒识能力有所下降；
- `HOS` 标准差增大，说明当前 OCCR 版本稳定性仍偏弱；
- 综合判断：当前保守 OCCR 属于“平均有效，但仍需稳健化”的增强模块。

### 5.4 当前建议结论

- 保留保守 OCCR 作为当前主线方法；
- 在 `Pavia` 十种子上，当前默认配置更新为 `R4`；
- 不将 `late pseudo-label` 作为默认开启项；
- 下一步优先解决跨 seed 波动较大的问题；
- 在稳定性改善前，不建议直接推进更复杂的扩展线。

---

## 6. 结果使用建议

当前这些实验结果适合用于：

- 指导 Phase 1 默认配置修订；
- 支撑“保守 OCCR 在平均意义上优于 baseline，但稳定性仍需优化”的方法学分析；
- 作为后续论文中超参数敏感性、随机种子稳健性与训练稳定性讨论的依据。

---

## 7. 稳健性优化单种子筛选（seed=7）

在完成 `Pavia` 十种子对比后，进一步围绕保守 OCCR 做了 6 组单种子稳健性筛选实验，用于决定下一步进入十种子正式验证的候选配置。

### 7.1 六组筛选结果

| 编号 | 配置摘要 | OA | OA_known | Unknown | HOS | 结论 |
|---|---|---:|---:|---:|---:|---|
| R1 | `warmup=60, delta_low=0.15, delta_pseudo=0.20, push_margin=0.30, lambda_occr=0.02` | 0.8846 | 0.8270 | 0.9375 | 0.8751 | 最优候选 |
| R2 | `warmup=75, 其余同 R1` | 0.8310 | 0.7955 | 0.8636 | 0.8238 | 推迟过多，收益下降 |
| R3 | `delta_low=0.18, delta_pseudo=0.25` | 0.7974 | 0.7587 | 0.8331 | 0.7899 | 过度保守，pull 不足 |
| R4 | `delta_low=0.20, delta_pseudo=0.30` | 0.8526 | 0.8145 | 0.8876 | 0.8419 | 稳健候选 |
| R5 | `lambda_occr=0.01` | 0.7314 | 0.7335 | 0.7295 | 0.7308 | OCCR 过弱，效果差 |
| R6 | `lambda_occr=0.01, push_margin=0.35` | 0.7314 | 0.7335 | 0.7295 | 0.7308 | 与 R5 相同，不入围 |

### 7.2 单种子筛选结论

- `R1` 是当前单种子最优配置，必须进入十种子正式验证；
- `R4` 虽然略弱于 `R1`，但代表“更保守分组”路线，仍有可能在多 seed 下更稳，因此也值得进入十种子正式验证；
- `R2` 说明将 `warmup` 从 `60` 继续推迟到 `75` 并没有带来收益；
- `R3` 说明将 pseudo-known 划分过度收紧会损伤有效的结构 pull；
- `R5/R6` 说明当前 `lambda_occr=0.01` 过弱，且 `push_margin=0.35` 在这一设置下没有产生额外收益。

### 7.3 入围十种子的配置

下一步进入十种子正式验证的两组配置确定为：

- `R1`：当前最优候选；
- `R4`：当前稳健候选。

后续目标为比较这两组在十种子上的：

- `HOS` 平均值；
- `HOS` 标准差；
- `OA_known` 与 `Unknown` 的平衡关系。

### 7.4 R1 vs R4 十种子最终结论

在完成 `R1` 与 `R4` 的十种子正式验证后，当前最终结论如下：

| 配置 | OA | OA_known | Unknown | HOS |
|---|---:|---:|---:|---:|
| R1 | 0.8084 ± 0.0946 | 0.7762 ± 0.0535 | 0.8380 ± 0.1373 | 0.7887 ± 0.0899 |
| R4 | 0.8124 ± 0.0543 | 0.7583 ± 0.0629 | 0.8620 ± 0.0628 | 0.7931 ± 0.0489 |

据此确定：

- `R4` 的 `HOS` 更高；
- `R4` 的 `HOS` 标准差显著更低，稳定性明显优于 `R1`；
- `R4` 的 `Unknown` 更高，更符合开集域适应的目标；
- `R1` 虽然 `OA_known` 更高，但整体平衡与稳健性不如 `R4`。

因此，`R4` 正式确定为当前 Phase 1 的默认主配置。

---

## 8. Houston 单种子开发与诊断记录

在将 `Pavia` 主配置迁移到 `Houston13_7gt -> Houston18_OS` 后，发现当前保守 OCCR 配置在 Houston 上明显不优于 baseline，因此进入 Houston 专项调参与结构诊断阶段。

### 8.1 Houston baseline 与直接迁移结果

已有十种子结果表明：

- `Houston baseline`：`HOS = 0.6115 ± 0.0285`
- `Houston + Pavia-R4 直接迁移`：`HOS = 0.5243 ± 0.0922`

这说明：

- `Pavia` 上的默认配置不能直接迁移到 `Houston`；
- Houston 上必须采用 dataset-specific tuning；
- 问题很可能来自 OCCR 对 unknown 的处理方式，而不仅仅是学习率或 warm-up。

### 8.2 Houston 第一轮单 seed 调参（H1~H4）

围绕“更保守分组、更晚启用、更弱 OCCR”进行了第一轮单种子实验，结果如下：

| 编号 | 方向 | HOS | 结论 |
|---|---|---:|---|
| H1 | 更晚启用 + 更保守筛选 | 0.4960 | 明显差于 baseline |
| H2 | 更弱 OCCR | 0.5602 | 仍低于 baseline |
| H3 | 更严格 pseudo-known | 0.5695 | 相对最好，但仍低于 baseline |
| H4 | 更强 unknown 保护 | 0.4960 | 与 H1 等效 |

结论：

- 单纯继续调 `warmup / delta / lambda_occr` 不足以解决 Houston 问题；
- 当前 OCCR 的主要矛盾不是“训练不稳定”，而是“对 Houston 的 unknown 保护仍然不够有效”。

### 8.3 Houston 第二轮结构微调（H5~H10）

在发现纯调参不足后，引入了结构微调版本：

- `push_only`
- `strict_pull_push`

但 H5~H10 的结果几乎都与 Houston baseline 的 `seed=1` 完全一致，说明：

- 新增的 OCCR 分支虽然被接入训练图，但**实际没有产生有效损失**；
- 训练本质上退化回了 baseline。

### 8.4 诊断日志揭示的根因

通过新增诊断项，最终确认 Houston 上 OCCR 不生效的真实原因如下：

- `num_hard_unknown` 明显大于 0，说明 unknown 分组是成功的；
- 但 `num_push_active = 0`，说明没有任何 hard-unknown 样本真正触发 push；
- 同时 `loss_occr_push_epoch = 0`、`loss_occr_epoch = 0`，说明 OCCR 分支数值上完全没有工作。

更关键的是距离尺度统计：

- 全体 target 样本：
  - `min_distance_mean_epoch ≈ 6.54 ~ 6.62`
  - `min_distance_min_epoch ≈ 2.34 ~ 2.59`
- `hard-unknown` 子集：
  - `hard_min_distance_mean_epoch ≈ 7.24 ~ 7.30`
  - `hard_min_distance_min_epoch ≈ 5.95 ~ 5.97`
  - `hard_min_distance_max_epoch ≈ 8.93 ~ 9.01`

而此前尝试的 `push_margin` 仅为：

- `0.35`
- `0.40`
- `0.50`
- `2.50`

这些数值全部远小于 `hard-unknown` 子集的最近锚点距离，因此导致：

> `relu(push_margin - d_min)` 在 Houston 上恒等于 0

最终结论：

- Houston 上当前问题不是分组失败；
- 也不是结构方向完全错误；
- 而是 `push_margin` **没有对齐 Houston 的真实距离量纲**，因此 push 分支始终未被激活。

### 8.5 当前 Houston 阶段结论

据此，当前 Houston 的最合理推进策略为：

- 暂时继续以 `push_only` 作为优先结构方向；
- 不再使用 `0.x` 或 `2.x` 的小 `push_margin`；
- 下一轮实验应直接试探与 `hard_min_distance_min_epoch ≈ 5.95` 同量级的 `push_margin`，如：
  - `6.0`
  - `7.0`
  - 更高必要值

因此，Houston 当前尚未进入正式十种子验证，而是继续处于**结构诊断后的有效区间搜索阶段**。

### 8.6 Houston 第三轮有效区间搜索（H14~H16）

在确认 `push_margin` 必须与 Houston 的真实距离量纲对齐后，继续测试与 `hard_min_distance_min_epoch ≈ 5.95` 同量级的 `push_margin`，结果如下：

| 编号 | 配置摘要 | OA | OA_known | Unknown | HOS | 结论 |
|---|---|---:|---:|---:|---:|---|
| H14 | `push_only, push_margin=6.0, lambda_occr=0.02` | 0.6201 | 0.5867 | 0.7121 | 0.6360 | 当前最优 Houston 单 seed |
| H15 | `push_only, push_margin=7.0, lambda_occr=0.02` | 0.5950 | 0.5215 | 0.7978 | 0.6294 | unknown 提升，但 known 明显下降 |
| H16 | `push_only, push_margin=7.0, lambda_occr=0.05` | 0.5759 | 0.4814 | 0.8366 | 0.5758 | push 过强，整体失衡 |

与 Houston baseline `seed=1`（`HOS = 0.6348`）相比，可以得到：

- `H14` 终于实现了**轻微正增益**：`0.6348 -> 0.6360`；
- `H15` 继续增强 unknown 保护后，`Unknown` 有提升，但 `OA_known` 下滑，最终 `HOS` 反而下降；
- `H16` 进一步加大总强度后，`Unknown` 上升更明显，但 `known` 端明显受损，整体平衡恶化。

### 8.7 Houston 当前阶段总结

据此可以确认：

- `push_only` 方向是正确的；
- 当 `push_margin` 进入正确量纲后，Houston 上的 OCCR **终于开始真正生效**；
- 但当前 fixed-margin 版本存在明显的 known/unknown trade-off：
  - margin 太小：几乎不生效；
  - margin 稍大：unknown 上升，但 known 快速受损；
  - 权重再大：整体 HOS 下降。

因此，Houston 当前最优单 seed 配置可暂时记为：

- `H14`：`push_only + push_margin=6.0 + lambda_occr=0.02`

但由于提升幅度仅为 `0.0012`，当前仍**不建议直接进入十种子正式验证**。

下一步更合理的方向不是继续硬调固定 `push_margin`，而是转向：

> **自适应 push（Adaptive Push）**

即让 push 阈值不再是固定绝对值，而是能够根据 Houston 当前 batch 或当前 hard-unknown 子集的距离分布自适应调整，以缓解 fixed-margin 在不同数据集上量纲不稳定的问题。

---

## 补充：下一阶段待办（Adaptive Gating OCCR）

为避免后续遗忘，现将“自适应门控 OCCR”明确登记为下一阶段优先事项。

### 背景
当前实验已经表明：
- `PaviaC_OS` 更适合 `pull_push`；
- `Houston18_OS` 更适合 `adaptive_push_only`；
- 这说明不同数据集上，`pull` 与 `push` 的最优组合并不一致。

### 下一阶段目标
设计一个统一的 `Adaptive Gating OCCR`，不再手工为每个数据集固定 `occr_mode`，而是让模型根据当前 batch / 当前训练阶段 / 当前目标样本分布，自适应决定：
- 是否启用 `pull`；
- 是否仅保留 `push`；
- `pull` 与 `push` 的相对强度如何分配。

### 最小实现设想
建议从“轻量门控”开始，而不是一开始就上复杂网络：
- 定义门控系数 `g in [0,1]`；
- 用它控制 `pull` 分支权重，`(1-g)` 或独立系数控制 `push` 分支权重；
- `g` 的输入优先使用当前已存在的统计量，例如：
  - `num_pseudo_known / num_hard_unknown`
  - `hard_min_distance_mean_epoch`
  - `adaptive_margin_epoch`
  - `num_push_active`
- 第一版可以先做规则式门控；
- 若规则式门控有效，再升级为可学习门控模块。

### 预期收益
- 统一 `Pavia` 与 `Houston` 的方法叙事；
- 减少 dataset-specific 手工切换 `occr_mode` 的需求；
- 让 `OCCR` 从“手工实例化”提升为“数据驱动自适应实例化”；
- 为后续论文中的方法升级版本提供自然延伸点。

---

## 补充：参数覆盖逻辑修复说明（CLI 优先）

在 Houston 十种子正式对比前，发现当前代码中的参数合并逻辑存在一个重要实现问题：
- 旧版本 `mergeArgs(args, dataset)` 会把 `datasets/dataset_params.json` 中的参数无条件写回 `args`；
- 这意味着即使用户已经在命令行中显式传入了参数，仍然会被数据集默认参数覆盖。

### 旧逻辑的问题
旧逻辑的优先级实际为：

`dataset_params.json > 命令行参数 > argparse 默认值`

这会带来两个直接问题：
- baseline 命令可能被数据集中的 OCCR 默认配置“污染”；
- 手工调参实验也可能没有真正按照命令行设置运行，从而影响对实验结论的解释。

### 新逻辑的修复方式
当前已完成修复：
- 先解析一份 `argparse` 默认参数；
- 再解析用户真实传入的命令行参数；
- 自动识别哪些参数是用户显式改动过的；
- `dataset_params.json` 仅补充“未显式指定”的默认项，不再覆盖 CLI 参数。

修复后的优先级变为：

`命令行参数 > dataset_params.json > argparse 默认值`

### 与之前相比，最大的不同
现在的 `dataset_params.json` 只负责“按数据集补默认值”，不再拥有最高优先级。

因此：
- 你在命令里显式写的 baseline 参数，一定会生效；
- 你在命令里显式写的 OCCR / adaptive push 调参，也一定会生效；
- 只有你没有写的参数，才会从 `dataset_params.json` 中自动继承。

### 实验层面的影响
据此，Houston 的 baseline 与后续十种子正式对比，应以“参数优先级修复后”的结果为准。

也就是说：
- 先前可能受默认参数干扰的 baseline 结果，不宜继续作为最终对照；
- 后续正式表格中，应统一采用“CLI 优先版本”的重新运行结果。

---

## Houston 十种子正式结果（论文原始种子）

按照原论文随机种子 `1, 23, 30, 35, 52, 64, 68, 72, 73, 91`，对 `Houston13_7gt -> Houston18_OS` 进行了严格 baseline 与当前最优 OCCR 配置的十种子正式对比。

### 对比结果

| 配置 | OA | OA_known | Unknown | HOS |
|---|---:|---:|---:|---:|
| Baseline | 0.5830 ± 0.0334 | 0.5395 ± 0.0409 | 0.7030 ± 0.0355 | 0.6115 ± 0.0285 |
| OCCR (`adaptive_push_only`) | 0.5718 ± 0.0360 | 0.5076 ± 0.0467 | 0.7490 ± 0.0237 | 0.6163 ± 0.0359 |

### 结论
- `HOS` 从 `0.6115 ± 0.0285` 提升到 `0.6163 ± 0.0359`，说明当前 Houston 版本 **平均有效**；
- `Unknown` 从 `0.7030 ± 0.0355` 提升到 `0.7490 ± 0.0237`，说明 `adaptive_push_only` 对未知类拒识确实有帮助；
- `OA_known` 从 `0.5395 ± 0.0409` 降到 `0.5076 ± 0.0467`，说明当前改进仍以牺牲部分已知类识别为代价；
- `HOS` 标准差由 `0.0285` 增加到 `0.0359`，说明跨种子稳定性仍未达到理想状态。

### 补充诊断：统一 `soft_occr` 在 Houston 上的正确口径
在后续复现中，对 `Houston13_7gt -> Houston18_OS, seed=1` 进行了额外核查，得到如下结果：

| 配置 | OA | OA_known | Unknown | HOS |
|---|---:|---:|---:|---:|
| Baseline (`lambda_occr=0, lambda_pl=0`) | 0.6189 | 0.6118 | 0.6386 | 0.6208 |
| `soft_occr` + `lambda_pull=1, lambda_push=1` | 0.5662 | 0.6237 | 0.4076 | 0.4939 |
| `soft_occr` + `lambda_pull=0, lambda_push=1` | 0.6184 | 0.6074 | 0.6488 | 0.6228 |

据此可确认：
- Houston 当前不适合在 `soft_occr` 中同时保留强 `pull` 项；
- 将 `lambda_pull` 置为 `0` 后，`soft_occr` 退化为“统一软权重形式下的 push-only 版本”，其表现与此前 Houston 的有效结论一致；
- 因此，当前仓库中 `Houston18_OS` 的默认 `soft_occr` 配置应记录为：`lambda_pull=0, lambda_push=1, lambda_occr=0.02`；
- `PaviaC_OS` 仍保持 `lambda_pull=1, lambda_push=1`，说明同一 `soft_occr` 结构在不同数据集上的最优分支权重不同。

这也说明：
- `soft_occr` 作为统一结构是成立的；
- 但 Houston 的最优工作点更接近“仅保留 soft push”的特例；
- 下一阶段若要形成更强论文叙事，应继续推进“自适应门控/自适应 pull-push 权重分配”，而不是手工长期固定数据集特定权重。

### 下一步已落地的最小创新：Adaptive Pull Gating
为统一 `PaviaC_OS` 与 `Houston18_OS` 的 `soft_occr` 叙事，当前已在代码中落地一个**最小可开关版本**的自适应门控：

- 不再新增新的 `occr_mode`；
- 仍统一使用 `soft_occr`；
- 仅对 `pull` 分支做动态门控，`push` 分支保持主线稳定；
- 门控输入使用当前 batch 的 `soft_pull_weight_mean`，再配合 EMA 平滑；
- 当 `soft_pull_weight_mean` 持续较低时，自动压低 `pull` 强度；当其较高时，自动恢复 `pull` 作用。

当前代码开关与参数为：

- `adaptive_pull_gate`
- `pull_gate_center`
- `pull_gate_scale`
- `gate_momentum`
- `gate_stable_epochs`

建议首轮单 seed 验证命令：

```bash
python main.py --log_name WGDT_GATE_SOFT_Pavia_seed7 --source_dataset PaviaU_7gt --target_dataset PaviaC_OS --seed 7 --adaptive_pull_gate True
```

```bash
python main.py --log_name WGDT_GATE_SOFT_Houston_seed1 --source_dataset Houston13_7gt --target_dataset Houston18_OS --seed 1 --adaptive_pull_gate True
```

这一版本的定位不是最终论文版，而是用于验证：

- 是否可以在**统一 `soft_occr` 结构**下，自动让 Pavia 保留更多 `pull`、让 Houston 抑制 `pull`；
- 若该机制有效，则后续可继续升级为更完整的 `Adaptive Gating OCCR`。

### 当前代码已升级到双分支门控最小版
在完成上述单侧门控验证后，当前代码已继续升级为一个更接近完整 `Adaptive Gating OCCR` 的最小实现：

- `adaptive_pull_gate=True`：仅对 `pull` 分支进行自适应门控；
- `adaptive_full_gate=True`：在 `soft_occr` 下同时进行
  - `pull` 分支自适应抑制/保留；
  - `push` 分支自适应增强。

当前完整门控版本仍保持保守设计：

- `pull` 通过 `gate_pull` 调节；
- `push` 不做削弱，只在 `1.0` 基础上做有限增强；
- 从而避免在跨数据集条件下因过强门控而破坏已有稳定的 push 主线。

建议下一步单 seed 验证命令：

```bash
python main.py --log_name WGDT_FULLGATE_Pavia_seed7 --source_dataset PaviaU_7gt --target_dataset PaviaC_OS --seed 7 --adaptive_full_gate True
```

```bash
python main.py --log_name WGDT_FULLGATE_Houston_seed1 --source_dataset Houston13_7gt --target_dataset Houston18_OS --seed 1 --adaptive_full_gate True --lambda_pull 1.0 --lambda_push 1.0
```

### 补充结果：Adaptive Pull Gating 单种子验证
随后完成了两组关键单种子验证：

| 配置 | OA | OA_known | Unknown | HOS |
|---|---:|---:|---:|---:|
| `Pavia + soft_occr + adaptive_pull_gate=True` | 0.8803 | 0.8406 | 0.9168 | 0.8804 |
| `Houston + soft_occr + adaptive_pull_gate=True` | 0.6184 | 0.6074 | 0.6488 | 0.6228 |
| `Houston + soft_occr + adaptive_pull_gate=True + lambda_pull=1` | 0.6184 | 0.6074 | 0.6488 | 0.6228 |

据此可得到当前阶段最重要的判断：

- `Pavia` 上，自适应门控版本取得了非常强的单 seed 结果，说明门控没有破坏 `pull+push` 主线，且具备进一步提升潜力；
- `Houston` 上，当显式将 `lambda_pull` 从 `0` 改为 `1` 后，性能没有退回到静态 `lambda_pull=1` 时的 `HOS=0.4939`，而仍保持在 `HOS=0.6228`；
- 这说明当前的 `Adaptive Pull Gating` 已经具备“自动压制 Houston 中有害 pull 项”的能力；
- 因而，这一模块已不再只是调参技巧，而是一个**统一 `soft_occr` 结构下的机制级最小创新雏形**。

但同时也要保持谨慎：

- 随后补充的日志验证已经给出直接证据：
  - `Pavia`：`gate_pull_raw_epoch=0.444`, `gate_pull_epoch=0.984`, `soft_pull_weight_epoch=0.444`, `soft_push_weight_epoch=0.539`
  - `Houston`：`gate_pull_raw_epoch=0.261`, `gate_pull_epoch=0.526`, `soft_pull_weight_epoch=0.261`, `soft_push_weight_epoch=0.650`
- 这说明当前门控并非恒等映射或固定关闭 `pull`，而是已经在跨数据集上学出了**显著不同的工作点**：
  - `Pavia` 更偏向保留 `pull`；
  - `Houston` 更偏向抑制 `pull`、强化 `push`。
- 因此，`Adaptive Pull Gating` 已可正式视为：
  - **统一 `soft_occr` 结构下成立的机制级创新雏形**；
  - **下一阶段完整 `Adaptive Gating OCCR` 的直接前置证据**。

### 阶段定位
据此，Houston 当前应定位为：
- **跨数据集可迁移性的补充证据**；
- **`adaptive_push_only` 有效性的实证支持**；
- 同时也是引出下一阶段 `Adaptive Gating OCCR` 的关键动机。

换言之，Houston 当前结果可以写入论文，但更适合支撑“方法可迁移但仍需稳健化”的论述，而不宜作为当前阶段最核心的性能亮点。

---

## 当前主线与命令清单

### 当前主线口径
结合当前已完成的十种子结果，Phase 1 的“性能主线”应按**已验证最优工作点**记录，而不是按统一叙事口径直接覆盖：

- `PaviaC_OS`：继续以保守 OCCR / `R4` 作为当前默认主线；
- `Houston18_OS`：`adaptive_push_only` 仍是历史上最优的 Houston 工作点，但在当前代码版本下尚待严格复现；
- `soft_occr + Adaptive Pull Gating`：保留为候选统一机制线，但当前这轮十种子结果尚不足以升级为默认主线；
- `Unknownness Head`：保留为第二阶段候选升级线，当前不进入默认主线。

需要特别说明：

- 仓库中的部分默认参数仍保留统一 `soft_occr` 口径；
- 但论文与实验记录中的“当前主线”应以十种子验证后的性能主线为准；
- 因此，现阶段不再将 `WGDT_MAIN_*` 或 `WGDT_UNKNOWN_*` 直接表述为默认主线结果。

### 单种子命令

#### 1. Baseline

为保证论文对照更严格，建议正式使用 **BASE_CLEAN** 口径：

- `lambda_occr=0`
- `lambda_pl=0`
- `warmup_epochs=150`
- `pseudo_last_epochs=0`
- `adaptive_pull_gate=False`
- `adaptive_full_gate=False`

这样可以确保：

- OCCR 不会进入有效训练阶段；
- late pseudo-label 不会启用；
- 门控逻辑不会参与；
- baseline 仅保留原始 WGDT 主干（`source supervision + DANN + Radius`）。

**Pavia seed=7**

```bash
python main.py --log_name WGDT_BASE_CLEAN_Pavia_seed7 --source_dataset PaviaU_7gt --target_dataset PaviaC_OS --seed 7 --lambda_occr 0 --lambda_pl 0 --warmup_epochs 150 --pseudo_last_epochs 0 --adaptive_pull_gate False --adaptive_full_gate False
```

**Houston seed=1**

```bash
python main.py --log_name WGDT_BASE_CLEAN_Houston_seed1 --source_dataset Houston13_7gt --target_dataset Houston18_OS --seed 1 --lambda_occr 0 --lambda_pl 0 --warmup_epochs 150 --pseudo_last_epochs 0 --adaptive_pull_gate False --adaptive_full_gate False
```

#### 2. Houston 历史最优待复现线（`adaptive_push_only`）

**Houston seed=1**

```bash
python main.py --log_name WGDT_APO_Houston_seed1 --source_dataset Houston13_7gt --target_dataset Houston18_OS --seed 1 --occr_mode adaptive_push_only --adaptive_pull_gate False --adaptive_full_gate False --lambda_pull 0 --lambda_push 1 --lambda_occr 0.02
```

#### 3. 候选统一机制线（`soft_occr` 工作点版）

**Pavia seed=7**

```bash
python main.py --log_name WGDT_SOFT_MAIN_Pavia_seed7 --source_dataset PaviaU_7gt --target_dataset PaviaC_OS --seed 7
```

**Houston seed=1**

```bash
python main.py --log_name WGDT_SOFT_MAIN_Houston_seed1 --source_dataset Houston13_7gt --target_dataset Houston18_OS --seed 1
```

#### 4. Adaptive Pull Gating

**Pavia seed=7**

```bash
python main.py --log_name WGDT_GATE_SOFT_Pavia_seed7 --source_dataset PaviaU_7gt --target_dataset PaviaC_OS --seed 7 --adaptive_pull_gate True
```

**Houston seed=1**

```bash
python main.py --log_name WGDT_GATE_SOFT_Houston_seed1 --source_dataset Houston13_7gt --target_dataset Houston18_OS --seed 1 --adaptive_pull_gate True --lambda_pull 1.0 --lambda_push 1.0
```

#### 5. Adaptive Full Gate（当前试验版）

**Pavia seed=7, 保守增强版 B02**

```bash
python main.py --log_name WGDT_FULLGATE_Pavia_seed7_B02 --source_dataset PaviaU_7gt --target_dataset PaviaC_OS --seed 7 --adaptive_full_gate True --push_gate_boost 0.2 --push_gate_center 0.65
```

**Houston seed=1, 保守增强版 B02**

```bash
python main.py --log_name WGDT_FULLGATE_Houston_seed1_B02 --source_dataset Houston13_7gt --target_dataset Houston18_OS --seed 1 --adaptive_full_gate True --lambda_pull 1.0 --lambda_push 1.0 --push_gate_boost 0.2 --push_gate_center 0.65
```

### 十种子命令

#### 1. Baseline 十种子

建议正式表格统一使用 **BASE_CLEAN** 十种子结果。

**Pavia 十种子**

```bash
python summarize_pavia_results.py --mode all --log_name WGDT_BASE_CLEAN_Pavia10 --source_dataset PaviaU_7gt --target_dataset PaviaC_OS --gpus 0 --seeds 7 43 45 46 66 67 77 81 88 98 --lambda_occr 0 --lambda_pl 0 --warmup_epochs 150 --pseudo_last_epochs 0 --adaptive_pull_gate False --adaptive_full_gate False --save summary_WGDT_BASE_CLEAN_Pavia10.md
```

**Houston 十种子**

```bash
python summarize_pavia_results.py --mode all --log_name WGDT_BASE_CLEAN_Houston10 --source_dataset Houston13_7gt --target_dataset Houston18_OS --gpus 0 --seeds 1 23 30 35 52 64 68 72 73 91 --lambda_occr 0 --lambda_pl 0 --warmup_epochs 150 --pseudo_last_epochs 0 --adaptive_pull_gate False --adaptive_full_gate False --save summary_WGDT_BASE_CLEAN_Houston10.md
```

#### 2. Houston 历史最优待复现线（`adaptive_push_only`）十种子

**Houston 十种子**

```bash
python summarize_pavia_results.py --mode all --log_name WGDT_APO_Houston10 --source_dataset Houston13_7gt --target_dataset Houston18_OS --gpus 0 1 2 3 --seeds 1 23 30 35 52 64 68 72 73 91 --occr_mode adaptive_push_only --adaptive_pull_gate False --adaptive_full_gate False --lambda_pull 0 --lambda_push 1 --lambda_occr 0.02 --save summary_WGDT_APO_Houston10.md
```

#### 3. 候选统一机制线（`soft_occr` 工作点版）十种子

**Pavia 十种子**

```bash
python summarize_pavia_results.py --mode all --log_name WGDT_SOFT_MAIN_Pavia10 --source_dataset PaviaU_7gt --target_dataset PaviaC_OS --gpus 0 --seeds 7 43 45 46 66 67 77 81 88 98 --save summary_WGDT_SOFT_MAIN_Pavia10.md
```

**Houston 十种子**

```bash
python summarize_pavia_results.py --mode all --log_name WGDT_SOFT_MAIN_Houston10 --source_dataset Houston13_7gt --target_dataset Houston18_OS --gpus 0 --seeds 1 23 30 35 52 64 68 72 73 91 --save summary_WGDT_SOFT_MAIN_Houston10.md
```

#### 4. Adaptive Pull Gating 十种子

**Pavia 十种子**

```bash
python summarize_pavia_results.py --mode all --log_name WGDT_GATE_SOFT_Pavia10 --source_dataset PaviaU_7gt --target_dataset PaviaC_OS --gpus 0 1 2 3 --seeds 7 43 45 46 66 67 77 81 88 98 --adaptive_pull_gate True --save summary_WGDT_GATE_SOFT_Pavia10.md
```

**Houston 十种子**

```bash
python summarize_pavia_results.py --mode all --log_name WGDT_GATE_SOFT_Houston10 --source_dataset Houston13_7gt --target_dataset Houston18_OS --gpus 0 1 2 3 --seeds 1 23 30 35 52 64 68 72 73 91 --adaptive_pull_gate True --lambda_pull 1.0 --lambda_push 1.0 --save summary_WGDT_GATE_SOFT_Houston10.md
```

#### 5. Adaptive Full Gate（B02）十种子

**Pavia 十种子**

```bash
python summarize_pavia_results.py --mode all --log_name WGDT_FULLGATE_Pavia10_B02 --source_dataset PaviaU_7gt --target_dataset PaviaC_OS --gpus 0 --seeds 7 43 45 46 66 67 77 81 88 98 --adaptive_full_gate True --push_gate_boost 0.2 --push_gate_center 0.65 --save summary_WGDT_FULLGATE_Pavia10_B02.md
```

**Houston 十种子**

```bash
python summarize_pavia_results.py --mode all --log_name WGDT_FULLGATE_Houston10_B02 --source_dataset Houston13_7gt --target_dataset Houston18_OS --gpus 0 --seeds 1 23 30 35 52 64 68 72 73 91 --adaptive_full_gate True --lambda_pull 1.0 --lambda_push 1.0 --push_gate_boost 0.2 --push_gate_center 0.65 --save summary_WGDT_FULLGATE_Houston10_B02.md
```

### 当前 Pavia10 阶段性结论（基于已完成结果）
截至目前，Pavia 十种子已经完成如下三组对比：

- `Baseline`：`HOS = 0.7519 ± 0.0825`
- `静态 soft_occr`：`HOS = 0.7323 ± 0.1087`
- `Adaptive Pull Gating`：`HOS = 0.7573 ± 0.1202`

据此可得：

- `Adaptive Pull Gating` 的十种子均值目前略优于 baseline，但优势较小；
- `静态 soft_occr` 在 Pavia 十种子上整体不优于 baseline；
- `Adaptive Pull Gating` 的亮点主要体现在部分 seed（如 `seed=7`）非常强，但跨 seed 稳定性仍需进一步观察；
- 因此，Pavia 当前尚不能只凭单 seed 高点下最终定论，正式结论仍应以后续统一口径下的 `BASE_CLEAN` 对照为准。

### 最新十种子复核：`BASE` vs `MAIN` vs `UNKNOWN`

在重新整理 `base / main / unknown_head` 三条线的十种子结果后，当前结论进一步明确：

#### Pavia

| 配置 | OA | OA_known | Unknown | HOS |
|---|---:|---:|---:|---:|
| Baseline | 0.8043 ± 0.0494 | 0.7023 ± 0.0556 | 0.8981 ± 0.0673 | 0.7772 ± 0.0474 |
| `WGDT_MAIN_Pavia10` | 0.7793 ± 0.0607 | 0.7700 ± 0.0552 | 0.7878 ± 0.0909 | 0.7704 ± 0.0550 |
| `WGDT_UNKNOWN_Pavia10` | 0.7814 ± 0.0873 | 0.7720 ± 0.0566 | 0.7900 ± 0.1438 | 0.7686 ± 0.0943 |

结论：

- `MAIN` 与 `UNKNOWN` 都提高了 `OA_known`，但 `Unknown` 明显下降；
- 二者的 `HOS` 均低于 `Baseline`；
- 因此，这两条线当前均不应覆盖 `Pavia` 既有主线结论。

#### Houston

| 配置 | OA | OA_known | Unknown | HOS |
|---|---:|---:|---:|---:|
| Baseline | 0.5830 ± 0.0334 | 0.5395 ± 0.0409 | 0.7030 ± 0.0355 | 0.6115 ± 0.0285 |
| `adaptive_push_only` | 0.5718 ± 0.0360 | 0.5076 ± 0.0467 | 0.7490 ± 0.0237 | 0.6163 ± 0.0359 |
| `WGDT_MAIN_Houston10` | 0.5809 ± 0.0236 | 0.6120 ± 0.0234 | 0.4952 ± 0.1161 | 0.5327 ± 0.0620 |
| `WGDT_UNKNOWN_Houston10` | 0.5784 ± 0.0256 | 0.6050 ± 0.0240 | 0.5048 ± 0.1318 | 0.5347 ± 0.0790 |

结论：

- 历史结果中，`adaptive_push_only` 是当前唯一在 Houston 十种子均值意义下优于 `Baseline` 的配置；
- 但在当前代码版本下重新复现时，尚未稳定回到历史表中的 `HOS = 0.6163 ± 0.0359`；
- `MAIN` 与 `UNKNOWN` 虽提升了 `OA_known`，但显著破坏了 `Unknown`，导致 `HOS` 大幅下降；
- 因此，Houston 现阶段更准确的定位应为：`adaptive_push_only` 是历史最优待复现线，`soft_occr + Adaptive Pull Gating` 与 `Unknownness Head` 为候选升级线，三者都暂不能直接作为“当前已复现默认主线”。
