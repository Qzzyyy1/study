顶级期刊（TPAMI/TGRS/ISPRS）论文规划书 V2：基于 WGDT 的可执行升级路线

## 1. 文档定位

本版本不再把“宏大叙事”直接作为第一阶段开发蓝图，而是将规划拆分为：

- **Phase 1 主线**：立刻可执行、可复现、可做消融、可快速定位收益来源；
- **Phase 2 扩展线**：在主线稳定后，再逐步升级到更强的理论与结构版本。

当前代码的真实任务定位是：

> **高光谱图像（HSI）开集域适应（Open-Set Domain Adaptation, OSDA）**

当前代码的真实基础框架不是笼统的“CNN+CBAM”，而是：

> **DCRN 双分支 3D CNN 特征提取 + 通道/空间注意力 + 类别规范锚点（Class Canonical Anchors）距离分类 + 可学习 Radius 拒识 + 加权 DANN 域对齐**

因此，本规划书的核心目标不是“一次性推翻重写”，而是：

> **在保留当前 WGDT 主体闭环的前提下，最小侵入式地引入目标域结构整形机制与晚期微调机制，从而稳定提升 OSDA 性能，并为后续扩展到更强模型打下基础。**

---

## 2. 当前代码基线的准确描述

为保证后续论文表述与代码事实一致，先统一术语。

### 2.1 Backbone 与注意力

- 当前主干是 `DCRN`；
- 内部包含光谱-空间双分支 3D 卷积结构；
- 并在融合后使用通道注意力与空间注意力；
- 因此后文统一称其为：**attention-enhanced DCRN backbone**。

### 2.2 锚点空间的真实含义

当前 `Anchor` 模块中的锚点并不是由源域样本动态统计得到的“类原型中心（prototype center）”，而是由固定正交基构造得到的**类别规范锚点**：

> **Class Canonical Anchors**

这一定义必须在后续实验记录、论文草稿和图示说明中保持一致。

这样表述的好处有两点：

- 与当前代码完全一致；
- 能清楚区分“固定规范锚点”与未来可能扩展的“动态原型中心”。

### 2.3 当前开集判决机制

当前代码中的拒识并不是简单的静态阈值后处理，而是：

- 先在 anchor 空间中计算每个样本到各类别规范锚点的距离；
- 再构造 `gamma` 分数；
- 最后通过可学习的 `Radius` 对目标域样本进行 known / unknown 判决。

因此，第一阶段新增方法必须尽量复用：

- 当前的 `distance`；
- 当前的 `gamma`；
- 当前的 `radius`；
- 当前的 `CACLoss`；
- 当前的 `DANN`。

---

## 3. 第一阶段总目标：在现有 WGDT 上做最小侵入式增强

### 3.1 第一阶段只做两件事

第一阶段（Phase 1）只引入以下两个新增模块：

1. **OCCR：Open-set Cross-domain Contrastive Regularization**
2. **Late-stage High-confidence Pseudo-label Refinement**

### 3.2 第一阶段明确不做的内容

为控制变量、降低工程风险、确保收益来源清晰，以下内容**不进入第一阶段主线**：

- 不替换 Backbone 为 Mamba；
- 不替换判决头为 EDL；
- 不替换对齐机制为 UOT；
- 不升维到 Open-World Novel Class Discovery；
- 不新增 feature projection head；
- 不把固定 canonical anchors 改成 prototype centers。

### 3.3 第一阶段方法主线

第一阶段主线定义为：

> **Attention-enhanced WGDT + OCCR + Late-stage Pseudo-label Refinement**

也即：

- 保留当前 `DCRN + Canonical Anchor + Radius + DANN` 主框架；
- 在目标域训练过程中新增“主动推拉”结构正则；
- 在训练末期仅使用极高置信目标样本进行伪监督微调。

---

## 4. 核心创新一：开放集跨域对比正则 OCCR

## 4.1 命名与定位

为避免与开放集识别领域常见的 `OSCR` 评估指标发生歧义，本文将新增训练正则命名为：

> **OCCR: Open-set Cross-domain Contrastive Regularization**

它不是评估指标，而是一个**训练期结构整形模块**。

### 4.2 设计动机

当前 WGDT 的主要能力体现在：

- 源域监督下已知类 anchor 空间的构建；
- 基于目标样本可靠性的加权域对齐；
- 基于可学习半径的 unknown 拒识。

但当前方法仍以“判决/拒识”为主，缺少一个明确的目标域结构塑形机制，即：

- 对“高置信伪已知”的目标样本，没有主动拉近其类别锚点；
- 对“高置信硬未知”的目标样本，没有主动推离已知类边界。

OCCR 的作用正是把这种“被动判决”升级为“训练期主动整形”。

### 4.3 目标样本三分组策略

第一阶段采用围绕可学习半径 `r` 的**固定偏移双阈值策略**。

记目标样本的最小 `gamma` 为 `g(x)`，当前可学习半径为 `r`，则：

- **pseudo-known**：`g(x) < r - delta_low`
- **hard-unknown**：`g(x) > r + delta_high`
- **ambiguous**：`r - delta_low <= g(x) <= r + delta_high`

其中：

- `delta_low` 用于给 known 区留出安全间隔；
- `delta_high` 用于给 unknown 区留出安全间隔；
- 中间区域作为缓冲带，避免阈值附近样本在训练中频繁跳变。

第一阶段对 ambiguous 样本的策略非常明确：

> **在新增模块中完全忽略 ambiguous 样本。**

也就是说，它们仍然参与原有 WGDT 的 DANN 与 Radius 学习，但不参与 OCCR，也不参与后续伪标签监督。

### 4.4 OCCR 的最小实现形式

第一阶段不在 feature 空间新增投影头，而是直接在当前的 **anchor / distance 空间** 中构造 OCCR。

对每个目标样本，记：

- `d_min(x)`：该样本到最近类别规范锚点的距离；
- `c_near(x)`：最近类别规范锚点；
- `X_pk`：pseudo-known 目标样本集合；
- `X_hu`：hard-unknown 目标样本集合。

则第一阶段的 OCCR 定义为：

```math
L_{pull} = \mathbb{E}_{x \in X_{pk}} [d_{min}(x)]
```

```math
L_{push} = \mathbb{E}_{x \in X_{hu}} [\max(0, m_{push} - d_{min}(x))]
```

```math
L_{OCCR} = L_{pull} + \lambda_{push} L_{push}
```

其中：

- `L_pull`：把高置信伪已知样本拉向最近类别规范锚点；
- `L_push`：仅当 hard-unknown 样本距离最近锚点仍小于安全边界 `m_push` 时，才继续施加排斥；
- `lambda_push`：控制排斥项强度。

### 4.5 为什么采用“最近锚点 Margin Push”

第一阶段明确采用：

> **Nearest-anchor margin push**

而不是“对所有锚点同时做平均排斥”。原因如下：

- Unknown 样本通常只与极少数相似的已知类发生混淆；
- 只对最近锚点施加排斥，梯度方向最集中；
- 当 unknown 样本已经被推到安全边界之外时，损失自然变为 0；
- 可避免全锚点排斥造成的梯度抵消与特征空间外推。

### 4.6 OCCR 在方法中的角色

OCCR 不替代当前的 DANN 与 Radius，而是作为额外结构项叠加到原 WGDT 中：

- `DANN`：负责宏观域对齐；
- `Radius`：负责边界学习与拒识；
- `OCCR`：负责目标域已知/未知结构的微观整形。

这三者在第一阶段是互补关系，而非替代关系。

---

## 5. 核心创新二：晚期高置信伪标签微调

### 5.1 设计动机

在训练中后期，经过源域监督、域对齐、半径学习和 OCCR 主动整形后，目标域中一部分样本会形成非常稳定的已知类归属。

此时，如果只停留在“柔性结构约束”层面，往往仍会留下少量边界模糊的 known 样本。为进一步榨干性能，需要在训练末期引入：

> **Late-stage High-confidence Pseudo-label Refinement**

### 5.2 晚期伪标签的严格筛选原则

晚期伪标签与 OCCR 中的 pseudo-known 不完全相同。

第一阶段明确设置一个更严格的阈值 `delta_pseudo`，并满足：

```math
delta_{pseudo} > delta_{low}
```

对目标样本，仅当：

```math
g(x) < r - delta_{pseudo}
```

时，才允许其进入晚期伪标签监督集合。

这样设计的原因是：

- `OCCR pull` 只是结构引导，可以适当宽松；
- `pseudo-label` 会被当作硬标签输入强监督项，必须更严格；
- 通过更纯净的样本集合，尽量避免 Confirmation Bias。

### 5.3 监督方式

第一阶段不额外引入 SupCon 等新型对比学习分支，而是**直接复用当前已存在的 `CACLoss`**。

具体做法为：

- 对满足高置信筛选条件的目标样本，取其当前预测类别作为伪标签；
- 将其直接送入当前 `Anchor` 模块已有的 `CACLoss(distance, pseudo_y)`；
- 作为晚期额外监督项叠加回总损失中。

这种做法的优点在于：

- 完全复用现有代码路径；
- 与源域监督保持一致的度量空间约束；
- 同时具备 anchor pull 与 tuplet push 的双重作用；
- 工程成本最低、稳定性最高。

---

## 6. 第一阶段完整训练日程

### 6.1 总体训练流程

设总训练轮数为 `150`，则第一阶段建议训练日程为：

#### 阶段 A：Warm-up（epoch 0 ~ 14）

- 使用原始 WGDT 训练；
- 不启用 OCCR；
- 不启用 late pseudo-label。

目的：

- 先让 backbone、canonical anchor 空间、radius 与 DANN 建立基本稳定性；
- 避免过早使用目标域伪结构信息带来噪音。

#### 阶段 B：结构整形（epoch 15 ~ 129）

- 使用原始 WGDT；
- 启用 OCCR；
- 不启用 late pseudo-label。

目的：

- 逐步将高置信 known 拉向已知类锚点；
- 将高置信 unknown 推离已知类边界；
- 构建更清晰的开集结构。

#### 阶段 C：末期锐化（epoch 130 ~ 149）

- 使用原始 WGDT；
- 继续启用 OCCR；
- 额外启用 late pseudo-label。

目的：

- 在已经成形的已知/未知结构基础上，进一步锐化 known 类边界；
- 提升 `OA_known`、`AA_known` 与最终 `HOS`。

### 6.2 叠加而非替换

末期伪标签不是替换 OCCR，而是与 OCCR 叠加使用：

> **原 WGDT + OCCR + Late Pseudo-label**

这种 additive 策略可以最大程度保证梯度连续性与训练稳定性。

---

## 7. 第一阶段总损失表达

记：

- `L_src`：源域已有监督项（来自当前源域 anchor / tuplet 训练）；
- `L_dann`：当前加权域对抗损失；
- `L_radius`：当前半径学习损失；
- `L_occr`：新增 OCCR；
- `L_pl`：晚期伪标签 CACLoss；
- `lambda_occr`：OCCR 总权重；
- `lambda_pl`：晚期伪标签权重。

则第一阶段整体训练目标可写为：

### 阶段 A（Warm-up）

```math
L = L_{src} + L_{dann} + L_{radius}
```

### 阶段 B（结构整形）

```math
L = L_{src} + L_{dann} + L_{radius} + \lambda_{occr} L_{OCCR}
```

### 阶段 C（末期锐化）

```math
L = L_{src} + L_{dann} + L_{radius} + \lambda_{occr} L_{OCCR} + \lambda_{pl} L_{pl}
```

其中，`lambda_occr` 在 warm-up 结束后建议采用**线性升权**策略，而 `lambda_pl` 仅在最后 20 个 epoch 激活。

---

## 8. 第一阶段建议默认超参数

为了尽快打通第一版实现，建议先固定以下默认超参数起点：

- `epochs = 150`
- `warmup_epochs = 15`
- `pseudo_last_epochs = 20`
- `delta_low = 0.05`
- `delta_high = 0.05`
- `delta_pseudo = 0.10`
- `push_margin = 0.20`
- `lambda_push = 1.0`
- `lambda_occr = 0.10`
- `lambda_pl = 0.10`

说明：

- `delta_pseudo > delta_low` 已在默认设置中得到满足；
- 若第一版训练较稳但提升不足，可优先尝试微调 `lambda_occr` 与 `push_margin`；
- 若伪标签样本过少或过多，可优先微调 `delta_pseudo`。

---

## 9. 第一阶段日志与可解释性输出

第一阶段必须加强训练日志记录，否则无法分析阈值划分是否合理。

建议在每个 epoch 至少记录以下统计量：

- `num_pseudo_known`
- `num_ambiguous`
- `num_hard_unknown`
- `num_pseudo_label_used`
- `loss_occr_pull`
- `loss_occr_push`
- `loss_occr`
- `loss_pseudo`

这样做的意义在于：

- 可直接诊断双阈值划分是否失衡；
- 可观察随着训练推进，目标样本从 ambiguous 向 pseudo-known / hard-unknown 的迁移趋势；
- 可为论文绘制“目标域动态结构分离过程”提供素材。

---

## 10. 第一阶段实验执行策略

### 10.1 开发优先数据集

第一轮开发与调试只使用：

> **PaviaU_7gt -> PaviaC_OS**

原因：

- 当前代码默认配置即为该设置；
- 类别规模适中，适合快速验证训练逻辑与损失尺度；
- 便于与已有实验现象做直接对比。

### 10.2 开发默认随机种子

第一阶段开发默认使用：

> **seed = 7**

原因：

- 该随机种子属于原论文报告中使用过的种子集合；
- 先固定单种子打通流程，有利于快速比较新旧方法差异；
- 待主线跑通后，再补充原论文使用的完整 10 个随机种子实验。

原论文对应种子集合如下：

> `PaviaU_7gt -> PaviaC_OS [7, 43, 45, 46, 66, 67, 77, 81, 88, 98]`

### 10.3 第一阶段推荐消融顺序

建议严格按以下顺序做消融，避免变量过多：

1. `Baseline WGDT`
2. `WGDT + OCCR (pull only)`
3. `WGDT + OCCR (pull + push)`
4. `WGDT + OCCR + warm-up`
5. `WGDT + OCCR + warm-up + late pseudo-label`

如果主线结果稳定，再做以下补充消融：

6. `delta_low / delta_high` 敏感性分析
7. `push_margin` 敏感性分析
8. `lambda_occr / lambda_pl` 敏感性分析

### 10.4 第一阶段重点指标

当前代码中已经具备以下指标：

- `oa`
- `aa`
- `oa_known`
- `aa_known`
- `unknown`
- `hos`

第一阶段论文撰写时，建议重点围绕：

- `OA_known`
- `Unknown`
- `HOS`

展开分析，因为这三者最能体现“已知类识别能力 / 未知类拒识能力 / 二者平衡性”。

### 10.5 当前阶段性实验结论（Pavia 单种子）

截至目前，已在 `PaviaU_7gt -> PaviaC_OS`、`seed=7` 条件下完成以下代表性实验：

| 配置 | 关键超参 | OA | OA_known | Unknown | HOS | 结论 |
|---|---|---:|---:|---:|---:|---|
| Baseline WGDT | `lambda_occr=0, lambda_pl=0` | 0.8123 | 0.6780 | 0.9357 | 0.7770 | 当前强基线 |
| 激进 OCCR | `warmup=15, delta_low=0.05, delta_high=0.05, delta_pseudo=0.10, push_margin=0.20, lambda_occr=0.10, lambda_pl=0` | 0.4386 | 0.6864 | 0.2109 | 0.3221 | 明显伤害 unknown |
| 激进 OCCR + PL | `warmup=15, ... , lambda_occr=0.10, lambda_pl=0.10` | 0.4475 | 0.6680 | 0.2450 | 0.3601 | 略回升，但仍远差于 baseline |
| 保守 OCCR | `warmup=60, delta_low=0.15, delta_high=0.10, delta_pseudo=0.20, push_margin=0.30, lambda_occr=0.02, lambda_pl=0` | 0.8552 | 0.8106 | 0.8962 | 0.8390 | 当前最优配置 |
| 保守 OCCR + PL | `warmup=60, ... , lambda_occr=0.02, lambda_pl=0.05` | 0.8370 | 0.8070 | 0.8646 | 0.8225 | 优于 baseline，但略弱于保守 OCCR |

据此可以得到当前阶段的经验结论：

- `OCCR` 不是不能用，但**不能激进启用**；
- `warmup` 需要显著延后，`pseudo-known` 划分必须更保守；
- `lambda_occr` 必须较小，否则会显著牺牲 unknown 拒识能力；
- 当前在 `Pavia, seed=7` 上，**late pseudo-label 没有带来额外收益**；
- 若后续多种子验证仍成立，则论文主线应优先采用“保守 OCCR”，并将 late pseudo-label 作为可选增强项，而非主贡献核心。

因此，第一阶段默认配置不再建议使用最初的激进版本，而应先调整为保守版 OCCR 基线：

- `warmup_epochs = 60`
- `pseudo_last_epochs = 20`
- `delta_low = 0.15`
- `delta_high = 0.10`
- `delta_pseudo = 0.20`
- `push_margin = 0.30`
- `lambda_push = 1.0`
- `lambda_occr = 0.02`
- `lambda_pl = 0.0`

### 10.6 当前阶段性实验结论（Pavia 十种子）

在完成 `PaviaU_7gt -> PaviaC_OS` 的 10 个随机种子实验后，当前阶段结论进一步明确。

#### 10.6.1 Baseline 与保守 OCCR 的十种子均值对比

| 方法 | OA | OA_known | Unknown | HOS |
|---|---:|---:|---:|---:|
| Baseline WGDT | 0.8043 ± 0.0494 | 0.7023 ± 0.0556 | 0.8981 ± 0.0673 | 0.7772 ± 0.0474 |
| 保守 OCCR | 0.8084 ± 0.0946 | 0.7762 ± 0.0535 | 0.8380 ± 0.1373 | 0.7887 ± 0.0899 |

其中：

- `HOS` 从 `0.7772 ± 0.0474` 提升到 `0.7887 ± 0.0899`；
- `OA_known` 从 `0.7023 ± 0.0556` 提升到 `0.7762 ± 0.0535`，提升非常明显；
- `Unknown` 从 `0.8981 ± 0.0673` 降至 `0.8380 ± 0.1373`，说明 OCCR 在提升已知类识别时牺牲了一部分未知类拒识；
- `OA` 仅有小幅提升，说明主增益主要来自已知类结构建模增强，而非整体分类率的大幅跃升。

#### 10.6.2 十种子实验的经验判断

根据当前 10 个种子的统计结果，可以得出如下结论：

- 保守版 `OCCR` 是**有效的**，因为平均 `HOS` 相比 baseline 有提升；
- 这种提升的主要来源是 `OA_known` 的明显增强；
- 代价是 `Unknown` 有所下降，因此该方法更偏向“强化已知类判别能力”；
- `HOS` 的标准差从 `0.0474` 上升到 `0.0899`，说明当前 OCCR **稳定性仍然不够理想**；
- 综合来看，当前最合理的表述不是“显著全面优于 baseline”，而是：

> **保守版 OCCR 在十种子平均意义上提升了 HOS，并显著改善了已知类识别性能，但同时引入了更大的跨种子波动，仍需进一步做稳健性优化。**

#### 10.6.3 当前主线定位修订

因此，基于十种子结果，第一阶段主线应进一步调整为：

- 将保守版 `OCCR` 保留为 `Pavia` 的当前主线增强模块；
- 在 `Pavia` 十种子上，继续以 `R4` 作为当前默认配置；
- 在 `Houston` 十种子上，`adaptive_push_only` 仍保留为历史最佳工作点，但在当前代码版本下尚待严格复现；
- 将 `soft_occr + Adaptive Pull Gating` 重新定位为“候选统一机制线”，而不是已经完成十种子验证的默认主线；
- `Unknownness Head` 保持为第二阶段升级线，在没有新的十种子优势前不进入默认主线；
- `late pseudo-label` 在当前阶段仍不建议作为默认开启模块。

#### 10.6.4 当前最终默认配置（R4）

综合 `R1` 与 `R4` 的十种子对比，当前默认配置正式确定为 `R4`，理由如下：

- `HOS` 更高：`0.7931 ± 0.0489 > 0.7887 ± 0.0899`；
- `HOS` 标准差显著更低，稳定性明显更好；
- `Unknown` 更高：`0.8620 > 0.8380`，更符合开集域适应的目标；
- 虽然 `OA_known` 略低于 `R1`，但整体 known/unknown 平衡更优。

因此，当前默认配置更新为：

- `warmup_epochs = 60`
- `pseudo_last_epochs = 20`
- `delta_low = 0.20`
- `delta_high = 0.10`
- `delta_pseudo = 0.30`
- `push_margin = 0.30`
- `lambda_push = 1.0`
- `lambda_occr = 0.02`
- `lambda_pl = 0.0`

---

## 11. 第一阶段论文贡献建议表述

若第一阶段结果在多种子与更多数据集上继续成立，建议将论文贡献压缩并明确表述为三点：

### 贡献 1：基于类别规范锚点空间的目标域主动结构整形

- 在现有 WGDT 的 canonical anchor 空间中提出 OCCR；
- 将原有被动开集判决升级为训练阶段主动推拉约束；
- 通过 pseudo-known pull 与 hard-unknown margin push 增强目标域结构可分性。

### 贡献 2：面向开集域适应的双阈值缓冲带划分策略

- 围绕可学习 radius 构造 `pseudo-known / ambiguous / hard-unknown` 三分区；
- 利用缓冲带机制减少阈值附近样本的训练震荡；
- 提升目标域样本划分与伪监督的鲁棒性。

### 贡献 3：晚期高置信伪标签锐化机制

- 在训练末期利用极高置信目标样本进行 CACLoss 微调；
- 将中期的柔性结构对齐平滑过渡到末期的硬性类别收缩；
- 提升已知类边界清晰度与最终判决稳定性。

---

## 12. 第二阶段扩展线（暂不进入主线实现）

在第一阶段充分验证后，可继续向更高风险、更高上限的方向扩展。

### 12.1 Backbone 升级：SS-Mamba

潜在路线：

- 用更强的光谱-空间状态空间模型替换当前 DCRN；
- 重点验证其对长程光谱依赖建模的优势；
- 但该方向不应进入第一阶段主线，以免变量失控。

### 12.2 判决头升级：EDL

潜在路线：

- 用 Evidential Deep Learning 替代当前距离/半径式判决中的不确定性刻画；
- 将 uncertainty 直接纳入样本分组与伪标签筛选；
- 但该方向需要重构分类头与损失，不适合第一阶段并行引入。

### 12.3 域对齐升级：UOT

潜在路线：

- 用 UOT 替代当前 DANN；
- 在总质量不守恒假设下进行更精细的已知/未知对齐；
- 但这会同时改变对齐机制与训练稳定性来源，应在主线稳定后再做。

### 12.4 任务升维：Open-World Discovery

潜在路线：

- 对 hard-unknown 样本进一步做聚类或自监督发现；
- 从 OSDA 升级到 Open-World Domain Adaptation；
- 但这依赖新的标签保留策略与新评估协议，不属于第一阶段范畴。

---

## 13. 结论：当前应采取的开发策略

综合当前代码基础、工程可行性与论文写作逻辑，当前最合理的路线不是立刻做大重构，而是：

> **先把 Phase 1 主线做强，再把 Phase 2 扩展线作为冲顶增强。**

当前的直接行动原则为：

- 先不碰 Mamba / EDL / UOT / Open-World；
- 先在现有 WGDT 的 canonical anchor 空间内加入**保守版 OCCR**；
- `late pseudo-label` 目前保留为可选增强项，不作为默认主配置；
- 先用 `PaviaU_7gt -> PaviaC_OS`、`seed=7` 打通流程；
- 优先补充多种子验证，再扩展到更多数据集；
- 等主线稳定后，再做扩展线实验。

这条路线具备以下优势：

- 改动最小；
- 变量最少；
- 可复现性强；
- 最容易定位收益来源；
- 最适合作为当前阶段的论文主体方案。

---

## 补充规划：自适应门控 OCCR（Adaptive Gating OCCR）

在第一阶段结束后，建议将“自适应门控 OCCR”作为第二阶段优先扩展线，目的不是推翻当前 `OCCR`，而是在统一框架下自动选择更合适的结构实例化方式。

### 当前最新进展：已完成最小可行前置验证
在第一阶段后续开发中，已经完成一个最小可行版本：

> **Adaptive Pull Gating under soft OCCR**

其核心思想是：

- 不新增新的 `occr_mode`；
- 继续统一使用 `soft_occr`；
- 仅对 `pull` 分支引入自适应门控，`push` 分支保持稳定主线；
- 门控输入使用当前 batch 的 `soft_pull_weight_mean`，并用 EMA 进行平滑。

单 seed 验证结果表明：

- `PaviaU_7gt -> PaviaC_OS, seed=7`
  - `HOS = 0.8804`
  - `gate_pull_epoch = 0.984`
  - `soft_pull_weight_epoch = 0.444`
- `Houston13_7gt -> Houston18_OS, seed=1`
  - `HOS = 0.6228`
  - `gate_pull_epoch = 0.526`
  - `soft_pull_weight_epoch = 0.261`

这说明当前门控已经在跨数据集上学出了不同工作点：

- `Pavia` 更倾向保留 `pull`；
- `Houston` 更倾向压低 `pull`、偏向 `push`。

因此，第二阶段不再是“从 0 开始构思自适应门控”，而是：

> **从已验证有效的 Adaptive Pull Gating，升级到完整的 Adaptive Gating OCCR。**

### 统一方法叙事（候选升级线写法）
当前更合适的写法应调整为：

- `soft_occr + Adaptive Pull Gating` 仍可作为统一方法叙事的候选方向；
- 它在统一框架内同时建模 `soft pull` 与 `soft push`；
- 不同数据集上，最优工作点可以表现为不同的 `pull/push` 强度比例；
- 但截至当前十种子结果，它尚未稳定优于各数据集已验证的最优工作点；
- 因而当前不宜将其直接写成“Phase 1 默认主线”，而更适合写成：
  - `PaviaC_OS` 上的候选统一机制版本；
  - `Houston18_OS` 上的候选 push-dominant 近似版本；
- 这不应被理解为“换了两种方法”，但也不应在现阶段强行覆盖性能主线。

> **同一个 `soft_occr` 方法在不同目标域分布下，自然落在不同的最优结构实例化工作点。**

在此基础上，`Adaptive Pull Gating / Adaptive Gating OCCR` 的作用不是引入第二种方法，而是：

> **让同一个 `soft_occr` 框架自动学出更合适的工作点，而不必长期手工指定 `lambda_pull / lambda_push`。**

当前仓库中的参数默认值曾按此口径配置，但论文与实验记录中的 Phase 1 主线不再直接沿用该默认口径：

- 代码默认值仍可保留为统一机制试验入口；
- 但正式性能主线应改记为：`PaviaC_OS -> R4`，`Houston18_OS -> 历史最佳的 adaptive_push_only（待复现）`；
- `Adaptive Pull Gating` 当前更适合作为后续统一建模升级的实验基础，而不是已完成验证的默认结论。

### 设计动机
当前结果已经显示：
- `PaviaC_OS` 上，`pull_push` 更稳定有效；
- `Houston18_OS` 上，`adaptive_push_only` 更稳定有效；
- 因而 `OCCR` 的最优作用模式具有明显的数据集依赖性。

这提示我们：下一阶段应从“手工指定模式”迈向“模型自适应选择模式”。

### 核心思想
在统一的 `OCCR` 框架中引入门控变量，用于动态调节：
- `pull` 是否开启；
- `push` 是否加强；
- `pull/push` 的相对比例。

可写成概念形式：

```math
L_{OCCR}^{gate} = g \cdot L_{pull} + (1-g) \cdot L_{push}
```

其中 `g` 为门控系数，可由统计量驱动，也可由小型可学习模块预测。

### 第二阶段的明确升级目标
基于当前已经成立的 `Adaptive Pull Gating`，建议按以下顺序推进：

1. **从单侧门控升级到双侧门控**
   - 当前仅对 `pull` 分支进行门控；
   - 下一步改为同时建模 `gate_pull` 与 `gate_push`，或引入归一化耦合关系。
2. **从启发式 EMA 门控升级到更稳健的权重分配**
   - 保留当前低风险实现作为初始化；
   - 再尝试加入轻量可学习标量或小型 MLP。
3. **从单 seed 行为验证升级到正式统计验证**
   - 在 `Pavia` 与 `Houston` 上分别做正式多 seed 评估；
   - 报告门控值统计与性能提升的一致性。

### 推荐实施顺序
1. **规则式门控版本**
    - 基于 `pseudo-known` 占比、`hard-unknown` 占比、`push_active` 比例等统计量构造 `g`；
    - 先验证是否优于手工 `pull_push / adaptive_push_only` 切换。
2. **可学习门控版本**
   - 用轻量 MLP 或标量参数学习 `g`；
   - 控制新增参数量，避免破坏当前主干稳定性。
3. **论文升级版本**
   - 将其表述为“从静态 OCCR 到自适应 OCCR”的方法进化；
   - 形成跨数据集统一叙事。

### 与当前阶段的关系
- 当前阶段仍保持：`PaviaC_OS -> R4 / 保守 OCCR`，`Houston18_OS -> 历史最佳的 adaptive_push_only（待复现）`；
- 第二阶段再尝试用 `Adaptive Gating OCCR` 统一两条线路；
- 这样既不影响当前可复现实验主线，也为后续方法升级预留了明确路线。

---

## 补充说明：参数优先级修复

在方法实现层面，后续实验需统一采用如下参数优先级：

`命令行参数 > dataset_params.json > argparse 默认值`

原因在于：若 `dataset_params.json` 高于命令行参数，则不同数据集中的默认配置会干扰 baseline、消融实验与调参实验的可解释性。为此，当前代码已修复为：
- `dataset_params.json` 只补默认值；
- 命令行显式指定的参数始终保留；
- 从而保证 baseline、公平对照与结构消融都可被准确复现。

这一修复也意味着：
- `PaviaC_OS` 与 `Houston18_OS` 可以安全保留各自的数据集默认配置；
- 同时仍可通过命令行开展严格 baseline、严格 ablation 与严格调参；
- 后续论文表格中的正式结果，应以该修复后的参数优先级为准。

---

## 补充结果：Houston 十种子正式对比（论文原始种子）

在采用原论文种子 `1, 23, 30, 35, 52, 64, 68, 72, 73, 91` 后，`Houston13_7gt -> Houston18_OS` 的十种子正式结果如下：

| 配置 | OA | OA_known | Unknown | HOS |
|---|---:|---:|---:|---:|
| Baseline | 0.5830 ± 0.0334 | 0.5395 ± 0.0409 | 0.7030 ± 0.0355 | 0.6115 ± 0.0285 |
| OCCR (`adaptive_push_only`) | 0.5718 ± 0.0360 | 0.5076 ± 0.0467 | 0.7490 ± 0.0237 | 0.6163 ± 0.0359 |

这组结果说明：
- 当前 Houston 版本在平均 `HOS` 上取得了**小幅提升**；
- 未知类识别能力提升明显；
- 但已知类识别有所下降，且跨种子波动增大；
- 因此该结果可作为“方法具有跨数据集迁移潜力”的证据，但同时也明确指出：`OCCR` 在 Houston 上仍需进一步稳健化。

### 补充复核：`MAIN / UNKNOWN` 十种子结果

后续对统一机制线与 `Unknownness Head` 进行了重新十种子复核，结果如下：

| 配置 | OA | OA_known | Unknown | HOS |
|---|---:|---:|---:|---:|
| Baseline | 0.5830 ± 0.0334 | 0.5395 ± 0.0409 | 0.7030 ± 0.0355 | 0.6115 ± 0.0285 |
| OCCR (`adaptive_push_only`) | 0.5718 ± 0.0360 | 0.5076 ± 0.0467 | 0.7490 ± 0.0237 | 0.6163 ± 0.0359 |
| `WGDT_MAIN_Houston10` | 0.5809 ± 0.0236 | 0.6120 ± 0.0234 | 0.4952 ± 0.1161 | 0.5327 ± 0.0620 |
| `WGDT_UNKNOWN_Houston10` | 0.5784 ± 0.0256 | 0.6050 ± 0.0240 | 0.5048 ± 0.1318 | 0.5347 ± 0.0790 |

这进一步说明：

- 历史结果中，`adaptive_push_only` 仍是 Houston 上最可靠的十种子工作点；
- 但在当前代码版本下重新复现时，尚未稳定回到历史表中的 `HOS = 0.6163 ± 0.0359`；
- `soft_occr + Adaptive Pull Gating` 在当前版本下虽然提升了 `OA_known`，但显著破坏了 `Unknown`，不适合作为默认主线；
- `Unknownness Head` 同样未在 Houston 十种子上带来主线级收益；
- 因此，Houston 在现阶段更合适的表述应为：存在历史最优工作点 `adaptive_push_only`，但当前代码状态下仍需先完成严格复现。

从论文组织上，建议将其放在：
- 主结果之后的跨数据集补充验证；
- 或者放入“讨论 / 局限性 / 后续工作”部分，作为引出 `Adaptive Gating OCCR` 的实验依据。

---

## 第二阶段升级方案：独立 Unknownness Head

### 为什么需要从 `gamma-guided gate` 升级
当前候选统一机制线 `soft_occr + Adaptive Pull Gating` 已经证明：
- 它能在统一 `soft_occr` 框架内学习不同数据集的 `pull` 工作点；
- 但它的门控信号仍主要来自 `gamma` 及其派生统计量；
- 并且在最新十种子复核中尚未稳定优于现有性能主线；
- 因而它本质上仍属于 **`gamma-guided refinement`**，而不是已经定型的最终主线。

这意味着：
- 原始 WGDT 能稳定区分的样本，当前主线通常也能进一步修正；
- 原始 WGDT 难以区分的样本，当前主线虽然可做温和重加权，但上限仍受 `gamma` 质量约束；
- 因此，下一阶段若希望取得更明显的结构创新与性能突破，需要引入 **相对独立的 unknownness 信号**。

### 最小实现目标
在不推翻 WGDT 主体的前提下，新增一个轻量 `Unknownness Head`，用于输出目标样本的未知性分数 `u ∈ [0, 1]`：
- `u` 越大，说明样本越像未知类；
- `u` 越小，说明样本越像已知类；
- 该分数不直接替代最终分类，而是用于调节 `soft_occr` 中的 `pull/push` 权重。

### 输入、输出与作用方式
#### 1. 输入
最小实现不直接使用原始高维特征，而是输入一组低维、可解释的几何统计量：
- `min_distance`：样本到最近锚点的距离；
- `distance_margin = second_min_distance - min_distance`：最近锚点与次近锚点的竞争间隔；
- `min_gamma`：当前 WGDT 的开放集几何量；
- `entropy`：由 `softmin(distance)` 得到的类别不确定性。

记为：

```math
z = [d_{min},\; d_{2nd}-d_{min},\; \gamma_{min},\; H(p)]
```

#### 2. 输出
使用轻量 MLP 预测：

```math
u = \sigma(\mathrm{MLP}(z))
```

其中：
- `u≈0` 表示更像已知；
- `u≈1` 表示更像未知。

#### 3. 如何作用到 OCCR
将 `u` 作为 `soft_occr` 的样本级调制因子：

```math
w_{pull} = (1-u) \cdot w_{pull}^{soft}
```

```math
w_{push} = u \cdot w_{push}^{soft}
```

这样可以理解为：
- 当分支判断样本更像已知时，保留更多 `pull`；
- 当分支判断样本更像未知时，增强 `push`、抑制 `pull`。

### 判断依据与“独立性”边界
这里的“独立”并不是完全脱离 WGDT 主干，而是指：
- 不再只用单一 `gamma` 做判别；
- 而是联合利用 **距离、锚点竞争关系、分类不确定性、开放集几何量** 四类证据；
- 并通过单独的 head 与单独的损失来学习未知性分数。

因此，更严谨的表述应为：
> 该模块不是完全独立于主干的第二分类器，而是一个 **relative-independent unknownness head**。

它的主要价值在于：
- 避免简单把 `gamma` 再套一层 sigmoid；
- 让“是否更像未知”的判断来自多种证据的联合建模；
- 为后续更强的结构创新打基础。

### 训练监督设计（最小版）
考虑到目标域无标注，第一版不追求复杂监督，而采用弱监督：
- `source` 样本全部视为已知：监督 `u -> 0`；
- `target` 中高置信 `hard_unknown` 样本：监督 `u -> 1`；
- `ambiguous` 样本暂不监督；
- 第一版实现中，输入统计量默认 `detach`，优先验证结构有效性，而不直接扰动主干特征学习。

该设计的优点是：
- 对现有主线侵入小；
- 不会直接破坏当前 `soft_occr + Adaptive Pull Gating` 的稳定工作点；
- 能先验证“新增 unknownness 头是否真的提供了新信息”。

### 与当前主线的关系
第二阶段建议采用如下推进方式：
1. **当前默认主线保持不变**
   - `PaviaC_OS` 继续以保守 OCCR / `R4` 作为当前主线；
   - `Houston18_OS` 继续以“历史最佳的 `adaptive_push_only` 待复现线”作为当前优先排查对象；
   - 保证当前已验证的十种子主线结论不被候选升级线覆盖。
2. **独立 Unknownness Head 作为显式升级线**
   - 默认关闭；
   - 以最小可控实现接入代码；
   - 在新的十种子优势成立前，不升级为正式主线实验。
3. **评价标准**
   - 如果它只能复现 `gamma` 的结论，则说明新增 head 信息不足；
   - 如果它能在 `gamma` 边界样本上提供额外判别力，则说明结构创新有效；
   - 重点观察 `u` 的分布、`hard_unknown` 上的均值、以及 `pull/push` 权重重分配是否更合理。

### 实现口径（当前仓库最小版）
当前代码实现建议采用以下默认口径：
- 新增参数：`unknown_head`、`unknown_hidden_dim`、`unknown_detach_input`、`lambda_unknown`；
- 默认均保持保守设置，其中 `unknown_head=False`；
- 只有在显式开启后，才会将 `Unknownness Head` 接入 `soft_occr` 的样本级权重。

### 论文表述建议
若该分支后续验证有效，可在论文中表述为：

> We extend the original gamma-guided OCCR into a geometry-aware unknownness estimation framework, where a lightweight unknownness head jointly models anchor distance, anchor competition, gamma, and prediction uncertainty to modulate pull-push behavior at the sample level.

对应中文可写为：

> 我们将原始依赖 `gamma` 的 OCCR 机制，升级为面向几何统计与不确定性的未知性估计框架，通过轻量 `Unknownness Head` 在样本层面对 `pull/push` 进行自适应调制。
