# WGDT++ 架构改造计划（以提升 HOS 为第一目标）

## 摘要
- 目标：在尽量保留现有训练骨架的前提下，对 `WGDT` 做一次“中等重构”，把原始的“浅层编码器 + 固定锚点 + 单全局半径 + 全样本DANN”升级为“多尺度谱空编码 + 原型开放集头 + 选择性域对齐 + 类自适应边界”。
- 预期收益：优先提升开放集域适应中的 `HOS` 与未知类识别，同时尽量不牺牲已知类 `OA/AA`。

## 已落地改造
- 主干编码器已由旧 `DCRN` 替换为多尺度谱空融合编码器 `MSSFEncoder`。
- 开放集头已由固定锚点替换为 `AdaptivePrototypeOpenHead`，包含 EMA 原型、类自适应半径、相似度阈值与未知分数阈值。
- 域对齐已改为选择性已知对齐，仅对高置信伪已知目标样本进行 DANN 与原型对比对齐。
- 训练流程已改为三阶段课程学习：`warmup / adapt / calibrate`。

## 当前实现要点
- `model/DCRN.py`
  - 引入谱分支金字塔卷积、空间分支残差空洞卷积、交叉门控融合。
  - 输出统一映射到 `embed_dim=256`。
- `model/OpenHead.py`
  - 维护按类原型 `prototype[c]`。
  - 使用混合未知分数：`0.5*(1-max_prob) + 0.3*min_proto_dist + 0.2*entropy`。
  - 推理同时参考 `radius / sim_threshold / unk_threshold`。
- `model/WGDT.py`
  - 保留 `get_model / run_model / parse_args` 入口。
  - 阶段 A：源域监督 + 原型紧致/分离。
  - 阶段 B：选择性 DANN + 原型对比对齐 + 灰区一致性约束。
  - 阶段 C：边界校准，重点优化能量阈值与类半径。

## 新增主要参数
- `--encoder_variant mssf|legacy`
- `--embed_dim 256`
- `--open_head_variant apo|anchor`
- `--align_variant selective|all`
- `--use_calibration True|False`
- `--proto_momentum 0.99`
- `--pseudo_known_tau 0.85`
- `--pseudo_known_unk_tau 0.35`
- `--pseudo_unknown_tau 0.60`
- `--warmup_epochs 20`
- `--adapt_epochs 80`
- `--calibrate_epochs 20`
- `--loss_proto 0.5`
- `--loss_adv 0.2`
- `--loss_contrast 0.3`
- `--loss_energy 0.2`
- `--loss_radius 0.2`
- `--loss_consistency 0.1`

## 建议实验
- 四组任务统一做 `3 seeds` 重复实验：
  - `PaviaU→PaviaC`
  - `Houston13→Houston18`
  - `HyRank_source→target`
  - `Yancheng_ZY→GF`
- 指标报告：`OA / AA / OA_known / Unknown / HOS`。
- 消融顺序建议：
  - 原始 `WGDT`
  - `+ MSSFEncoder`
  - `+ APOH`
  - `+ Selective Alignment`
  - `+ Calibration`

## 当前消融开关对应关系
- 关闭 `MSSFEncoder`：`--encoder_variant legacy`
- 关闭 `APOH`：`--open_head_variant anchor`
- 关闭 `Selective Alignment`：`--align_variant all`
- 关闭 `Calibration`：`--use_calibration False`
