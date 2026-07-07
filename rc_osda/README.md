# RC-OSDA

`RC-OSDA` 是当前论文主线算法的正式代码入口，面向论文实验与复现实验统一使用。

## 目录说明

- `train.py`：正式训练 / 测试入口。
- `models/`：正式方法模型入口，暴露 `Reliability-Calibrated Open-Set Domain Adaptation` 模型。
- `configs/dataset_tuned.json`：三个主实验数据集的 tuned 参数。
- `run_seed_sweep.py`：多 GPU 随机种子扫描入口。
- `results/`：建议保存该方法的汇总结果。

## 单次实验

```powershell
python rc_osda/train.py --source_dataset PaviaU_7gt --target_dataset PaviaC_OS --seed 82 --log_name RC_OSDA_Pavia_seed82
```

```powershell
python rc_osda/train.py --source_dataset Houston13_7gt --target_dataset Houston18_OS --seed 1 --log_name RC_OSDA_Houston_seed1
```

```powershell
python rc_osda/train.py --source_dataset HanChuan_4gt --target_dataset Honghu_OS --seed 70 --log_name RC_OSDA_Wuhan_seed70
```

## 多 GPU seed sweep

```powershell
python rc_osda/run_seed_sweep.py --datasets Pavia Houston Honghu --devices 0 1 2 3 --seed-start 0 --seed-end 99 --top-k 10 --log-prefix RC_OSDA_seed_sweep
```

该命令会使用当前激活环境的 `python`，不需要额外指定解释器。
