from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dataLoader import getDataLoader
from utils.utils import getDatasetInfo, getDevice, mergeArgs, seed_torch


DEFAULT_VARIANT_CONFIG = "rc_osda/configs/dataset_tuned.json"
TUNED_KEYS = (
    "ova_loss_weight",
    "ova_threshold",
    "open_entropy_weight",
    "consistency_threshold",
    "consistency_loss_weight",
    "feature_noise_std",
    "density_quantile",
    "rescue_margin",
    "variance_floor",
    "teacher_momentum",
)


def get_variant_tuned_defaults(args: argparse.Namespace) -> dict:
    if args.use_dataset_tuned != "True":
        return {}

    config_path = Path(args.variant_config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"RC-OSDA tuned 配置文件不存在：{config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    dataset_override = config.get(args.target_dataset)
    if dataset_override is None:
        return {}

    return {
        key: dataset_override[key]
        for key in TUNED_KEYS
        if key in dataset_override
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RC-OSDA: reliability-calibrated open-set hyperspectral domain adaptation"
    )
    parser.add_argument("--log_name", type=str, default="RC_OSDA")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--train_num", type=int, default=180)
    parser.add_argument("--few_train_num", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--patch", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--source_dataset", choices=["Houston13_7gt", "PaviaU_7gt", "HyRank_source", "Yancheng_ZY", "HanChuan_4gt"], default="PaviaU_7gt")
    parser.add_argument("--target_dataset", choices=["Houston18_OS", "PaviaC_OS", "HyRank_target", "Yancheng_GF", "Honghu_OS"], default="PaviaC_OS")
    parser.add_argument("--anchor_type", type=str, choices=["anchor"], default="anchor")
    parser.add_argument("--radius_loss_type", type=str, choices=["MarginMSELoss"], default="MarginMSELoss")
    parser.add_argument("--radius_init", type=float, default=0.0)
    parser.add_argument("--pre_train", type=str, default="True")
    parser.add_argument("--pre_train_epochs", type=int, default=20)
    parser.add_argument("--draw", type=str, default="False")

    parser.add_argument("--ova_loss_weight", type=float, default=1.0)
    parser.add_argument("--open_entropy_weight", type=float, default=0.1)
    parser.add_argument("--ova_threshold", type=float, default=0.5)
    parser.add_argument("--consistency_loss_weight", type=float, default=0.1)
    parser.add_argument("--consistency_threshold", type=float, default=0.7)
    parser.add_argument("--feature_noise_std", type=float, default=0.05)
    parser.add_argument("--density_quantile", type=float, default=0.05)
    parser.add_argument("--rescue_margin", type=float, default=0.15)
    parser.add_argument("--variance_floor", type=float, default=1e-4)
    parser.add_argument("--teacher_momentum", type=float, default=0.999)
    parser.add_argument("--variant_config", type=str, default=DEFAULT_VARIANT_CONFIG)
    parser.add_argument("--use_dataset_tuned", choices=["True", "False"], default="True")

    pre_args, _ = parser.parse_known_args()
    tuned_defaults = get_variant_tuned_defaults(pre_args)
    if tuned_defaults:
        parser.set_defaults(**tuned_defaults)

    args = parser.parse_args()
    mergeArgs(args, args.target_dataset)
    return args


def get_model(args, source_info, target_info):
    from rc_osda.models.reliability_calibrated_osda import Model

    model_args = {
        "args": args,
        "source_info": source_info,
        "target_info": target_info,
        "device": getDevice(args.device),
        "in_channels": args.pca if hasattr(args, "pca") and args.pca > 0 else source_info.bands_num,
        "patch": args.patch,
        "known_num_classes": len(args.source_known_classes),
        "anchor_type": args.anchor_type,
        "radius_loss_type": args.radius_loss_type,
        "radius_init": args.radius_init,
        "radius_margin": args.radius_margin,
        "alpha": args.alpha,
        "domain_loss_weight": args.domain_loss_weight,
        "radius_loss_weight": args.radius_loss_weight,
        "ova_loss_weight": args.ova_loss_weight,
        "open_entropy_weight": args.open_entropy_weight,
        "ova_threshold": args.ova_threshold,
        "consistency_loss_weight": args.consistency_loss_weight,
        "consistency_threshold": args.consistency_threshold,
        "feature_noise_std": args.feature_noise_std,
        "density_quantile": args.density_quantile,
        "rescue_margin": args.rescue_margin,
        "variance_floor": args.variance_floor,
        "teacher_momentum": args.teacher_momentum,
    }
    return Model(**model_args)


def run_model(model, data_loader: dict):
    from utils.dataLoader import CombinedLoader
    from utils.Trainer import Trainer

    trainer = Trainer(model, model.device)
    if model.args.pre_train == "True":
        trainer.train("pre_train", data_loader["source"]["train"], model.args.pre_train_epochs)
    trainer.train("train", CombinedLoader([data_loader["source"]["train"], data_loader["target"]["train"]]), model.args.epochs)
    model.calibrate_density_statistics(data_loader["source"]["train"])
    trainer.test("test", data_loader["target"]["test"])


def main() -> None:
    args = parse_args()
    seed_torch(args.seed)
    source_info = getDatasetInfo(args.source_dataset)
    target_info = getDatasetInfo(args.target_dataset)
    data_loader = getDataLoader(args, source_info, target_info, drop_last=True)
    model = get_model(args, source_info, target_info)
    run_model(model, data_loader)


if __name__ == "__main__":
    main()
