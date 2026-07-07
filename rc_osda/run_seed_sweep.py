from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED_SWEEP_HELPER = "seed_sweep_topk.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RC-OSDA multi-GPU seed sweep")
    parser.add_argument("--datasets", nargs="+", default=["Pavia", "Houston", "Honghu"])
    parser.add_argument("--devices", nargs="+", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=99)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--log-prefix", default="RC_OSDA_seed_sweep")
    parser.add_argument("--summary-root", default="rc_osda/results/seed_sweep_reports")
    parser.add_argument("--skip-completed", choices=["True", "False"], default="True")
    parser.add_argument("--export-seed-config", default="rc_osda/results/top10_seed_config.json")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def normalize_extra_args(extra_args: list[str]) -> list[str]:
    if extra_args and extra_args[0] == "--":
        return extra_args[1:]
    return extra_args


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        str(ROOT / "rc_osda" / SEED_SWEEP_HELPER),
        "--python", sys.executable,
        "--runner", "rc_osda/train.py",
        "--variant-config", "rc_osda/configs/dataset_tuned.json",
        "--use-dataset-tuned", "True",
        "--datasets", *args.datasets,
        "--devices", *(str(device) for device in args.devices),
        "--seed-start", str(args.seed_start),
        "--seed-end", str(args.seed_end),
        "--top-k", str(args.top_k),
        "--log-prefix", args.log_prefix,
        "--summary-root", args.summary_root,
        "--skip-completed", args.skip_completed,
        "--export-seed-config", args.export_seed_config,
        *normalize_extra_args(args.extra_args),
    ]
    subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
