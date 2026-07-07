from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path


DATASET_PRESETS = {
    'Pavia': {
        'source': 'PaviaU_7gt',
        'target': 'PaviaC_OS',
        'suffix': 'Pavia',
    },
    'Houston': {
        'source': 'Houston13_7gt',
        'target': 'Houston18_OS',
        'suffix': 'Houston',
    },
    'HyRank': {
        'source': 'HyRank_source',
        'target': 'HyRank_target',
        'suffix': 'HyRank',
    },
    'Honghu': {
        'source': 'HanChuan_4gt',
        'target': 'Honghu_OS',
        'suffix': 'Honghu',
    },
}

PM_SIGN = chr(177)


@dataclass
class Task:
    dataset_key: str
    source_dataset: str
    target_dataset: str
    log_name: str
    seed: int


@dataclass
class RunningTask:
    task: Task
    gpu_id: int
    process: subprocess.Popen
    result_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Multi-GPU seed sweep for RC-OSDA tuned with top-k seed export and classwise summary'
    )
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--runner', default='rc_osda/train.py')
    parser.add_argument('--variant-config', default='rc_osda/configs/dataset_tuned.json')
    parser.add_argument('--use-dataset-tuned', choices=['True', 'False'], default='True')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_PRESETS.keys()), default=list(DATASET_PRESETS.keys()))
    parser.add_argument('--devices', nargs='+', type=int, required=True)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--seed-end', type=int, default=99)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--log-prefix', default='RC_OSDA_seed_sweep')
    parser.add_argument('--summary-root', default='rc_osda/results/seed_sweep_reports')
    parser.add_argument('--skip-completed', choices=['True', 'False'], default='True')
    parser.add_argument('--poll-seconds', type=float, default=15.0)
    parser.add_argument('--export-seed-config', default='')
    parser.add_argument('--extra-args', nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def normalize_extra_args(extra_args: list[str]) -> list[str]:
    if extra_args and extra_args[0] == '--':
        return extra_args[1:]
    return extra_args


def build_log_name(log_prefix: str, dataset_key: str) -> str:
    return f'{log_prefix}_{DATASET_PRESETS[dataset_key]["suffix"]}'


def result_path(root: Path, log_name: str, source_dataset: str, target_dataset: str, seed: int) -> Path:
    return root / 'logs' / log_name / f'{log_name} {source_dataset}-{target_dataset} seed={seed}.json'


def has_valid_result(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open('r', encoding='utf-8') as file:
            result = json.load(file)
    except Exception:
        return False
    return isinstance(result.get('hos'), (int, float))


def create_tasks(args: argparse.Namespace) -> deque[Task]:
    tasks: list[Task] = []
    for seed in range(args.seed_start, args.seed_end + 1):
        for dataset_key in args.datasets:
            preset = DATASET_PRESETS[dataset_key]
            log_name = build_log_name(args.log_prefix, dataset_key)
            tasks.append(
                Task(
                    dataset_key=dataset_key,
                    source_dataset=preset['source'],
                    target_dataset=preset['target'],
                    log_name=log_name,
                    seed=seed,
                )
            )
    return deque(tasks)


def launch_task(root: Path, args: argparse.Namespace, task: Task, gpu_id: int, extra_args: list[str]) -> RunningTask:
    output_path = result_path(root, task.log_name, task.source_dataset, task.target_dataset, task.seed)
    command = [
        args.python,
        str(root / args.runner),
        '--source_dataset', task.source_dataset,
        '--target_dataset', task.target_dataset,
        '--seed', str(task.seed),
        '--log_name', task.log_name,
        '--variant_config', args.variant_config,
        '--use_dataset_tuned', args.use_dataset_tuned,
        '--device', '0',
        *extra_args,
    ]
    env = os.environ.copy()
    env.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    env.setdefault('PYTHONUNBUFFERED', '1')
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f'[LAUNCH][GPU {gpu_id}] {task.dataset_key} seed={task.seed}')
    print('[CMD]', shlex.join(command))
    process = subprocess.Popen(command, cwd=str(root), env=env)
    return RunningTask(task=task, gpu_id=gpu_id, process=process, result_path=output_path)


def sort_results(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda item: (
            float(item['hos']),
            float(item['unknown']),
            float(item['oa_known']),
            -int(item['seed']),
        ),
        reverse=True,
    )


def trim_number(value: float, digits: int = 1) -> str:
    text = f'{value:.{digits}f}'
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return text


def format_percent_pm(mean_value: float | None, std_value: float | None, digits: int = 1) -> str:
    if mean_value is None:
        return '-'
    mean_pct = mean_value * 100.0
    std_pct = 0.0 if std_value is None else std_value * 100.0
    return f'{trim_number(mean_pct, digits)} {PM_SIGN} {trim_number(std_pct, digits)}'


def metric_stats(values: list[float]) -> dict:
    if not values:
        return {
            'mean': None,
            'std': None,
            'mean_percent_pm': '-',
        }
    mean_value = statistics.mean(values)
    std_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        'mean': mean_value,
        'std': std_value,
        'mean_percent_pm': format_percent_pm(mean_value, std_value),
    }


def normalize_classes_acc(values: object) -> list[float]:
    if not isinstance(values, list):
        return []
    normalized: list[float] = []
    for item in values:
        try:
            normalized.append(float(item))
        except Exception:
            return []
    return normalized


def infer_class_names(length: int) -> list[str]:
    if length <= 0:
        return []
    if length == 1:
        return ['unknown']
    return [*(f'class_{idx + 1}' for idx in range(length - 1)), 'unknown']


def summarize_classwise(top_rows: list[dict]) -> tuple[list[dict], int]:
    class_rows = [row for row in top_rows if row.get('classes_acc')]
    if not class_rows:
        return [], 0

    length_counter = Counter(len(row['classes_acc']) for row in class_rows)
    dominant_length, _ = length_counter.most_common(1)[0]
    filtered_rows = [row for row in class_rows if len(row['classes_acc']) == dominant_length]
    skipped_count = len(class_rows) - len(filtered_rows)
    class_names = infer_class_names(dominant_length)

    summaries: list[dict] = []
    for class_index, class_name in enumerate(class_names):
        values = [row['classes_acc'][class_index] for row in filtered_rows]
        best_seed_value = filtered_rows[0]['classes_acc'][class_index] if filtered_rows else None
        stats = metric_stats(values)
        summaries.append(
            {
                'class_index': class_index,
                'class_name': class_name,
                'mean': stats['mean'],
                'std': stats['std'],
                'mean_percent_pm': stats['mean_percent_pm'],
                'best_seed_value': best_seed_value,
            }
        )
    return summaries, skipped_count


def build_topk_seed_config(dataset_summaries: list[dict]) -> dict:
    config: dict = {}
    for item in dataset_summaries:
        config[item['dataset_key']] = {
            'source_dataset': item['source_dataset'],
            'target_dataset': item['target_dataset'],
            'seeds': item['top_k_sorted_seeds'],
        }
    return config


def format_classwise_inline(classwise_summary: list[dict]) -> str:
    if not classwise_summary:
        return '-'
    parts: list[str] = []
    for item in classwise_summary:
        parts.append(f"{item['class_name']}={item['mean_percent_pm']}%")
    return ', '.join(parts)


def summarize_dataset(
    root: Path,
    summary_dir: Path,
    dataset_key: str,
    log_name: str,
    source_dataset: str,
    target_dataset: str,
    seed_start: int,
    seed_end: int,
    top_k: int,
) -> dict:
    rows: list[dict] = []
    missing_seeds: list[int] = []

    for seed in range(seed_start, seed_end + 1):
        path = result_path(root, log_name, source_dataset, target_dataset, seed)
        if not has_valid_result(path):
            missing_seeds.append(seed)
            continue
        with path.open('r', encoding='utf-8') as file:
            result = json.load(file)
        rows.append(
            {
                'seed': seed,
                'oa': float(result['oa']),
                'aa': float(result['aa']),
                'oa_known': float(result['oa_known']),
                'aa_known': float(result['aa_known']),
                'unknown': float(result['unknown']),
                'hos': float(result['hos']),
                'classes_acc': normalize_classes_acc(result.get('classes_acc', [])),
                'path': str(path.relative_to(root)).replace('\\', '/'),
            }
        )

    sorted_rows = sort_results(rows)
    top_rows = sorted_rows[:top_k]
    top_ranked_seeds = [row['seed'] for row in top_rows]
    top_sorted_seeds = sorted(top_ranked_seeds)

    top_metrics = {
        'hos': metric_stats([row['hos'] for row in top_rows]),
        'unknown': metric_stats([row['unknown'] for row in top_rows]),
        'oa_known': metric_stats([row['oa_known'] for row in top_rows]),
        'aa_known': metric_stats([row['aa_known'] for row in top_rows]),
        'oa': metric_stats([row['oa'] for row in top_rows]),
        'aa': metric_stats([row['aa'] for row in top_rows]),
    }
    classwise_summary, classwise_skipped_count = summarize_classwise(top_rows)

    dataset_summary = {
        'dataset_key': dataset_key,
        'source_dataset': source_dataset,
        'target_dataset': target_dataset,
        'log_name': log_name,
        'seed_start': seed_start,
        'seed_end': seed_end,
        'successful_count': len(rows),
        'missing_or_failed_seeds': missing_seeds,
        'best_seed': top_rows[0]['seed'] if top_rows else None,
        'best_hos': top_rows[0]['hos'] if top_rows else None,
        'hos_mean': statistics.mean([row['hos'] for row in rows]) if rows else None,
        'hos_median': statistics.median([row['hos'] for row in rows]) if rows else None,
        'top_k_count': len(top_rows),
        'top_k_ranked_seeds': top_ranked_seeds,
        'top_k_sorted_seeds': top_sorted_seeds,
        'top_k_metrics': top_metrics,
        'top_k_classwise': classwise_summary,
        'top_k_classwise_skipped_count': classwise_skipped_count,
        'top_k': top_rows,
        'all_results_sorted': sorted_rows,
    }

    dataset_dir = summary_dir / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with (dataset_dir / 'topk_summary.json').open('w', encoding='utf-8') as file:
        json.dump(dataset_summary, file, indent=2, ensure_ascii=False)

    with (dataset_dir / 'topk_seeds.json').open('w', encoding='utf-8') as file:
        json.dump(
            {
                'dataset_key': dataset_key,
                'source_dataset': source_dataset,
                'target_dataset': target_dataset,
                'top_k_ranked_seeds': top_ranked_seeds,
                'top_k_sorted_seeds': top_sorted_seeds,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    with (dataset_dir / 'topk_summary.md').open('w', encoding='utf-8') as file:
        file.write(f'# {dataset_key} RC-OSDA tuned seed sweep summary\n\n')
        file.write(f'- source_dataset: `{source_dataset}`\n')
        file.write(f'- target_dataset: `{target_dataset}`\n')
        file.write(f'- log_name: `{log_name}`\n')
        file.write(f'- successful_count: `{len(rows)}`\n')
        file.write(f'- seed_range: `{seed_start}~{seed_end}`\n')
        file.write(f'- top_k: `{len(top_rows)}`\n')
        if rows:
            file.write(f'- best_seed: `{dataset_summary["best_seed"]}`\n')
            file.write(f'- best_hos: `{dataset_summary["best_hos"] * 100:.2f}%`\n')
            file.write(f'- hos_mean(all seeds): `{dataset_summary["hos_mean"] * 100:.2f}%`\n')
            file.write(f'- hos_median(all seeds): `{dataset_summary["hos_median"] * 100:.2f}%`\n')
        file.write(f'- top_k_ranked_seeds: `{top_ranked_seeds}`\n')
        file.write(f'- top_k_sorted_seeds: `{top_sorted_seeds}`\n')
        file.write(f'- top_k_hos(mean {PM_SIGN} std): `{top_metrics["hos"]["mean_percent_pm"]}%`\n')
        file.write(f'- top_k_unknown(mean {PM_SIGN} std): `{top_metrics["unknown"]["mean_percent_pm"]}%`\n')
        file.write(f'- top_k_oa_known(mean {PM_SIGN} std): `{top_metrics["oa_known"]["mean_percent_pm"]}%`\n')
        file.write(f'- top_k_aa_known(mean {PM_SIGN} std): `{top_metrics["aa_known"]["mean_percent_pm"]}%`\n')
        if missing_seeds:
            file.write(f'- missing_or_failed_seeds: `{missing_seeds}`\n')
        if classwise_skipped_count:
            file.write(f'- top_k_classwise_skipped_count: `{classwise_skipped_count}`\n')

        file.write('\n## Top K by HOS\n\n')
        file.write('| rank | seed | hos(%) | unknown(%) | oa_known(%) | aa_known(%) | oa(%) | aa(%) | result |\n')
        file.write('| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n')
        for rank, row in enumerate(top_rows, start=1):
            file.write(
                f'| {rank} | {row["seed"]} | {row["hos"]*100:.2f} | {row["unknown"]*100:.2f} | '
                f'{row["oa_known"]*100:.2f} | {row["aa_known"]*100:.2f} | {row["oa"]*100:.2f} | '
                f'{row["aa"]*100:.2f} | `{row["path"]}` |\n'
            )

        if classwise_summary:
            header_names = [item['class_name'] for item in classwise_summary]
            file.write('\n## Top K per-seed class accuracy\n\n')
            file.write('| seed | hos(%) | ' + ' | '.join(f'{name}(%)' for name in header_names) + ' |\n')
            file.write('| --- | --- | ' + ' | '.join('---' for _ in header_names) + ' |\n')
            for row in top_rows:
                values = row.get('classes_acc', [])
                if len(values) != len(header_names):
                    continue
                class_text = ' | '.join(f'{value * 100:.2f}' for value in values)
                file.write(f'| {row["seed"]} | {row["hos"]*100:.2f} | {class_text} |\n')

            file.write(f'\n## Top K classwise mean {PM_SIGN} std\n\n')
            file.write('| class | acc(mean ? std, %) | best_seed_acc(%) |\n')
            file.write('| --- | --- | --- |\n')
            for item in classwise_summary:
                best_seed_value = '-' if item['best_seed_value'] is None else f'{item["best_seed_value"] * 100:.2f}'
                file.write(
                    f'| {item["class_name"]} | {item["mean_percent_pm"]}% | {best_seed_value} |\n'
                )

    return dataset_summary


def write_suite_summary(summary_dir: Path, dataset_summaries: list[dict], top_k: int) -> None:
    summary_json = summary_dir / 'suite_summary.json'
    with summary_json.open('w', encoding='utf-8') as file:
        json.dump({'top_k': top_k, 'datasets': dataset_summaries}, file, indent=2, ensure_ascii=False)

    seed_config = build_topk_seed_config(dataset_summaries)
    with (summary_dir / 'topk_seeds_config.json').open('w', encoding='utf-8') as file:
        json.dump(seed_config, file, indent=2, ensure_ascii=False)

    summary_md = summary_dir / 'suite_summary.md'
    with summary_md.open('w', encoding='utf-8') as file:
        file.write('# RC-OSDA tuned multi-GPU seed sweep summary\n\n')
        file.write('| dataset | best_seed | best_hos(%) | top_k_hos(%) | top_k_unknown(%) | top_k_oa_known(%) | top_k_sorted_seeds | successful_count |\n')
        file.write('| --- | --- | --- | --- | --- | --- | --- | --- |\n')
        for item in dataset_summaries:
            best_seed = item['best_seed'] if item['best_seed'] is not None else '-'
            best_hos = '-' if item['best_hos'] is None else f'{item["best_hos"] * 100:.2f}'
            file.write(
                f'| {item["dataset_key"]} | {best_seed} | {best_hos} | '
                f'{item["top_k_metrics"]["hos"]["mean_percent_pm"]}% | '
                f'{item["top_k_metrics"]["unknown"]["mean_percent_pm"]}% | '
                f'{item["top_k_metrics"]["oa_known"]["mean_percent_pm"]}% | '
                f'{item["top_k_sorted_seeds"]} | {item["successful_count"]} |\n'
            )
        file.write('\n')
        for item in dataset_summaries:
            file.write(f'## {item["dataset_key"]}\n\n')
            file.write(f'- top-{top_k} seeds(sorted): `{item["top_k_sorted_seeds"]}`\n')
            file.write(f'- top-{top_k} HOS(mean {PM_SIGN} std): `{item["top_k_metrics"]["hos"]["mean_percent_pm"]}%`\n')
            file.write(f'- top-{top_k} Unknown(mean {PM_SIGN} std): `{item["top_k_metrics"]["unknown"]["mean_percent_pm"]}%`\n')
            file.write(f'- top-{top_k} OA_known(mean {PM_SIGN} std): `{item["top_k_metrics"]["oa_known"]["mean_percent_pm"]}%`\n')
            file.write(f'- detail: `{item["dataset_key"]}/topk_summary.md`\n\n')
            if item.get('top_k_classwise'):
                file.write('| class | acc(mean ? std, %) | best_seed_acc(%) |\n')
                file.write('| --- | --- | --- |\n')
                for class_item in item['top_k_classwise']:
                    best_seed_value = '-' if class_item['best_seed_value'] is None else f"{class_item['best_seed_value'] * 100:.2f}"
                    file.write(
                        f"| {class_item['class_name']} | {class_item['mean_percent_pm']}% | {best_seed_value} |\n"
                    )
                file.write('\n')


def export_seed_config(root: Path, dataset_summaries: list[dict], export_path: str) -> None:
    seed_config = build_topk_seed_config(dataset_summaries)
    if not export_path:
        return
    path = Path(export_path)
    if not path.is_absolute():
        path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        json.dump(seed_config, file, indent=2, ensure_ascii=False)
    print(f'[EXPORT] top-k seed config saved to {path}')


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    extra_args = normalize_extra_args(args.extra_args)
    tasks = create_tasks(args)

    available_gpus = deque(args.devices)
    running: list[RunningTask] = []
    skipped = 0
    failed_tasks: list[tuple[str, int, int]] = []

    while tasks or running:
        while tasks and available_gpus:
            task = tasks.popleft()
            output_path = result_path(root, task.log_name, task.source_dataset, task.target_dataset, task.seed)
            if args.skip_completed == 'True' and has_valid_result(output_path):
                skipped += 1
                print(f'[SKIP] {task.dataset_key} seed={task.seed} -> {output_path}')
                continue
            gpu_id = available_gpus.popleft()
            running.append(launch_task(root, args, task, gpu_id, extra_args))

        if not running:
            continue

        time.sleep(max(args.poll_seconds, 0.5))
        next_running: list[RunningTask] = []
        for item in running:
            return_code = item.process.poll()
            if return_code is None:
                next_running.append(item)
                continue

            available_gpus.append(item.gpu_id)
            if return_code != 0 or not has_valid_result(item.result_path):
                failed_tasks.append((item.task.dataset_key, item.task.seed, return_code))
                print(f'[FAIL][GPU {item.gpu_id}] {item.task.dataset_key} seed={item.task.seed} returncode={return_code}')
            else:
                print(f'[DONE][GPU {item.gpu_id}] {item.task.dataset_key} seed={item.task.seed}')
        running = next_running

    summary_dir = root / args.summary_root / args.log_prefix
    summary_dir.mkdir(parents=True, exist_ok=True)

    dataset_summaries: list[dict] = []
    for dataset_key in args.datasets:
        preset = DATASET_PRESETS[dataset_key]
        log_name = build_log_name(args.log_prefix, dataset_key)
        dataset_summaries.append(
            summarize_dataset(
                root=root,
                summary_dir=summary_dir,
                dataset_key=dataset_key,
                log_name=log_name,
                source_dataset=preset['source'],
                target_dataset=preset['target'],
                seed_start=args.seed_start,
                seed_end=args.seed_end,
                top_k=args.top_k,
            )
        )

    write_suite_summary(summary_dir, dataset_summaries, args.top_k)
    export_seed_config(root, dataset_summaries, args.export_seed_config)

    for item in dataset_summaries:
        print(
            f"[RESULT] {item['dataset_key']}: top_k_sorted_seeds={item['top_k_sorted_seeds']}, "
            f"HOS_topk={item['top_k_metrics']['hos']['mean_percent_pm']}%, "
            f"Unknown_topk={item['top_k_metrics']['unknown']['mean_percent_pm']}%, "
            f"OA_known_topk={item['top_k_metrics']['oa_known']['mean_percent_pm']}%"
        )
        if item.get('top_k_classwise'):
            print(f"[CLASSWISE] {item['dataset_key']}: {format_classwise_inline(item['top_k_classwise'])}")

    print(f'[SUMMARY] saved to {summary_dir}')
    print(f'[SUMMARY] skipped_completed={skipped}')
    if failed_tasks:
        print('[SUMMARY] failed_tasks=')
        for dataset_key, seed, return_code in failed_tasks:
            print(f'  - {dataset_key} seed={seed} returncode={return_code}')


if __name__ == '__main__':
    main()
