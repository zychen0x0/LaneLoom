"""Evaluate a LaneLoom checkpoint on ONCE-3DLanes validation data."""

import argparse
import json
from pathlib import Path

from mmengine.config import Config

from LaneLoom.engine import (
    ExperimentRunner,
    configure_test_experiment,
    register_all_modules,
    remove_runtime_config_artifacts,
)


register_all_modules()


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LaneLoom on ONCE.')
    parser.add_argument('config', help='Path to the ONCE MMEngine config.')
    parser.add_argument('checkpoint', help='Checkpoint to evaluate.')
    parser.add_argument(
        '--work-dir', default=None,
        help='Evaluation directory; defaults beside the checkpoint directory.')
    parser.add_argument(
        '--prediction-dir', default=None,
        help='Official-format prediction directory; defaults under work-dir.')
    parser.add_argument('--launcher', choices=('none', 'pytorch', 'slurm', 'mpi'),
                        default='none')
    return parser.parse_args()


def resolve_evaluation_dir(checkpoint, work_dir):
    if work_dir is not None:
        return Path(work_dir).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    experiment_dir = (
        checkpoint_path.parent.parent
        if checkpoint_path.parent.name == 'checkpoints'
        else checkpoint_path.parent)
    return experiment_dir / 'once_evaluation'


def write_summary(output_dir, checkpoint, metrics):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = (
        'F1', 'Precision', 'Recall', 'CDE', 'score_threshold',
        'best_F1', 'best_Precision', 'best_Recall', 'best_CDE',
        'best_score_threshold', 'num_gt', 'num_pred', 'num_tp')
    prefix = 'ONCEOfficial/'
    payload = {
        'checkpoint': str(Path(checkpoint).resolve()),
        'metrics': {
            name: float(metrics[prefix + name])
            for name in names if prefix + name in metrics
        },
    }
    (output_dir / 'once_evaluation_results.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    return payload


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint}')
    cfg = Config.fromfile(config_path)
    if str(cfg.get('dataset_name', '')).lower() != 'once':
        raise ValueError(
            f'{config_path} is not an ONCE config; use tools/test.py for OpenLane.')

    output_dir = resolve_evaluation_dir(checkpoint, args.work_dir)
    prediction_dir = (
        Path(args.prediction_dir).expanduser().resolve()
        if args.prediction_dir else output_dir / 'pred_dir')
    cfg.launcher = args.launcher
    cfg.load_from = str(checkpoint)
    cfg.test_evaluator.prediction_dir = str(prediction_dir)
    experiment_dir = configure_test_experiment(
        cfg, str(checkpoint), output_dir)
    runner = ExperimentRunner.from_cfg(cfg)
    remove_runtime_config_artifacts(experiment_dir, config_path)
    metrics = runner.test()
    if runner.rank != 0:
        return

    write_summary(output_dir, checkpoint, metrics)


if __name__ == '__main__':
    main()
