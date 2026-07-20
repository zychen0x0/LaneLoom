"""Evaluate a LaneLoom checkpoint with MMEngine's standard TestLoop."""
import argparse
from pathlib import Path

from mmengine.config import Config

import LaneLoom  # noqa: F401 - register project components.
from LaneLoom.engine import (
    ExperimentRunner,
    build_openlane_test_configs,
    configure_test_experiment,
    remove_runtime_config_artifacts,
)
from LaneLoom.engine.openlane_test_cases import (
    OPENLANE_TEST_CASE_CHOICES,
    normalise_score_thresholds,
    score_threshold_metric_prefix,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LaneLoom.')
    parser.add_argument('config', help='Path to the MMEngine config.')
    parser.add_argument('checkpoint', help='Checkpoint to evaluate.')
    parser.add_argument(
        '--work-dir', default=None,
        help='Experiment directory for test logs; defaults to checkpoint parent.')
    parser.add_argument(
        '--case', choices=OPENLANE_TEST_CASE_CHOICES, default='all',
        help=('Evaluation selection: all runs validation plus every official '
              'test case; validation runs only validation; test runs only '
              'the official test cases; a case name runs that case alone.'))
    parser.add_argument(
        '--score-thresholds', type=float, nargs='+', default=None,
        help=('Evaluate several objectness thresholds in one inference pass, '
              'for example: --score-thresholds 0.3 0.5 0.6 0.7 0.8.'))
    parser.add_argument('--launcher', choices=('none', 'pytorch', 'slurm', 'mpi'),
                        default='none')
    return parser.parse_args()


def main():
    args = parse_args()
    score_thresholds = normalise_score_thresholds(args.score_thresholds)
    config_path = Path(args.config).resolve()
    base_cfg = Config.fromfile(config_path)
    checkpoint = str(Path(args.checkpoint).resolve())
    base_work_dir = (
        Path(args.work_dir).resolve()
        if args.work_dir else Path(checkpoint).parent.parent / 'test_cases'
    )
    for case_name, cfg in build_openlane_test_configs(
            base_cfg, args.case, score_thresholds=score_thresholds):
        print(f'\n===== OpenLane evaluation: {case_name} =====')
        cfg.launcher = args.launcher
        cfg.load_from = checkpoint
        experiment_dir = configure_test_experiment(
            cfg, checkpoint, base_work_dir / case_name)
        runner = ExperimentRunner.from_cfg(cfg)
        remove_runtime_config_artifacts(experiment_dir, config_path)
        metrics = runner.test()
        if runner.rank == 0 and score_thresholds is not None:
            print(f'\n===== Score threshold summary: {case_name} =====')
            for threshold in score_thresholds:
                prefix = score_threshold_metric_prefix(case_name, threshold)
                metric_names = (
                    'F1', 'Recall', 'Precision', 'num_pred', 'num_tp')
                values = {
                    name: metrics[f'{prefix}/{name}']
                    for name in metric_names
                }
                print(
                    f'{threshold:g}: F1={values["F1"]:.8f} '
                    f'Recall={values["Recall"]:.8f} '
                    f'Precision={values["Precision"]:.8f} '
                    f'Pred={values["num_pred"]:.0f} '
                    f'TP={values["num_tp"]:.0f}')


if __name__ == '__main__':
    main()
