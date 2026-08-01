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
    distance_nms_metric_prefix,
    normalise_distance_nms_thresholds,
    normalise_score_thresholds,
    score_threshold_metric_prefix,
    select_recall_safe_distance_threshold,
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
    parser.add_argument(
        '--distance-nms-thresholds', type=float, nargs='+', default=None,
        help=('Evaluate post-Tube mean 3D distance NMS thresholds in one '
              'inference pass, for example: '
              '--distance-nms-thresholds 0.2 0.25 0.3 0.35.'))
    parser.add_argument('--launcher', choices=('none', 'pytorch', 'slurm', 'mpi'),
                        default='none')
    return parser.parse_args()


def main():
    args = parse_args()
    score_thresholds = normalise_score_thresholds(args.score_thresholds)
    distance_nms_thresholds = normalise_distance_nms_thresholds(
        args.distance_nms_thresholds)
    config_path = Path(args.config).resolve()
    base_cfg = Config.fromfile(config_path)
    checkpoint = str(Path(args.checkpoint).resolve())
    base_work_dir = (
        Path(args.work_dir).resolve()
        if args.work_dir else Path(checkpoint).parent.parent / 'test_cases'
    )
    distance_case_results = {}
    last_rank = 0
    for case_name, cfg in build_openlane_test_configs(
            base_cfg, args.case, score_thresholds=score_thresholds,
            distance_nms_thresholds=distance_nms_thresholds):
        print(f'\n===== OpenLane evaluation: {case_name} =====')
        cfg.launcher = args.launcher
        cfg.load_from = checkpoint
        experiment_dir = configure_test_experiment(
            cfg, checkpoint, base_work_dir / case_name)
        runner = ExperimentRunner.from_cfg(cfg)
        remove_runtime_config_artifacts(experiment_dir, config_path)
        metrics = runner.test()
        last_rank = runner.rank
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
        if runner.rank == 0 and distance_nms_thresholds is not None:
            baseline_prefix = cfg.test_evaluator[0].prefix
            metric_names = (
                'F1', 'Recall', 'Precision', 'num_pred',
                'pred_per_frame', 'num_tp')

            def values_for(prefix):
                return {
                    name: metrics[f'{prefix}/{name}']
                    for name in metric_names
                }

            baseline = values_for(baseline_prefix)
            case_results = {None: baseline}
            print(f'\n===== Distance NMS summary: {case_name} =====')
            print(
                f'baseline: F1={baseline["F1"]:.8f} '
                f'Recall={baseline["Recall"]:.8f} '
                f'Precision={baseline["Precision"]:.8f} '
                f'Pred={baseline["num_pred"]:.0f} '
                f'Pred/frame={baseline["pred_per_frame"]:.4f} '
                f'TP={baseline["num_tp"]:.0f}')
            metric_prefix = baseline_prefix.rsplit('/', 1)[0]
            for threshold in distance_nms_thresholds:
                prefix = distance_nms_metric_prefix(
                    case_name, threshold, metric_prefix)
                values = values_for(prefix)
                case_results[threshold] = values
                print(
                    f'{threshold:g}: F1={values["F1"]:.8f} '
                    f'Recall={values["Recall"]:.8f} '
                    f'Precision={values["Precision"]:.8f} '
                    f'Pred={values["num_pred"]:.0f} '
                    f'Pred/frame={values["pred_per_frame"]:.4f} '
                    f'TP={values["num_tp"]:.0f} '
                    f'RecallDelta={values["Recall"] - baseline["Recall"]:+.8f}')
            distance_case_results[case_name] = case_results

    if last_rank == 0 and distance_nms_thresholds is not None:
        selected = select_recall_safe_distance_threshold(
            distance_case_results)
        print('\n===== Recall-safe distance NMS selection =====')
        if selected is None:
            print('No threshold preserves baseline recall on every case.')
        else:
            print(
                f'Selected {selected:g} m: highest validation F1 among '
                'thresholds preserving baseline recall on every case.')


if __name__ == '__main__':
    main()
