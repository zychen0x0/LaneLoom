"""Evaluate a LaneFormer checkpoint with MMEngine's standard TestLoop."""
import argparse
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner

import LaneFormer  # noqa: F401 - register project components.
from LaneFormer.engine import (
    build_openlane_test_configs,
    configure_test_experiment,
    remove_runtime_config_artifacts,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LaneFormer.')
    parser.add_argument('config', help='Path to the MMEngine config.')
    parser.add_argument('checkpoint', help='Checkpoint to evaluate.')
    parser.add_argument(
        '--work-dir', default=None,
        help='Experiment directory for test logs; defaults to checkpoint parent.')
    parser.add_argument('--launcher', choices=('none', 'pytorch', 'slurm', 'mpi'),
                        default='none')
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    base_cfg = Config.fromfile(config_path)
    checkpoint = str(Path(args.checkpoint).resolve())
    base_work_dir = (
        Path(args.work_dir).resolve()
        if args.work_dir else Path(checkpoint).parent.parent / 'test_cases'
    )
    for case_name, cfg in build_openlane_test_configs(base_cfg):
        print(f'\n===== OpenLane evaluation: {case_name} =====')
        cfg.launcher = args.launcher
        cfg.load_from = checkpoint
        experiment_dir = configure_test_experiment(
            cfg, checkpoint, base_work_dir / case_name)
        runner = Runner.from_cfg(cfg)
        remove_runtime_config_artifacts(experiment_dir, config_path)
        runner.test()


if __name__ == '__main__':
    main()
