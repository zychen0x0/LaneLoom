"""Train LaneFormer with the standard MMEngine Runner."""
import argparse
from pathlib import Path

from mmengine.config import Config

import LaneFormer  # noqa: F401 - register models, datasets, and hooks.
from LaneFormer.engine import (
    LaneFormerRunner,
    configure_train_experiment,
    remove_runtime_config_artifacts,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train LaneFormer with MMEngine.')
    parser.add_argument('config', help='Path to LaneFormer MMEngine config.')
    parser.add_argument(
        '--work-dir', default=None,
        help='Parent directory for a new exp_date_YYYYMMDD_HHMMSS run.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--launcher', choices=('none', 'pytorch', 'slurm', 'mpi'),
                        default='none')
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = Config.fromfile(config_path)
    cfg.launcher = args.launcher
    experiment_dir = configure_train_experiment(cfg, args.work_dir)
    cfg.resume = args.resume
    runner = LaneFormerRunner.from_cfg(cfg)
    remove_runtime_config_artifacts(experiment_dir, config_path)
    runner.logger.info('LaneFormer experiment directory: %s', experiment_dir)
    runner.train()


if __name__ == '__main__':
    main()
