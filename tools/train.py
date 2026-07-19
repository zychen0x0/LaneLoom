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
    parser.add_argument(
        '--load-from', default=None,
        help='Warm-start model weights from a checkpoint with a new optimizer.')
    parser.add_argument(
        '--max-epochs', type=int, default=None,
        help='Override train_cfg.max_epochs for a short diagnostic run.')
    parser.add_argument('--launcher', choices=('none', 'pytorch', 'slurm', 'mpi'),
                        default='none')
    return parser.parse_args()


def apply_runtime_overrides(cfg, args):
    if args.resume and args.load_from:
        raise ValueError('--resume and --load-from cannot be used together.')
    cfg.resume = bool(args.resume)
    if args.load_from:
        checkpoint = Path(args.load_from).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint}')
        cfg.load_from = str(checkpoint)
    if args.max_epochs is not None:
        if args.max_epochs <= 0:
            raise ValueError('--max-epochs must be positive.')
        cfg.train_cfg.max_epochs = int(args.max_epochs)


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = Config.fromfile(config_path)
    cfg.launcher = args.launcher
    apply_runtime_overrides(cfg, args)
    experiment_dir = configure_train_experiment(cfg, args.work_dir)
    runner = LaneFormerRunner.from_cfg(cfg)
    remove_runtime_config_artifacts(experiment_dir, config_path)
    runner.logger.info('LaneFormer experiment directory: %s', experiment_dir)
    runner.train()


if __name__ == '__main__':
    main()
