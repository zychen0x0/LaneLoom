"""Small MMEngine configuration helpers for reproducible experiment layouts."""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import tempfile
import time


def configure_train_experiment(cfg, work_root: str | None = None,
                               now: datetime | None = None) -> Path:
    """Create one dated LaneLoom experiment and point MMEngine at it.

    The caller invokes this before building ``Runner``.  MMEngine still owns
    checkpointing, TensorBoard writing, logging, and distributed execution;
    this helper only gives those standard components explicit directories.
    """
    root = Path(work_root or cfg.work_dir).expanduser().resolve()
    timestamp, is_primary = _shared_timestamp(root, now)
    experiment_dir = root / f'exp_date_{timestamp}'
    if is_primary and experiment_dir.exists():
        raise FileExistsError(
            f'Experiment directory already exists: {experiment_dir}. '
            'Wait one second or pass a different --work-dir.')
    return _configure_layout(cfg, experiment_dir, create=True)


def _shared_timestamp(root: Path, now: datetime | None) -> tuple[str, bool]:
    """Return one timestamp for every ``torchrun`` rank without old APIs."""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    if world_size <= 1:
        return (now or datetime.now()).strftime('%Y%m%d_%H%M%S'), True

    marker_dir = Path(tempfile.gettempdir()) / 'laneloom_torchrun'
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"timestamp_{os.environ.get('MASTER_PORT', 'default')}"
    if rank == 0:
        timestamp = (now or datetime.now()).strftime('%Y%m%d_%H%M%S')
        marker.write_text(timestamp, encoding='utf-8')
        return timestamp, True

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if marker.is_file():
            timestamp = marker.read_text(encoding='utf-8').strip()
            if timestamp:
                return timestamp, False
        time.sleep(0.05)
    raise TimeoutError(
        f'Rank {rank} timed out waiting for rank 0 to create {marker}.')


def configure_test_experiment(cfg, checkpoint: str,
                              work_root: str | None = None) -> Path:
    """Reuse the checkpoint's dated experiment directory for test artifacts."""
    if work_root is not None:
        experiment_dir = Path(work_root).expanduser().resolve()
    else:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        experiment_dir = (
            checkpoint_path.parent.parent
            if checkpoint_path.parent.name == 'checkpoints'
            else Path(cfg.work_dir).expanduser().resolve())
    return _configure_layout(cfg, experiment_dir, create=True)


def _configure_layout(cfg, experiment_dir: Path, create: bool) -> Path:
    logs_dir = experiment_dir / 'logs'
    tensorboard_dir = experiment_dir / 'Tensorboard'
    checkpoints_dir = experiment_dir / 'checkpoints'
    if create:
        for directory in (logs_dir, tensorboard_dir, checkpoints_dir):
            directory.mkdir(parents=True, exist_ok=True)

    cfg.work_dir = str(logs_dir)
    cfg.experiment_dir = str(experiment_dir)
    cfg.default_hooks = dict(cfg.default_hooks)
    checkpoint_cfg = dict(cfg.default_hooks['checkpoint'])
    checkpoint_cfg['type'] = 'LaneLoomCheckpointHook'
    checkpoint_cfg['out_dir'] = str(checkpoints_dir)
    cfg.default_hooks['checkpoint'] = checkpoint_cfg
    cfg.visualizer = dict(
        type='Visualizer',
        save_dir=str(experiment_dir),
        vis_backends=[
            dict(type='TensorboardVisBackend', save_dir=str(tensorboard_dir)),
        ])
    return experiment_dir


def remove_runtime_config_artifacts(experiment_dir: Path,
                                    config_path: str | Path) -> None:
    """Delete MMEngine's copied config files; the source config is canonical."""
    logs_dir = Path(experiment_dir) / 'logs'
    for path in (logs_dir / Path(config_path).name, logs_dir / 'config.py'):
        if path.is_file():
            path.unlink()
