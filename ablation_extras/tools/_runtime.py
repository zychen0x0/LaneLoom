"""Shared helpers: build a LaneLoom model and validation loader for analysis."""

from datetime import datetime, timedelta
from pathlib import Path
import time

import torch
from mmengine.config import Config

from LaneLoom.engine import (
    ExperimentRunner,
    configure_test_experiment,
    register_all_modules,
    remove_runtime_config_artifacts,
)
from LaneLoom.engine.openlane_test_cases import build_openlane_test_configs

REPO_ROOT = Path(__file__).resolve().parents[2]
RETURN_ROOT = REPO_ROOT / 'ablation_extras' / 'return_back'


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def build_model_and_loader(config, checkpoint, analysis_name,
                           batch_size=None, case='validation'):
    """Return (model, dataloader) for one OpenLane split, weights loaded."""
    register_all_modules()
    config_path = Path(config).resolve()
    checkpoint = str(Path(checkpoint).resolve())
    base_cfg = Config.fromfile(config_path)
    [(_, cfg)] = build_openlane_test_configs(base_cfg, case)
    cfg.launcher = 'none'
    cfg.load_from = checkpoint
    if batch_size is not None:
        cfg.test_dataloader.batch_size = int(batch_size)
    checkpoint_path = Path(checkpoint)
    experiment_dir = (checkpoint_path.parent.parent
                      if checkpoint_path.parent.name == 'checkpoints'
                      else checkpoint_path.parent)
    analysis_dir = experiment_dir / f'analysis_{analysis_name}'
    configure_test_experiment(cfg, checkpoint, str(analysis_dir))
    runner = ExperimentRunner.from_cfg(cfg)
    remove_runtime_config_artifacts(analysis_dir, config_path)
    runner.load_checkpoint(checkpoint, map_location='cpu')
    model = runner.model
    model.eval()
    loader = runner.build_dataloader(cfg.test_dataloader)
    return model, loader


class Progress:
    """Print timestamped progress with an estimated finish time."""

    def __init__(self, total, label, every):
        self.total = max(int(total), 1)
        self.label = label
        self.every = max(int(every), 1)
        self.start = time.time()

    def update(self, done):
        if done % self.every and done != self.total:
            return
        elapsed = time.time() - self.start
        rate = elapsed / max(done, 1)
        remaining = rate * (self.total - done)
        finish = datetime.now() + timedelta(seconds=remaining)
        print(f'[{now()}] {self.label}: {done}/{self.total} '
              f'({100.0 * done / self.total:.1f}%), elapsed '
              f'{elapsed / 60:.1f} min, estimated finish '
              f'{finish.strftime("%H:%M")}', flush=True)


def gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device())
    return 'cpu'
