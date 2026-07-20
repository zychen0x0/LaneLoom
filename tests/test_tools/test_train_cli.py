from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from mmengine.config import Config

TRAIN_PATH = Path(__file__).resolve().parents[2] / 'tools/train.py'
SPEC = spec_from_file_location('laneloom_train_cli', TRAIN_PATH)
TRAIN_MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_MODULE)
apply_runtime_overrides = TRAIN_MODULE.apply_runtime_overrides


def test_train_runtime_overrides_warm_start_with_fresh_optimizer(tmp_path):
    checkpoint = tmp_path / 'epoch_3.pth'
    checkpoint.touch()
    cfg = Config(dict(
        train_cfg=dict(type='EpochBasedTrainLoop', max_epochs=24),
        load_from=None,
        resume=False))
    args = Namespace(
        load_from=str(checkpoint), max_epochs=1, resume=False)

    apply_runtime_overrides(cfg, args)

    assert cfg.load_from == str(checkpoint.resolve())
    assert cfg.resume is False
    assert cfg.train_cfg.max_epochs == 1


def test_train_runtime_overrides_reject_resume_and_load_from(tmp_path):
    checkpoint = tmp_path / 'epoch_3.pth'
    checkpoint.touch()
    cfg = Config(dict(
        train_cfg=dict(type='EpochBasedTrainLoop', max_epochs=24)))
    args = Namespace(
        load_from=str(checkpoint), max_epochs=None, resume=True)

    with pytest.raises(ValueError, match='resume.*load-from'):
        apply_runtime_overrides(cfg, args)
