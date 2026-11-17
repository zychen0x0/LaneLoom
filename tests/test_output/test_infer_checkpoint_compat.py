"""Regression coverage for legacy checkpoint configs at inference call sites."""
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from mmengine.config import Config

from LaneLoom.datasets.openlane.dataset import OpenLaneDataset
from LaneLoom.models.backbones.dynamic_pillar_bev import DynamicPillarBEVEncoder
from LaneLoom.vis.output import infer_openlane_sample as infer


@pytest.fixture
def legacy_cfg(tmp_path):
    image_root = tmp_path / 'images'
    label_root = tmp_path / 'labels'
    label = label_root / 'validation/scene/frame.json'
    label.parent.mkdir(parents=True)
    label.write_text('{}')
    return Config(dict(
        dataset_cfg=dict(data_image_dir=str(image_root), data_label_dir=str(label_root)),
        val_dataloader=dict(dataset=dict(
            type='OpenLaneDataset', dataset_cfg=dict(train_pipeline=[], val_pipeline=[]),
            frames=(-10, -5, 0), history_num_frames=2,
            history_distance_m=10.0, history_distance_step_m=5.0,
            history_fallback_frame_offsets=(10, 5), temporal_gt_augmentation=True,
            smooth_3d_gt=True, gt_outlier_residual=0.75, modality={},
        )),
        model=dict(pts_bev_encoder=dict(
            type='DynamicPillarBEVEncoder', point_cloud_range=[-25, 0, -5, 25, 100, 5],
            pillar_size=[0.1, 0.1], queue_length=3,
        )),
    ))


def test_legacy_dataset_build(legacy_cfg, monkeypatch):
    label = Path(legacy_cfg.dataset_cfg.data_label_dir) / 'validation/scene/frame.json'

    class Sample:
        label_files = [label]

        def __getitem__(self, index):
            return {'index': index}

    def build(config):
        args = {k: v for k, v in config.items() if k != 'type'}
        inspect.signature(OpenLaneDataset).bind(**args)
        assert args['temporal_sampling_distance'] == (10.0, 5.0)
        return Sample()

    monkeypatch.setattr(infer.DATASETS, 'build', build)
    monkeypatch.setattr(infer, 'openlane_collate', lambda samples: samples[0])
    image = Path(legacy_cfg.dataset_cfg.data_image_dir) / 'validation/scene/frame.jpg'
    assert infer._build_sample(legacy_cfg, image) == {'index': 0}
    assert legacy_cfg.val_dataloader.dataset.frames == (-10, -5, 0)


def test_legacy_encoder_build(legacy_cfg, monkeypatch):
    model = SimpleNamespace(to=lambda device: SimpleNamespace(eval=lambda: 'model'))

    def build(config):
        encoder = config['pts_bev_encoder']
        inspect.signature(DynamicPillarBEVEncoder).bind(
            **{k: v for k, v in encoder.items() if k != 'type'})
        assert encoder['temporal_sampling_distance'] == (10.0, 5.0)
        assert 'queue_length' not in encoder
        return model

    monkeypatch.setattr(infer.MODELS, 'build', build)
    monkeypatch.setattr(infer, 'load_checkpoint', lambda *args, **kwargs: None)
    assert infer._build_model(legacy_cfg, Path('checkpoint.pth'), 'cpu') == 'model'
    assert legacy_cfg.model.pts_bev_encoder.queue_length == 3
