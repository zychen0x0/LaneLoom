"""A head comparison must share a training-only encoder and freeze its buffers."""
import hashlib
from pathlib import Path

import pytest
import torch
from mmengine.config import Config
from torch import nn

from LaneLoom.engine import register_all_modules
from LaneLoom.tools.train_q0_prior import Q0Pretrainer, load_training_encoder
from LaneLoom.tools.evaluate_q0_prior import load_model


def source_checkpoint(path, split='train', train_split='training'):
    encoder = nn.Linear(3, 2)
    cfg = Config(dict(train_dataloader=dict(dataset=dict(type='OpenLaneDataset', split=split,
                      dataset_cfg=dict(train_split=train_split)))))
    torch.save(dict(meta=dict(cfg=cfg.pretty_text, epoch=43),
                    state_dict={**{'pts_bev_encoder.'+k:v for k,v in encoder.state_dict().items()},
                                'lane_head.unused':torch.tensor(999.)}), path)
    return encoder


def test_only_encoder_weights_are_imported_and_the_source_is_fingerprinted(tmp_path):
    path = tmp_path/'source.pth'
    source = source_checkpoint(path)
    target = nn.Linear(3, 2)
    info = load_training_encoder(target, path)
    for key,value in target.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[key], rtol=0, atol=0)
    assert info['encoder_source_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert info['encoder_source_epoch'] == 43


def test_sparse_encoder_import_preserves_version_metadata_for_kernel_layout(tmp_path):
    register_all_modules()
    # Use the same MMDetection3D sparse-loader override as model construction.
    from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
    import spconv.pytorch as spconv
    assert IS_SPCONV2_AVAILABLE
    source = nn.Sequential(spconv.SubMConv2d(4, 8, 3, bias=False))
    holder = nn.Module()
    holder.pts_bev_encoder = source
    weights = holder.state_dict()
    assert weights._metadata['pts_bev_encoder.0']['version'] == 2
    cfg = Config(dict(train_dataloader=dict(dataset=dict(type='OpenLaneDataset', split='train',
                      dataset_cfg=dict(train_split='training')))))
    path = tmp_path/'sparse.pth'
    torch.save(dict(meta=dict(cfg=cfg.pretty_text), state_dict=weights), path)
    target = nn.Sequential(spconv.SubMConv2d(4, 8, 3, bias=False))
    load_training_encoder(target, path)
    torch.testing.assert_close(target[0].weight, source[0].weight, rtol=0, atol=0)


@pytest.mark.parametrize('split,train_split', [('val','training'), ('train','validation')])
def test_encoder_sources_cannot_alias_validation_as_training(tmp_path, split, train_split):
    path = tmp_path/'source.pth'
    source_checkpoint(path, split, train_split)
    with pytest.raises(ValueError, match='official training'):
        load_training_encoder(nn.Linear(3, 2), path)


def test_train_mode_cannot_update_frozen_encoder_parameters_or_batchnorm():
    model = Q0Pretrainer.__new__(Q0Pretrainer)
    nn.Module.__init__(model)
    model.freeze_encoder = True
    model.pts_bev_encoder = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4)).requires_grad_(False)
    model.anchor_prior = nn.Linear(4, 2)
    before = {k:v.clone() for k,v in model.pts_bev_encoder.state_dict().items()}
    model.train()
    assert not model.pts_bev_encoder.training and model.anchor_prior.training
    model.anchor_prior(model.pts_bev_encoder(torch.randn(8, 3))).square().mean().backward()
    assert all(p.grad is None for p in model.pts_bev_encoder.parameters())
    assert model.anchor_prior.weight.grad.abs().sum() > 0
    for key,value in model.pts_bev_encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_saved_model_is_self_contained_and_need_not_reopen_initialization_checkpoint(tmp_path):
    register_all_modules()
    root = Path(__file__).resolve().parents[2]
    cfg = Config.fromfile(root / 'configs/main/laneloom_openlane_image_lidar_temporal.py')
    cfg.q0_encoder_checkpoint = str(tmp_path/'missing-source.pth')
    cfg.q0_freeze_encoder = True
    cfg.q0_evaluation_split = 'validation'
    model = Q0Pretrainer(cfg, initialize_encoder=False)
    assert all(not p.requires_grad for p in model.pts_bev_encoder.parameters())
    protocol = dict(target_overlap=0, feature_cache_used=False, target_oversampling=False)
    path = tmp_path/'trained.pth'
    torch.save(dict(state_dict=model.state_dict(), step=12,
                    meta=dict(protocol=protocol,cfg=cfg.pretty_text)), path)
    restored,_,step,_ = load_model(path, 'cpu')
    assert step == 12
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)


def test_training_pretrained_encoder_rejects_training_scene_holdout(tmp_path):
    cfg = Config(dict(q0_encoder_checkpoint=str(tmp_path / 'training.pth'),
                      q0_freeze_encoder=True, q0_evaluation_split='training'))
    with pytest.raises(ValueError, match='holdout'):
        Q0Pretrainer(cfg)
