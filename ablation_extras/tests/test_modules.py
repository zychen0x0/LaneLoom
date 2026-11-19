"""CPU checks for the ablation modules against the real LaneLoom parents.

Run from the repository root:
    PYTHONPATH="$(dirname "$PWD")" python -m pytest ablation_extras/tests -q

On a machine without spconv or mmdet3d, lightweight stand-ins are installed
for those two imports only; every LaneLoom class under test is the real one.
"""

import importlib
import sys
import types

import pytest
import torch


def _install_stand_ins():
    try:
        importlib.import_module('mmdet3d.registry')
    except Exception:
        from mmengine.registry import Registry
        registry = types.ModuleType('mmdet3d.registry')
        registry.MODELS = Registry('model')
        registry.METRICS = Registry('metric')
        registry.DATASETS = Registry('dataset')
        registry.TASK_UTILS = Registry('task util')
        registry.TRANSFORMS = Registry('transform')
        package = types.ModuleType('mmdet3d')
        package.registry = registry
        sys.modules.setdefault('mmdet3d', package)
        sys.modules['mmdet3d.registry'] = registry
    try:
        importlib.import_module('spconv.pytorch')
    except Exception:
        spconv = types.ModuleType('spconv')
        spconv_pytorch = types.ModuleType('spconv.pytorch')

        class SparseModule(torch.nn.Module):
            pass

        def _unavailable(*args, **kwargs):
            return torch.nn.Identity()

        spconv_pytorch.SparseModule = SparseModule
        for name in ('SubMConv2d', 'SparseConv2d', 'SparseSequential',
                     'SparseConvTensor', 'SparseInverseConv2d'):
            setattr(spconv_pytorch, name, _unavailable)
        spconv.pytorch = spconv_pytorch
        sys.modules['spconv'] = spconv
        sys.modules['spconv.pytorch'] = spconv_pytorch


_install_stand_ins()
modules = importlib.import_module('LaneLoom.ablation_extras.modules')
temporal = importlib.import_module('LaneLoom.utils.temporal_sampling')


def _prior_kwargs():
    return dict(lidar_in_channels=8,
                anchor_y_steps=[2.0 * i for i in range(1, 51)],
                max_templates=5, hidden_channels=16)


def test_straight_prior_matches_parent_except_shape():
    torch.manual_seed(0)
    parent_cls = importlib.import_module(
        'LaneLoom.models.head.lidar_shape_token_prior').LiDARShapeTokenPrior
    straight = modules.AblationStraightTemplatePrior(**_prior_kwargs())
    parent = parent_cls(**_prior_kwargs())
    parent.load_state_dict(straight.state_dict())
    with torch.no_grad():
        for p in straight.template_head.parameters():
            p.normal_()
        parent.template_head.load_state_dict(
            straight.template_head.state_dict())
    feats = torch.randn(2, 8, 20, 10)
    out = straight._relative_controls(feats)
    ref = parent._relative_controls(feats)
    assert len(out) == 8
    for tensor in out[:6]:
        assert torch.count_nonzero(tensor) == 0
    assert torch.count_nonzero(ref[0]) > 0
    assert torch.equal(out[6], ref[6])
    assert torch.equal(out[7], ref[7])
    prior = straight(feats)
    assert torch.count_nonzero(prior['shape_points'][..., 0]) == 0
    assert torch.count_nonzero(prior['shape_points'][..., 2]) == 0


def test_current_sweep_encoder_drops_history(monkeypatch):
    cls = modules.AblationCurrentSweepPillarEncoder
    seen = {}

    def fake_forward(self, points, img_feats=None, img_metas=None,
                     images=None):
        seen['points'] = points
        return 'ok'

    parent = cls.__mro__[1]
    monkeypatch.setattr(parent, 'forward', fake_forward)
    encoder = cls.__new__(cls)
    torch.nn.Module.__init__(encoder)
    encoder.has_queue_index = True
    encoder.raw_queue_index = 5
    encoder.queue_length = temporal.resolve_temporal_sampling(
        (10.0, 5.0)).queue_length
    sample = torch.zeros(9, 6)
    sample[:, 5] = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2.0])
    assert encoder.forward([sample, sample[:4]]) == 'ok'
    kept = seen['points']
    assert [len(item) for item in kept] == [4, 0]
    assert torch.all(kept[0][:, 5] == 2)
    encoder.queue_length = 1
    encoder.forward([sample])
    assert len(seen['points'][0]) == 9


def _sampler(cls):
    return cls(image_in_channels=6, lidar_in_channels=4, embed_dims=8,
               pc_range=[-25, 0, -5, 25, 100, 5], num_image_levels=1,
               temporal_kernel_size=3)


def test_current_image_sampler_masks_history(monkeypatch):
    cls = modules.AblationCurrentImageSampler
    parent = cls.__mro__[1]
    sampled = torch.ones(1, 2, 3, 1, 3, 6)
    valid = torch.ones(1, 2, 3, 1, 3, dtype=torch.bool)
    monkeypatch.setattr(parent, '_sample_image',
                        lambda self, *a: (sampled, valid))
    out_sampled, out_valid = _sampler(cls)._sample_image(None, None, None, None)
    assert torch.count_nonzero(out_sampled[..., :2, :]) == 0
    assert torch.all(out_sampled[..., 2, :] == 1)
    assert not out_valid[..., :2].any() and out_valid[..., 2].all()
    assert torch.all(sampled == 1) and valid.all()


def test_mean_time_sampler_is_masked_mean():
    sampler = _sampler(modules.AblationMeanTimeSampler)
    sampled = torch.randn(1, 2, 3, 1, 3, 6)
    valid = torch.tensor([True, False, True]).view(1, 1, 1, 1, 3).expand(
        1, 2, 3, 1, 3)
    fused = sampler._fuse_time(sampled, valid)
    projected = sampler.image_adapter(sampled)
    expected = (projected[..., 0, :] + projected[..., 2, :]) / 2
    assert fused.shape == (1, 2, 3, 1, 8)
    assert torch.allclose(fused, expected, atol=1e-6)
    none_valid = torch.zeros_like(valid)
    assert torch.count_nonzero(sampler._fuse_time(sampled, none_valid)) == 0


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
