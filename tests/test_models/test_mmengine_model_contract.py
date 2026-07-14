import inspect
from pathlib import Path
from unittest.mock import Mock

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.model import BaseModel
from mmdet3d.registry import MODELS

import LaneFormer  # noqa: F401 - register the project model.
from LaneFormer.detectors import LaneFormer as LaneFormerModel


def test_laneformer_is_the_single_mmengine_model_entrypoint():
    assert issubclass(LaneFormerModel, BaseModel)
    assert MODELS.get('LaneFormer') is LaneFormerModel

    parameters = inspect.signature(LaneFormerModel.forward).parameters
    assert 'mode' in parameters
    assert parameters.get('return' + '_loss') is None


def test_laneformer_forward_accepts_ddp_keyword_batches():
    signature = inspect.signature(LaneFormerModel.forward)
    signature.bind(
        None,
        mode='loss',
        img='image-batch',
        img_metas='metadata-batch',
        lidar='point-batch')


def test_config_enables_unused_parameter_detection_for_dynamic_bev_branches():
    config_path = Path(__file__).resolve().parents[2] / 'configs/laneformer_openlane_image_lidar.py'
    cfg = Config.fromfile(config_path)
    assert cfg.model_wrapper_cfg.type == 'MMDistributedDataParallel'
    assert cfg.model_wrapper_cfg.find_unused_parameters is True


class _ImageHistoryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward_image_bev(self, img_feats, img_metas):
        self.calls += 1
        return img_feats[0].mean(dim=1)


class _PriorRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_count = None

    def forward(self, points):
        self.point_count = points[0].size(0)
        return {'road_valid': torch.ones(1, 1, dtype=torch.bool)}


def _bare_laneformer():
    model = LaneFormerModel.__new__(LaneFormerModel)
    nn.Module.__init__(model)
    model.frames = (-1, 0)
    model.num_levels = None
    return model


def test_history_bev_is_image_only_even_when_point_queue_is_available():
    model = _bare_laneformer()
    model.lane_head = _ImageHistoryHead()
    model.extract_img_feat = Mock(
        return_value=[torch.randn(1, 1, 4, 2, 2)])
    model.extract_feat = Mock(side_effect=AssertionError(
        'history must not invoke the multimodal feature extractor'))
    image_queue = torch.randn(1, 1, 1, 3, 4, 4)
    points = [torch.randn(8, 6)]
    metas = [dict(queue={0: {}, 1: {}})]

    history = model.obtain_history_bev(
        image_queue, img_metas_queue=metas, points_queue=points)

    assert len(history) == 1
    assert model.lane_head.calls == 1
    model.extract_feat.assert_not_called()


def test_current_lidar_backbone_receives_the_complete_aligned_point_queue():
    model = _bare_laneformer()
    model.obtain_history_bev = Mock(return_value=[])
    model.extract_feat = Mock(return_value=(
        [torch.randn(1, 1, 4, 2, 2)], [torch.randn(1, 4, 2, 2)]))
    model.ground_prior_extractor = _PriorRecorder()
    accumulated = torch.tensor([
        [0., 2., -1., 1., 0., 0.],
        [0., 3., -1., 1., 0., 1.],
        [0., 4., -1., 1., 0., 2.],
    ])
    batch = dict(
        img=torch.randn(1, 3, 1, 3, 4, 4),
        lidar=[accumulated],
        img_metas=[dict(queue={0: {}, 1: {}, 2: {}})])

    encoded = model._encode_batch(batch)

    passed_points = model.extract_feat.call_args.kwargs['points']
    assert passed_points[0].size(0) == 3
    assert model.ground_prior_extractor.point_count == 3
    assert encoded[-1]['road_valid'].all()
