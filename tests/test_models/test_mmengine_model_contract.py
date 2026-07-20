import inspect
from pathlib import Path
from unittest.mock import Mock

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.model import BaseModel
from mmdet3d.registry import MODELS

import LaneLoom  # noqa: F401 - register the project model.
from LaneLoom.detectors import LaneLoom as LaneLoomModel


def test_laneloom_is_the_single_mmengine_model_entrypoint():
    assert issubclass(LaneLoomModel, BaseModel)
    assert MODELS.get('LaneLoom') is LaneLoomModel

    parameters = inspect.signature(LaneLoomModel.forward).parameters
    assert 'mode' in parameters
    assert parameters.get('return' + '_loss') is None


def test_laneloom_forward_accepts_ddp_keyword_batches():
    signature = inspect.signature(LaneLoomModel.forward)
    signature.bind(
        None,
        mode='loss',
        img='image-batch',
        img_metas='metadata-batch',
        lidar='point-batch')


def test_config_enables_unused_parameter_detection_for_dynamic_bev_branches():
    config_path = Path(__file__).resolve().parents[2] / 'configs/laneloom_openlane_image_lidar.py'
    cfg = Config.fromfile(config_path)
    assert cfg.model_wrapper_cfg.type == 'MMDistributedDataParallel'
    assert cfg.model_wrapper_cfg.find_unused_parameters is True


def test_config_expands_five_lidar_shapes_into_505_decoder_queries():
    config_path = Path(__file__).resolve().parents[2] / 'configs/laneloom_openlane_image_lidar.py'
    cfg = Config.fromfile(config_path)
    prior = cfg.model.lane_head.anchor_prior
    expander = cfg.model.lane_head.proposal_expander
    decoder = cfg.model.lane_head.lane_decoder

    assert prior.max_templates == 5
    assert prior.lateral_slope_scale == 0.1
    assert prior.vertical_slope_scale == 0.03
    assert 'max_yaw_degrees' not in prior
    assert 'max_pitch_degrees' not in prior
    assert 'pc_range' not in prior
    assert 'shape_nms_threshold' not in prior
    assert expander.num_x_seeds == 101
    assert decoder.num_query == 505
    assert decoder.layer_query_budget == 64
    assert prior.anchor_y_steps == expander.anchor_y_steps == decoder.anchor_y_steps
    assert decoder.anchor_y_steps == [2. * index for index in range(1, 51)]
    assert decoder.evidence_y_steps == [5. * index for index in range(1, 21)]
    assert decoder.num_layers == 2


def test_config_uses_single_level_temporal_fv_and_dynamic_pillar_bev():
    config_path = Path(__file__).resolve().parents[2] / 'configs/laneloom_openlane_image_lidar.py'
    cfg = Config.fromfile(config_path)

    assert cfg.model.num_levels == 1
    assert cfg.model.img_backbone.out_indices == (1,)
    assert cfg.model.pts_bev_encoder.type == (
        'DynamicPillarBEVEncoder')
    assert cfg.model.pts_bev_encoder.pillar_size == [0.25, 0.25]
    assert cfg.model.pts_bev_encoder.out_channels == 128
    assert cfg.model.pts_bev_encoder.base_channels == 32
    assert 'pts_voxel_layer' not in cfg.model
    assert cfg.model.lane_head.anchor_prior.lidar_in_channels == 128
    assert cfg.model.lane_head.bev_encoder.bev_h == 200
    assert cfg.model.lane_head.bev_encoder.bev_w == 100
    assert 'embed_dims' not in cfg.model.lane_head.bev_encoder
    assert 'pc_range' not in cfg.model.lane_head.bev_encoder
    assert 'bev_h' not in cfg.model.lane_head.lane_decoder
    assert 'bev_w' not in cfg.model.lane_head.lane_decoder
    assert 'frames' not in cfg.model
    assert 'add_extra_convs' not in cfg.model.img_neck
    assert 'relu_before_extra_convs' not in cfg.model.img_neck
    assert 'resume_from' not in cfg
    assert (
        cfg.model.lane_head.bev_encoder.lidar_bev_adapter.lidar_in_channels
        == 128)
    assert (
        cfg.model.lane_head.bev_encoder.lidar_bev_adapter.embed_dims
        == 128)
    assert cfg.model.lane_head.query_sampler.type == (
        'DualPlaneTemporalSampler')
    assert cfg.model.lane_head.query_sampler.lidar_in_channels == 128
    assert 'spatial_cross_attn' not in cfg.model.lane_head.bev_encoder
    assert 'lidar_fusion' not in cfg.model.lane_head.bev_encoder
    assert 'temporal_self_attn' not in cfg.model.lane_head.bev_encoder


def test_config_uses_strict_q0_and_one_to_one_assignment():
    config_path = Path(__file__).resolve().parents[2] / 'configs/laneloom_openlane_image_lidar.py'
    cfg = Config.fromfile(config_path)
    loss = cfg.model.lane_head.loss_module

    assert loss.type == 'LaneLoss'
    assert loss.objectness_loss_weight == 1.0
    assert loss.category_loss_weight == 0.25
    assert loss.negative_objectness_weight == 1.0
    assert loss.ambiguous_negative_weight == 0.1
    assert loss.diagnostic_geometry_distance_threshold == 0.3
    assert loss.diagnostic_geometry_match_ratio == 0.8
    assert loss.diagnostic_score_threshold == 0.5
    assert loss.assigner.type == 'HungarianAssigner'
    assert loss.visibility_loss_weight == 1.0
    assert loss.focal_alpha == 0.25
    assert loss.focal_gamma == 2.0
    assert 'visibility_dice_loss_weight' not in loss
    assert 'visibility_boundary_loss_weight' not in loss
    assert cfg.model.lane_head.lane_decoder.get('dn_num_groups', 0) == 0
    assert (
        cfg.model.lane_head.lane_decoder.objectness_prior_probability
        == 0.01)
    expander = cfg.model.lane_head.proposal_expander
    assert expander.q0_distance_threshold == 0.3
    assert expander.q0_match_ratio == 0.8
    assert expander.q0_geometry_loss_weight == 1.0
    assert cfg.model.lane_head.test_cfg.curve_nms.distance_threshold == 0.3
    assert cfg.model.lane_head.test_cfg.curve_nms.match_ratio_threshold == 0.75
    assert cfg.model.lane_head.test_cfg.score_threshold == 0.5
    assert cfg.val_evaluator.score_threshold == 0.5


def _bare_laneloom():
    model = LaneLoomModel.__new__(LaneLoomModel)
    nn.Module.__init__(model)
    model.num_levels = None
    return model


def test_temporal_image_features_keep_every_frame_for_curve_sampling():
    model = _bare_laneloom()
    model.extract_img_feat = Mock(
        return_value=[torch.randn(6, 1, 4, 2, 2)])
    image_queue = torch.randn(2, 3, 1, 3, 4, 4)

    temporal = model.extract_temporal_img_feat(image_queue)

    assert temporal[0].shape == (2, 3, 1, 4, 2, 2)
    assert model.extract_img_feat.call_args.args[0].shape == (6, 1, 3, 4, 4)


def test_current_lidar_backbone_receives_the_complete_aligned_point_queue():
    model = _bare_laneloom()
    model.extract_temporal_img_feat = Mock(
        return_value=[torch.randn(1, 3, 1, 4, 2, 2)])
    model.extract_pts_feat = Mock(return_value=[torch.randn(1, 4, 2, 2)])
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

    passed_points = model.extract_pts_feat.call_args.args[0]
    assert passed_points[0].size(0) == 3
    assert encoded[3][0].size(0) == 3
    assert len(encoded) == 5
