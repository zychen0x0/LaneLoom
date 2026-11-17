import numpy as np
import torch

from LaneLoom.datasets.openlane.dataset import openlane_raw_visible_gt, _build_cam_to_ground_lane
from LaneLoom.models.losses.q0_visible_geometry import raw_visible_geometry_loss
from LaneLoom.models.head.lidar_shape_token_prior import LiDARShapeTokenPrior


def test_raw_points_keep_short_nonuniform_support_and_visibility():
    ground = np.array([[0., 9.5, .1], [0., 9.8, .2], [0., 10.1, .3]])
    transform = _build_cam_to_ground_lane(np.eye(4, dtype=np.float32))
    camera = np.linalg.inv(transform) @ np.column_stack((ground, np.ones(3))).T
    label = dict(extrinsic=np.eye(4).tolist(), lane_lines=[dict(xyz=camera[:3].tolist(), visibility=[1, 1, 0])])
    points, visible = openlane_raw_visible_gt([label], [-25, 0, -5, 25, 100, 5])
    assert visible.sum() == 2
    torch.testing.assert_close(points[0, 0], torch.tensor(ground[:2], dtype=torch.float32))


def test_joint_loss_corrects_horizontal_error_inside_old_dead_zone():
    controls = torch.full((1, 1, 4), .29, requires_grad=True)
    prior = dict(curve_x_controls=controls, curve_z_controls=torch.zeros_like(controls))
    points = torch.tensor([[[[0., 9.5, 0.], [0., 9.8, 0.]]]])
    losses = raw_visible_geometry_loss(prior, points, torch.ones(1, 1, 2, dtype=torch.bool))
    losses['loss_q0_raw_geometry'].backward()
    assert losses['loss_q0_raw_geometry'] > 0
    assert controls.grad.abs().sum() > 0


def test_distinct_query_assignment_prevents_reusing_one_good_candidate():
    controls = torch.tensor([[[0., 0., 0., 0.], [2., 2., 2., 2.]]], requires_grad=True)
    points = torch.tensor([[[[0., 10., 0.], [0., 20., 0.]], [[.1, 10., 0.], [.1, 20., 0.]]]])
    losses = raw_visible_geometry_loss(dict(curve_x_controls=controls, curve_z_controls=torch.zeros_like(controls)),
                                      points, torch.ones(1, 2, 2, dtype=torch.bool))
    assert losses['diag_q0_raw_gt'] == 2
    assert losses['diag_q0_raw_covered'] == 1
    losses['loss_q0_raw_geometry'].backward()
    assert controls.grad[0, 1].abs().sum() > 0


def test_empty_gt_has_finite_differentiable_zero_loss():
    controls = torch.randn(1, 3, 4, requires_grad=True)
    result = raw_visible_geometry_loss(dict(curve_x_controls=controls, curve_z_controls=controls),
                                       torch.full((1, 1, 1, 3), float('nan')), torch.zeros(1, 1, 1, dtype=torch.bool))
    assert result['loss_q0_raw_geometry'] == 0
    result['loss_q0_raw_geometry'].backward()
    assert torch.isfinite(controls.grad).all()


def test_perfect_relative_height_template_can_disagree_with_absolute_visible_gt():
    # An analytic road with a nonzero height intercept, independent of target scenes.
    y = torch.tensor([5., 10., 20., 30.])
    relative = torch.stack((torch.zeros_like(y), y, .01 * y), -1)
    model = LiDARShapeTokenPrior(lidar_in_channels=4, hidden_channels=8,
                                anchor_y_steps=y, max_templates=5,
                                shape_supervision='full_templates')
    targets = relative[None, None].expand(1, 5, -1, -1).clone()
    template_loss = model.loss(dict(shape_points=targets, from_lidar=True),
                               torch.empty(1, 0, 4, 3), torch.empty(1, 0, 4),
                               shape_templates=targets,
                               shape_template_valid=torch.tensor([[True, False, False, False, False]]))
    assert template_loss['loss_lidar_shape_template'] == 0
    controls = torch.tensor([[[0., 1/3, 2/3, 1.]]])
    absolute = relative[None, None].clone()
    absolute[..., 2] += .4
    raw_loss = raw_visible_geometry_loss(dict(curve_x_controls=torch.zeros_like(controls),
                                             curve_z_controls=controls),
                                        absolute, torch.ones(1, 1, 4, dtype=torch.bool))
    assert raw_loss['loss_q0_raw_geometry'] > 0
    assert raw_loss['diag_q0_raw_covered'] == 0
