"""Guard query identity and modality interventions in diagnostic exports."""
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from LaneLoom.vis.output import visualize_query_stages as vis


def test_query_identity_survives_pruning_and_visibility_gaps():
    stage = dict(query_ids=torch.tensor([9, 2]),
                 x=torch.tensor([[[9., 9., 9.], [2., 2., 2.]]]),
                 z=torch.zeros(1, 2, 3))
    curves = vis.curves_for_stage(stage, np.array([5., 10., 15.]),
                                 np.array([2, 9]),
                                 np.array([[True, False, True], [True, True, True]]))
    assert [q for q, _ in curves] == [2, 9]
    assert np.isnan(curves[0][1][1]).all()
    np.testing.assert_equal(curves[0][1][[0, 2], 0], [2, 2])


def test_stage_ids_map_initial_and_pruned_layers():
    def prediction(x):
        x = torch.tensor(x, dtype=torch.float32)[None, :, None].repeat(1, 1, 2)
        return dict(pred_x=x, pred_z=torch.zeros_like(x),
                    pred_cls=torch.ones(1, x.shape[1]), pred_vis=torch.ones_like(x),
                    pred_category=torch.ones(1, x.shape[1], 2),
                    curve_x_controls=x, curve_z_controls=torch.zeros_like(x))

    first = prediction([0, 1, 2, 3])
    last = prediction([30, 10])
    last.update(aux_predictions=[first], selected_query_indices=torch.tensor([[3, 1]]))
    data = dict(decoder=last, expanded=dict(
        curve_x_controls=torch.tensor([[[0., 0.], [1., 1.], [2., 2.], [3., 3.]]]),
        curve_z_controls=torch.zeros(1, 4, 2)))
    head = SimpleNamespace(lane_decoder=SimpleNamespace(bezier_basis=torch.eye(2)),
                           curve_nms_cfg={}, score_threshold=.5)
    stages = vis.stages_from_capture(data, head)
    assert [s['query_ids'].tolist() for s in stages] == [[0, 1, 2, 3], [0, 1, 2, 3], [3, 1]]
    curves = vis.curves_for_stage(stages[0], np.array([5., 10.]), np.array([3, 1]),
                                 np.ones((2, 2), dtype=bool))
    np.testing.assert_equal(curves[0][1][:, 0], [3, 3])


def test_modality_mask_and_hook_cleanup():
    class Sampler(nn.Module):
        def forward(self, points):
            return dict(tokens=torch.ones(1, 2, 6, 3),
                        key_padding_mask=torch.tensor([[[False, True, False, False, False, True]]]).expand(1, 2, 6),
                        image_valid_mask=torch.ones(1, 2, 2, 2, 1, dtype=torch.bool),
                        lidar_valid_mask=torch.ones(1, 2, 2, dtype=torch.bool))

    head = SimpleNamespace(anchor_prior=nn.Identity(), proposal_expander=nn.Identity(),
                           lane_decoder=nn.Identity(), query_sampler=Sampler())
    points = torch.zeros(1, 2, 2, 3)
    original = head.query_sampler(points)
    for mode, hidden in [('image_only', slice(4, 6)), ('bev_only', slice(0, 4))]:
        with vis.StageCapture(head, mode):
            output = head.query_sampler(points)
            assert output['key_padding_mask'][..., hidden].all()
            torch.testing.assert_close(output['tokens'], original['tokens'])
        torch.testing.assert_close(head.query_sampler(points)['key_padding_mask'], original['key_padding_mask'])
    assert not head.query_sampler._forward_hooks


def test_paired_error_support_never_extrapolates_gt():
    stage = dict(name='Q0', query_ids=torch.tensor([7]),
                 x=torch.tensor([[[100., 1., 1., 100.]]]), z=torch.zeros(1, 1, 4))
    y = np.array([5., 10., 15., 20.])
    gt = [np.array([[0., 10., 0.], [0., 15., 0.]])]
    result = vis.paired_errors([stage], y, gt, np.array([7]), np.ones((1, 4), dtype=bool))
    assert result['pairs'][0]['anchor_count'] == 2
    assert result['pairs'][0]['stages'][0]['x_mae_m'] == 1.
    empty = vis.paired_errors([stage], y, gt, np.array([], dtype=int), np.zeros((0, 4), dtype=bool))
    assert empty['unmatched_gt'] == 1
    assert empty['pairs'] == []


def test_no_detections_still_shows_scored_query_candidates():
    stage = dict(valid_queries=torch.zeros(1, 3, dtype=torch.bool),
                 scores=torch.tensor([[.1, .4, .2]]),
                 visibility=torch.tensor([[[.8, .8], [.9, .9], [.1, .1]]]))
    indices, visible, fallback = vis.select_tracked_queries(stage)
    assert fallback
    assert indices.tolist() == [1, 0]
    assert visible.all()
    stage['valid_queries'][0, 0] = True
    indices, _, fallback = vis.select_tracked_queries(stage)
    assert not fallback
    assert indices.tolist() == [0]
