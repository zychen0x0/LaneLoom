import torch
import pytest

from LaneLoom.models.head.curve_nms import (
    curve_nms,
    mean_distance_curve_nms,
)


def _nms(x, visibility, scores, z=None):
    if z is None:
        z = torch.zeros(1, len(x), len(x[0]))
    else:
        z = torch.tensor([z], dtype=torch.float32)
    return curve_nms(
        torch.tensor([x], dtype=torch.float32),
        z,
        torch.tensor([visibility], dtype=torch.float32),
        torch.tensor([scores]),
        tube_radius=0.5,
        iou_threshold=0.6,
        containment_threshold=0.9)


def test_curve_nms_keeps_highest_scoring_duplicate_and_adjacent_lane():
    keep = _nms(
        x=[[0.0, 0.0, 0.0, 0.0],
           [0.1, 0.1, 0.1, 0.1],
           [3.0, 3.0, 3.0, 3.0]],
        visibility=[[1.0] * 4] * 3,
        scores=[0.8, 0.9, 0.7])

    assert keep.tolist() == [[False, True, True]]


def test_curve_nms_keeps_lanes_without_enough_visible_geometric_overlap():
    keep = _nms(
        x=[[0.0, 0.0, 0.0, 0.0],
           [0.1, 0.1, 1.0, 1.0],
           [0.1, 0.1, 0.1, 0.1]],
        visibility=[[1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0]],
        scores=[0.9, 0.8, 0.7])

    assert keep.tolist() == [[True, True, True]]


def test_curve_nms_suppresses_a_lower_score_contained_fragment():
    keep = _nms(
        x=[[0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0],
           [3.0, 3.0, 3.0, 3.0]],
        visibility=[[1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0]],
        scores=[0.9, 0.8, 0.7])

    assert keep.tolist() == [[True, False, True]]


def test_curve_nms_keeps_a_better_short_lane_over_a_worse_overextension():
    keep = _nms(
        x=[[0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0]],
        visibility=[[1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 0.0]],
        scores=[0.8, 0.9])

    assert keep.tolist() == [[False, True]]


def test_curve_nms_uses_height_when_comparing_3d_lanes():
    keep = _nms(
        x=[[0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0]],
        z=[[0.0, 0.0, 0.0, 0.0],
           [0.4, 0.4, 0.4, 0.4]],
        visibility=[[1.0] * 4] * 2,
        scores=[0.9, 0.8])

    assert keep.tolist() == [[True, True]]


def test_curve_nms_only_compares_score_and_visibility_candidates():
    x = torch.zeros(1, 3, 4)
    z = torch.zeros_like(x)
    visibility = torch.ones_like(x)
    scores = torch.tensor([[.9, .8, .1]])
    candidate_mask = torch.tensor([[True, True, False]])

    keep = curve_nms(
        x, z, visibility, scores,
        tube_radius=.5,
        iou_threshold=.6,
        containment_threshold=.9,
        candidate_mask=candidate_mask)

    assert keep.tolist() == [[True, False, False]]


def test_mean_distance_curve_nms_suppresses_only_close_post_tube_candidates():
    x = torch.tensor([[[0.0, 0.0, 0.0],
                       [0.24, 0.24, 0.24],
                       [0.10, 0.10, 0.10],
                       [3.0, 3.0, 3.0]]])
    z = torch.zeros_like(x)
    visibility = torch.ones_like(x)
    scores = torch.tensor([[0.9, 0.8, 0.95, 0.7]])
    post_tube_candidates = torch.tensor([[True, True, False, True]])

    keep = mean_distance_curve_nms(
        x, z, visibility, scores,
        distance_threshold=0.25,
        candidate_mask=post_tube_candidates)

    assert keep.tolist() == [[True, False, False, True]]


@pytest.mark.parametrize('threshold', [0.0, float('nan'), float('inf')])
def test_mean_distance_curve_nms_requires_a_finite_positive_threshold(
        threshold):
    values = torch.zeros(1, 1, 2)
    with pytest.raises(ValueError, match='finite positive'):
        mean_distance_curve_nms(
            values, values, values, torch.ones(1, 1),
            distance_threshold=threshold)
