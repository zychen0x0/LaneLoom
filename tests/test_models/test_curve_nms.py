import torch

from LaneFormer.models.head.curve_nms import curve_nms


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
        visibility_threshold=0.5,
        distance_threshold=0.3,
        match_ratio_threshold=0.75)


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


def test_curve_nms_uses_height_when_comparing_3d_lanes():
    keep = _nms(
        x=[[0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0]],
        z=[[0.0, 0.0, 0.0, 0.0],
           [0.4, 0.4, 0.4, 0.4]],
        visibility=[[1.0] * 4] * 2,
        scores=[0.9, 0.8])

    assert keep.tolist() == [[True, True]]
