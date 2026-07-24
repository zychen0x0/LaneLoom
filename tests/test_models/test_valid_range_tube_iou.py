import pytest
import torch

from LaneLoom.models.valid_range_tube_iou import (
    pairwise_valid_range_tube_overlap,
    valid_range_tube_iou,
)


def test_valid_range_tube_iou_is_one_for_identical_tubes():
    x = torch.tensor([[0.0, 0.1, 0.2, 0.3]])
    z = torch.zeros_like(x)
    visibility = torch.ones_like(x)

    tube_iou = valid_range_tube_iou(
        x, z, visibility, x, z, visibility)

    assert torch.allclose(tube_iou, torch.ones_like(tube_iou), atol=1e-5)


def test_valid_range_tube_iou_is_zero_for_disjoint_valid_ranges():
    x = torch.zeros(1, 4)
    z = torch.zeros_like(x)
    first_visibility = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    second_visibility = 1.0 - first_visibility

    tube_iou = valid_range_tube_iou(
        x, z, first_visibility, x, z, second_visibility)

    assert torch.equal(tube_iou, torch.zeros_like(tube_iou))


def test_pairwise_overlap_detects_a_short_tube_contained_by_a_long_tube():
    x = torch.zeros(1, 2, 4)
    z = torch.zeros_like(x)
    visibility = torch.tensor([[
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
    ]])

    tube_iou, containment = pairwise_valid_range_tube_overlap(
        x, z, visibility)

    assert torch.allclose(tube_iou[0, 0, 1], torch.tensor(0.5), atol=1e-5)
    assert torch.allclose(containment[0, 0, 1], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(tube_iou, tube_iou.transpose(1, 2))
    assert torch.allclose(containment, containment.transpose(1, 2))


def test_valid_range_tube_iou_is_zero_beyond_two_tube_radii():
    first_x = torch.zeros(1, 3)
    second_x = torch.full((1, 3), 1.0)
    z = torch.zeros_like(first_x)
    visibility = torch.ones_like(first_x)

    tube_iou = valid_range_tube_iou(
        first_x, z, visibility,
        second_x, z, visibility,
        tube_radius=0.5)

    assert torch.equal(tube_iou, torch.zeros_like(tube_iou))


def test_valid_range_tube_iou_supports_soft_visibility():
    x = torch.zeros(1, 3)
    z = torch.zeros_like(x)
    predicted_visibility = torch.tensor([[0.25, 0.5, 0.75]])
    target_visibility = torch.ones_like(predicted_visibility)

    tube_iou = valid_range_tube_iou(
        x, z, predicted_visibility,
        x, z, target_visibility)

    assert torch.allclose(
        tube_iou, predicted_visibility.mean(dim=-1), atol=1e-5)


def test_valid_range_tube_iou_returns_zero_for_empty_visibility():
    x = torch.zeros(1, 3)
    z = torch.zeros_like(x)
    visibility = torch.zeros_like(x)

    tube_iou = valid_range_tube_iou(
        x, z, visibility, x, z, visibility)

    assert torch.equal(tube_iou, torch.zeros_like(tube_iou))


def test_valid_range_tube_iou_has_finite_geometry_and_visibility_gradients():
    predicted_x = torch.full((1, 3), 0.2, requires_grad=True)
    predicted_z = torch.zeros(1, 3, requires_grad=True)
    predicted_visibility = torch.full(
        (1, 3), 0.5, requires_grad=True)
    target_x = torch.zeros(1, 3)
    target_z = torch.zeros(1, 3)
    target_visibility = torch.ones(1, 3)

    loss = 1.0 - valid_range_tube_iou(
        predicted_x, predicted_z, predicted_visibility,
        target_x, target_z, target_visibility)
    loss.sum().backward()

    for tensor in (
            predicted_x, predicted_z, predicted_visibility):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_identical_tubes_have_finite_zero_geometry_gradients():
    predicted_x = torch.zeros(1, 3, requires_grad=True)
    predicted_z = torch.zeros(1, 3, requires_grad=True)
    visibility = torch.ones(1, 3)

    loss = 1.0 - valid_range_tube_iou(
        predicted_x, predicted_z, visibility,
        torch.zeros_like(predicted_x), torch.zeros_like(predicted_z),
        visibility)
    loss.sum().backward()

    assert torch.isfinite(predicted_x.grad).all()
    assert torch.isfinite(predicted_z.grad).all()


def test_pairwise_overlap_returns_zero_for_empty_tubes():
    x = torch.zeros(1, 2, 3)
    z = torch.zeros_like(x)
    visibility = torch.zeros_like(x)

    tube_iou, containment = pairwise_valid_range_tube_overlap(
        x, z, visibility)

    assert torch.equal(tube_iou, torch.zeros_like(tube_iou))
    assert torch.equal(containment, torch.zeros_like(containment))


@pytest.mark.parametrize(
    ('name', 'value'),
    [('tube_radius', 0.0), ('z_weight', -1.0), ('epsilon', 0.0)])
def test_pairwise_overlap_rejects_invalid_tube_parameters(name, value):
    x = torch.zeros(1, 2, 3)
    kwargs = {name: value}

    with pytest.raises(ValueError):
        pairwise_valid_range_tube_overlap(x, x, x, **kwargs)
