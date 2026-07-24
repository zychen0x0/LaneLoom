"""Visibility-aware overlap metrics for fixed-y 3D lane tubes."""

import math

import torch


def _validate_tube_parameters(tube_radius, z_weight, epsilon):
    if tube_radius <= 0:
        raise ValueError("tube_radius must be positive.")
    if z_weight < 0:
        raise ValueError("z_weight must be non-negative.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")


def _circle_intersection_area(distance, tube_radius):
    radius = float(tube_radius)
    ratio = distance / (2.0 * radius)
    safe_max = 1.0 - torch.finfo(distance.dtype).eps
    safe_ratio = ratio.clamp(min=0.0, max=safe_max)
    overlap = 2.0 * radius ** 2 * (
        torch.acos(safe_ratio)
        - safe_ratio * torch.sqrt((1.0 - safe_ratio.square()).clamp_min(0.0)))
    return torch.where(
        distance < 2.0 * radius, overlap, torch.zeros_like(overlap))


def _tube_statistics(
        first_x, first_z, first_visibility,
        second_x, second_z, second_visibility,
        *, tube_radius, z_weight):
    first_x, first_z, first_visibility, second_x, second_z, second_visibility = (
        torch.broadcast_tensors(
            first_x, first_z, first_visibility,
            second_x, second_z, second_visibility))
    first_visibility = first_visibility.to(first_x).clamp(0.0, 1.0)
    second_visibility = second_visibility.to(first_x).clamp(0.0, 1.0)
    distance = torch.linalg.vector_norm(torch.stack((
        first_x - second_x,
        (first_z - second_z) * math.sqrt(float(z_weight)),
    ), dim=-1), dim=-1)
    cross_section = first_x.new_tensor(math.pi * float(tube_radius) ** 2)
    intersection_per_anchor = (
        torch.minimum(first_visibility, second_visibility)
        * _circle_intersection_area(distance, tube_radius))
    intersection = intersection_per_anchor.sum(dim=-1)
    first_volume = (first_visibility * cross_section).sum(dim=-1)
    second_volume = (second_visibility * cross_section).sum(dim=-1)
    return intersection, first_volume, second_volume


def valid_range_tube_iou(
        first_x, first_z, first_visibility,
        second_x, second_z, second_visibility,
        *, tube_radius=0.5, z_weight=1.0, epsilon=1e-6):
    """Return VR-TubeIoU for aligned fixed-y lane samples."""
    _validate_tube_parameters(tube_radius, z_weight, epsilon)
    intersection, first_volume, second_volume = _tube_statistics(
        first_x, first_z, first_visibility,
        second_x, second_z, second_visibility,
        tube_radius=tube_radius, z_weight=z_weight)
    union = first_volume + second_volume - intersection
    return (intersection / (union + epsilon)).clamp(0.0, 1.0)


def pairwise_valid_range_tube_overlap(
        x, z, visibility, *, tube_radius=0.5, z_weight=1.0,
        epsilon=1e-6, chunk_size=64):
    """Return pairwise VR-TubeIoU and smaller-tube containment matrices."""
    if x.ndim != 3 or z.shape != x.shape or visibility.shape != x.shape:
        raise ValueError(
            "x, z, and visibility must have matching [B, Q, A] shapes.")
    _validate_tube_parameters(tube_radius, z_weight, epsilon)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    pairwise_iou = []
    pairwise_containment = []
    for start in range(0, x.size(1), chunk_size):
        stop = min(start + chunk_size, x.size(1))
        intersection, first_volume, second_volume = _tube_statistics(
            x[:, start:stop, None], z[:, start:stop, None],
            visibility[:, start:stop, None],
            x[:, None], z[:, None], visibility[:, None],
            tube_radius=tube_radius, z_weight=z_weight)
        union = first_volume + second_volume - intersection
        pairwise_iou.append(
            (intersection / (union + epsilon)).clamp(0.0, 1.0))
        pairwise_containment.append(
            (intersection / (
                torch.minimum(first_volume, second_volume) + epsilon
            )).clamp(0.0, 1.0))
    return (
        torch.cat(pairwise_iou, dim=1),
        torch.cat(pairwise_containment, dim=1),
    )
