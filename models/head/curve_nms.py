"""VR-TubeIoU non-maximum suppression for sampled 3D lanes."""

import torch

from ..valid_range_tube_iou import pairwise_valid_range_tube_overlap


def curve_nms(
        x, z, visibility, scores, *, tube_radius,
        iou_threshold, containment_threshold,
        z_weight=1.0, epsilon=1e-6, candidate_mask=None):
    """Keep the highest-score lane from each overlapping tube group."""
    keep = torch.zeros_like(scores, dtype=torch.bool)
    if candidate_mask is None:
        candidate_mask = torch.ones_like(scores, dtype=torch.bool)
    for batch_idx in range(scores.size(0)):
        indices = torch.where(candidate_mask[batch_idx])[0]
        if indices.numel() == 0:
            continue
        pairwise_iou, pairwise_containment = (
            pairwise_valid_range_tube_overlap(
                x[batch_idx, indices].unsqueeze(0),
                z[batch_idx, indices].unsqueeze(0),
                visibility[batch_idx, indices].unsqueeze(0),
                tube_radius=tube_radius,
                z_weight=z_weight,
                epsilon=epsilon))
        duplicate = (
            (pairwise_iou[0] > iou_threshold)
            | (pairwise_containment[0] > containment_threshold))
        duplicate.diagonal().fill_(False)

        order = torch.argsort(
            scores[batch_idx, indices], descending=True)
        suppressed = torch.zeros(
            indices.numel(), dtype=torch.bool, device=scores.device)
        for local_candidate in order:
            selected = ~suppressed[local_candidate]
            keep[batch_idx, indices[local_candidate]] = selected
            suppressed |= duplicate[local_candidate] & selected
    return keep


def mean_distance_curve_nms(
        x, z, visibility, scores, *, distance_threshold,
        visibility_threshold=0.5, candidate_mask=None):
    """Keep the highest-score lane among close post-Tube candidates."""
    if distance_threshold <= 0:
        raise ValueError('distance_threshold must be positive.')
    keep = torch.zeros_like(scores, dtype=torch.bool)
    if candidate_mask is None:
        candidate_mask = torch.ones_like(scores, dtype=torch.bool)
    for batch_idx in range(scores.size(0)):
        indices = torch.where(candidate_mask[batch_idx])[0]
        if indices.numel() == 0:
            continue
        candidate_x = x[batch_idx, indices]
        candidate_z = z[batch_idx, indices]
        visible = (
            visibility[batch_idx, indices] >= visibility_threshold)
        shared = visible[:, None] & visible[None]
        shared_count = shared.sum(dim=-1)
        distance = torch.linalg.vector_norm(torch.stack((
            candidate_x[:, None] - candidate_x[None],
            candidate_z[:, None] - candidate_z[None],
        ), dim=-1), dim=-1)
        mean_distance = (
            (distance * shared).sum(dim=-1)
            / shared_count.clamp_min(1))
        duplicate = (
            (shared_count >= 2)
            & (mean_distance <= distance_threshold))
        duplicate.diagonal().fill_(False)

        order = torch.argsort(
            scores[batch_idx, indices], descending=True)
        suppressed = torch.zeros(
            indices.numel(), dtype=torch.bool, device=scores.device)
        for local_candidate in order:
            selected = ~suppressed[local_candidate]
            keep[batch_idx, indices[local_candidate]] = selected
            suppressed |= duplicate[local_candidate] & selected
    return keep
