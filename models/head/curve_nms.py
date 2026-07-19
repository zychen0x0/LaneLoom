"""Visibility-aware non-maximum suppression for sampled 3D lanes."""

import torch


def curve_nms(x, z, visibility, scores, *, visibility_threshold,
              distance_threshold,
              match_ratio_threshold, candidate_mask=None):
    """Return the highest-scoring non-duplicate lane queries.

    Two predictions are duplicates only when anchors closer than
    ``distance_threshold`` cover at least ``match_ratio_threshold`` of both
    predictions' visible intervals.  This prevents a short shared segment from
    suppressing two lanes that diverge later.
    """
    keep = torch.zeros_like(scores, dtype=torch.bool)
    if candidate_mask is None:
        candidate_mask = torch.ones_like(scores, dtype=torch.bool)
    for batch_idx in range(scores.size(0)):
        indices = torch.where(candidate_mask[batch_idx])[0]
        if indices.numel() == 0:
            continue
        local_x = x[batch_idx, indices]
        local_z = z[batch_idx, indices]
        local_visible = (
            visibility[batch_idx, indices] >= visibility_threshold)
        shared = local_visible.unsqueeze(1) & local_visible.unsqueeze(0)
        distance = torch.hypot(
            local_x.unsqueeze(1) - local_x.unsqueeze(0),
            local_z.unsqueeze(1) - local_z.unsqueeze(0))
        matched_count = (
            shared & (distance < distance_threshold)).sum(dim=-1)
        visible_count = local_visible.sum(dim=-1)
        duplicate = (
            (matched_count > 0)
            & (matched_count >=
               match_ratio_threshold * visible_count.unsqueeze(1))
            & (matched_count >=
               match_ratio_threshold * visible_count.unsqueeze(0)))
        duplicate.diagonal().fill_(False)

        order = torch.argsort(scores[batch_idx, indices], descending=True)
        suppressed = torch.zeros(
            indices.numel(), dtype=torch.bool, device=scores.device)
        for local_candidate in order:
            selected = ~suppressed[local_candidate]
            keep[batch_idx, indices[local_candidate]] = selected
            suppressed |= duplicate[local_candidate] & selected
    return keep
