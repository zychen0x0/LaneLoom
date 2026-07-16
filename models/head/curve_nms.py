"""Visibility-aware non-maximum suppression for sampled 3D lanes."""

import torch


def curve_nms(x, z, visibility, scores, *, visibility_threshold,
              distance_threshold,
              match_ratio_threshold):
    """Return the highest-scoring non-duplicate lane queries.

    Two predictions are duplicates only when anchors closer than
    ``distance_threshold`` cover at least ``match_ratio_threshold`` of both
    predictions' visible intervals.  This prevents a short shared segment from
    suppressing two lanes that diverge later.
    """
    visible = visibility >= visibility_threshold
    shared = visible.unsqueeze(2) & visible.unsqueeze(1)
    distance = torch.hypot(
        x.unsqueeze(2) - x.unsqueeze(1),
        z.unsqueeze(2) - z.unsqueeze(1))
    matched_count = (shared & (distance < distance_threshold)).sum(dim=-1)
    visible_count = visible.sum(dim=-1)
    duplicate = (
        (matched_count > 0) &
        (matched_count >=
         match_ratio_threshold * visible_count.unsqueeze(2)) &
        (matched_count >=
         match_ratio_threshold * visible_count.unsqueeze(1)))
    duplicate.diagonal(dim1=1, dim2=2).fill_(False)

    keep = torch.zeros_like(scores, dtype=torch.bool)
    for batch_idx, order in enumerate(torch.argsort(scores, dim=1, descending=True)):
        suppressed = torch.zeros_like(scores[batch_idx], dtype=torch.bool)
        for candidate in order:
            selected = ~suppressed[candidate]
            keep[batch_idx, candidate] = selected
            suppressed |= duplicate[batch_idx, candidate] & selected
    return keep
