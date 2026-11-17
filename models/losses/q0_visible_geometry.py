"""Supervise Q0 against observed JSON points, without completing their curves."""
import torch
import torch.nn.functional as F

from .hungarian_assigner import batched_hungarian_assignment


def raw_visible_geometry_loss(expanded, points, visibility, length=100., threshold=.3, ratio=.8):
    """One distinct Q0 per GT, evaluated at that GT's actual longitudinal positions.

    Matching maximizes threshold coverage first, as in the diagnostic metric.
    The single regression objective uses all visible points and has no dead
    zone: already-small horizontal errors still receive corrective gradients.
    Each valid GT contributes equally, including short fragments.
    """
    x, z = expanded['curve_x_controls'], expanded['curve_z_controls']
    points = points.to(x)
    visible = visibility.bool() & torch.isfinite(points).all(-1)
    points = torch.where(visible[..., None], points, 0)
    t = points[..., 1] / length
    basis = torch.stack(((1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3), -1)
    predicted_x = torch.einsum('bqc,blnc->bqln', x, basis)
    predicted_z = torch.einsum('bqc,blnc->bqln', z, basis)
    delta = torch.stack((predicted_x - points[:, None, :, :, 0],
                         predicted_z - points[:, None, :, :, 2]), -1)
    distance = torch.linalg.vector_norm(delta, dim=-1)
    counts = visible.sum(-1)
    valid = counts >= 2
    sample_valid = expanded.get('from_lidar')
    if torch.is_tensor(sample_valid):
        valid = valid & sample_valid[:, None]
    elif sample_valid is False:
        valid = valid & False
    point_loss = F.smooth_l1_loss(distance, torch.zeros_like(distance),
                                beta=threshold, reduction='none')
    pair_loss = (point_loss * visible[:, None]).sum(-1) / counts[:, None].clamp_min(1)
    ordered = distance.masked_fill(~visible[:, None], torch.inf).sort(-1).values
    index = (torch.ceil(ratio * counts).long() - 1).clamp_min(0)
    percentile = ordered.gather(-1, index[:, None, :, None].expand(x.size(0), x.size(1), -1, 1)).squeeze(-1)
    assignment_cost = ((percentile >= threshold).to(x.dtype) * 1e6
                       + percentile.clamp_max(1e3)).masked_fill(~valid[:, None], torch.inf)
    assignments = batched_hungarian_assignment(assignment_cost)
    matched = assignments >= 0
    indices = assignments.clamp_min(0)[..., None]
    selected = pair_loss.gather(-1, indices).squeeze(-1)
    loss = torch.where(matched, selected, 0).sum() / matched.sum().clamp_min(1)
    covered = (percentile < threshold).gather(-1, indices).squeeze(-1) & matched
    return dict(loss_q0_raw_geometry=loss,
                diag_q0_raw_gt=valid.sum().to(loss).detach(),
                diag_q0_raw_covered=covered.sum().to(loss).detach())
