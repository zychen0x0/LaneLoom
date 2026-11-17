"""Sparse lane-field supervision from raw visible points, without GT completion."""
import torch
import torch.nn.functional as F


def bev_lane_field_loss(prior, points, visibility, pc_range, sigma=.3):
    """Multi-label positions per supported row, and heights only at GT points.

    Every raw point contributes a lateral Gaussian to its BEV row. Multiple
    lanes are combined by maximum, rather than competing in one softmax.
    Rows with no annotated points have no foreground/background supervision;
    neither LiDAR holes nor unlabelled distant rows become negative labels.
    No fitting, extrapolation, instance clustering or curb pseudo-labels.
    """
    logits, height = prior['lane_field_logits'], prior['lane_height_field']
    batch, _, rows, width = logits.shape
    points = points.to(logits)
    visible = visibility.bool() & torch.isfinite(points).all(-1)
    observed = prior.get('from_lidar')
    if torch.is_tensor(observed):
        visible = visible & observed[:, None, None]
    elif observed is False:
        visible = visible & False
    x0, y0, _, x1, y1, _ = pc_range
    visible = (visible & (points[..., 0] >= x0) & (points[..., 0] <= x1)
               & (points[..., 1] >= y0) & (points[..., 1] <= y1))
    safe = torch.where(visible[..., None], points, 0)
    # Rasterize annotations only for this training objective. The coverage
    # metric continues to use every original point at its exact y coordinate.
    with torch.no_grad():
        column_x = x0 + (torch.arange(width, device=logits.device, dtype=logits.dtype) + .5) * (x1-x0)/width
        raw_x = safe[..., 0].flatten(1)
        gaussian = torch.exp(-.5 * ((column_x - raw_x[..., None]) / sigma).square())
        gaussian *= visible.flatten(1)[..., None]
        row = ((safe[..., 1] - y0) * rows/(y1-y0)).long().clamp(0, rows-1).flatten(1)
        target = logits.new_zeros(batch, rows, width)
        target.scatter_reduce_(1, row[..., None].expand_as(gaussian), gaussian,
                               reduce='amax', include_self=True)
        supported = target.amax(-1) > 0
    if prior.get('lane_field_mode', 'balanced_sigmoid') == 'row_distribution':
        # This proper distributional objective has optimum p = normalized
        # target. Class-balanced soft BCE instead flattens Gaussian peaks when
        # decoded as sigmoid probabilities, weakening geometric localization.
        distribution = target / target.sum(-1, keepdim=True).clamp_min(1e-6)
        row_loss = F.kl_div(logits[:, 0].log_softmax(-1), distribution,
                            reduction='none').sum(-1)
    else:
        positive = -(F.logsigmoid(logits[:, 0]) * target).sum(-1) / target.sum(-1).clamp_min(1e-6)
        negative = -(F.logsigmoid(-logits[:, 0]) * (1-target)).sum(-1) / (1-target).sum(-1).clamp_min(1e-6)
        row_loss = .5 * (positive + negative)
    position_loss = torch.where(supported, row_loss, 0).sum() / supported.sum().clamp_min(1)
    grid = torch.stack((2*(safe[..., 0]-x0)/(x1-x0)-1,
                        2*(safe[..., 1]-y0)/(y1-y0)-1), -1)
    predicted_height = F.grid_sample(height, grid, mode='bilinear',
                                     padding_mode='border', align_corners=False)[:, 0]
    height_error = F.smooth_l1_loss(predicted_height, safe[..., 2], reduction='none', beta=.3)
    count = visible.sum(-1)
    valid_lane = count > 0
    per_lane = (height_error * visible).sum(-1) / count.clamp_min(1)
    height_loss = torch.where(valid_lane, per_lane, 0).sum() / valid_lane.sum().clamp_min(1)
    return dict(loss_bev_lane_position=position_loss, loss_bev_lane_height=height_loss,
                diag_bev_supervised_rows=supported.sum().to(position_loss).detach())
