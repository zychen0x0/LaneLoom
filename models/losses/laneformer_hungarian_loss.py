"""Hungarian assignment and losses for fixed-anchor 3D lanes."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


@MODELS.register_module()
class LaneFormerHungarianLaneLoss(nn.Module):
    """Cascade lane supervision after Hungarian one-to-one matching.

    Objectness distinguishes every query from background.  The remaining
    attributes are instance properties, so they are supervised only on the
    query matched to a visible GT lane.
    """

    def __init__(self, geometry_loss_weight=20., visibility_loss_weight=1.,
                 objectness_loss_weight=10., category_loss_weight=.25,
                 geometry_huber_beta=.5,
                 match_cost_geometry=1., match_cost_vis=.5,
                 memory_loss_weight=1., focal_alpha=.25, focal_gamma=2.,
                 **kwargs):
        super().__init__()
        self.geometry_loss_weight = geometry_loss_weight
        self.visibility_loss_weight = visibility_loss_weight
        self.objectness_loss_weight = objectness_loss_weight
        self.category_loss_weight = category_loss_weight
        self.geometry_huber_beta = geometry_huber_beta
        self.match_cost_geometry = match_cost_geometry
        self.match_cost_vis = match_cost_vis
        self.memory_loss_weight = memory_loss_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def _match(self, pred_x, pred_z, pred_vis, gt_x, gt_z, gt_vis):
        valid_gt = torch.where(gt_vis.any(dim=-1))[0]
        if valid_gt.numel() == 0:
            return []
        gt_x, gt_z, gt_vis = gt_x[valid_gt], gt_z[valid_gt], gt_vis[valid_gt]
        mask = gt_vis.bool().unsqueeze(0)
        count = mask.sum(-1).clamp_min(1)
        geometry_cost = (
            torch.linalg.vector_norm(torch.stack((
                pred_x[:, None] - gt_x[None], pred_z[:, None] - gt_z[None]),
                dim=-1), dim=-1) * mask).sum(-1) / count
        vis_cost = F.binary_cross_entropy_with_logits(
            pred_vis[:, None].expand(-1, gt_vis.size(0), -1),
            gt_vis[None].expand(pred_vis.size(0), -1, -1), reduction='none').mean(-1)
        # Assignment is purely geometric/visibility-based.  Query objectness
        # is supervised only after the one-to-one geometry assignment.
        cost = (self.match_cost_geometry * geometry_cost
                + self.match_cost_vis * vis_cost)
        if linear_sum_assignment is None:
            pairs, available = [], set(range(pred_x.size(0)))
            for local_gt in range(valid_gt.numel()):
                query = min(available, key=lambda index: float(cost[index, local_gt]))
                available.remove(query)
                pairs.append((query, int(valid_gt[local_gt])))
            return pairs
        query_indices, local_gt_indices = linear_sum_assignment(cost.detach().cpu().numpy())
        return [(int(query), int(valid_gt[local_gt]))
                for query, local_gt in zip(query_indices, local_gt_indices)]

    def _focal_loss(self, logits, targets):
        probability = logits.sigmoid()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probability_target = probability * targets + (1 - probability) * (1 - targets)
        alpha = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        return (alpha * (1 - probability_target).pow(self.focal_gamma) * bce).mean()

    def forward(self, predictions, gt_lanes_3d, gt_lanes_visibility,
                gt_lanes_category=None,
                gt_lanes_3d_enhanced=None, gt_lanes_visibility_enhanced=None,
                gt_lanes_memory_mask=None, **kwargs):
        if gt_lanes_3d_enhanced is not None and gt_lanes_visibility_enhanced is not None:
            gt_lanes_3d, gt_lanes_visibility = (
                gt_lanes_3d_enhanced, gt_lanes_visibility_enhanced)
        pred_x, pred_z = predictions['pred_x'], predictions['pred_z']
        pred_vis, pred_objectness = predictions['pred_vis'], predictions['pred_cls']
        pred_category = predictions['pred_category']
        # Keep every prediction branch in the graph even for batches without
        # a visible/matchable lane; DDP requires the same parameters each step.
        total_geometry = (pred_x.sum() + pred_z.sum()) * 0
        total_visibility = pred_vis.sum() * 0
        total_objectness = pred_objectness.sum() * 0
        total_category = pred_category.sum() * 0
        match_count = 0
        for batch_idx in range(pred_x.size(0)):
            gt_x = gt_lanes_3d[batch_idx, :, :, 0]
            gt_z = gt_lanes_3d[batch_idx, :, :, 2]
            gt_vis = gt_lanes_visibility[batch_idx]
            pairs = self._match(
                pred_x[batch_idx].detach(), pred_z[batch_idx].detach(),
                pred_vis[batch_idx].detach(),
                gt_x, gt_z, gt_vis)
            objectness_target = torch.zeros_like(pred_objectness[batch_idx])
            for query_idx, gt_idx in pairs:
                valid = gt_vis[gt_idx].bool()
                objectness_target[query_idx] = 1.
                weights = valid.float()
                if gt_lanes_memory_mask is not None:
                    memory = gt_lanes_memory_mask[batch_idx, gt_idx].to(weights.dtype)
                    weights = weights * (1. + (self.memory_loss_weight - 1.) * memory)
                weight_sum = weights.sum().clamp_min(1.)
                distance = torch.linalg.vector_norm(torch.stack((
                    pred_x[batch_idx, query_idx] - gt_x[gt_idx],
                    pred_z[batch_idx, query_idx] - gt_z[gt_idx]), dim=-1), dim=-1)
                total_geometry = total_geometry + (
                    F.smooth_l1_loss(
                        distance, torch.zeros_like(distance), reduction='none',
                        beta=self.geometry_huber_beta) * weights).sum() / weight_sum

                # A lane's visible and invisible anchors carry equal total
                # weight.  Background queries intentionally have no visibility
                # loss: objectness alone supervises their lane/background role.
                visibility_terms = []
                if valid.any():
                    visibility_terms.append(F.binary_cross_entropy_with_logits(
                        pred_vis[batch_idx, query_idx][valid],
                        torch.ones_like(pred_vis[batch_idx, query_idx][valid])))
                invisible = ~valid
                if invisible.any():
                    visibility_terms.append(F.binary_cross_entropy_with_logits(
                        pred_vis[batch_idx, query_idx][invisible],
                        torch.zeros_like(pred_vis[batch_idx, query_idx][invisible])))
                total_visibility = total_visibility + torch.stack(visibility_terms).mean()
                if gt_lanes_category is not None:
                    total_category = total_category + F.cross_entropy(
                        pred_category[batch_idx, query_idx].unsqueeze(0),
                        gt_lanes_category[batch_idx, gt_idx].long().reshape(1))
                match_count += 1
            total_objectness = total_objectness + self._focal_loss(
                pred_objectness[batch_idx], objectness_target)
        return dict(
            loss_geometry=(total_geometry / max(match_count, 1)
                           * self.geometry_loss_weight),
            loss_visibility=(total_visibility / max(match_count, 1)
                             * self.visibility_loss_weight),
            loss_objectness=(total_objectness / max(pred_x.size(0), 1)
                             * self.objectness_loss_weight),
            loss_category=(total_category / max(match_count, 1)
                           * self.category_loss_weight))
