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
class LaneLoomHungarianLaneLoss(nn.Module):
    """Cascade lane supervision after Hungarian one-to-one matching.

    Objectness distinguishes every query from background.  The remaining
    attributes are instance properties, so they are supervised only on the
    query matched to a visible GT lane.
    """

    def __init__(self, geometry_loss_weight=20., visibility_loss_weight=1.,
                 visibility_dice_loss_weight=1.,
                 visibility_boundary_loss_weight=.25,
                 objectness_loss_weight=10., category_loss_weight=.25,
                 dn_loss_weight=1.,
                 geometry_huber_beta=.5, geometry_normalization=1.,
                 match_cost_geometry=1., match_cost_vis=.5,
                 match_cost_vis_dice=1.,
                 match_cost_objectness=1.,
                 memory_loss_weight=1., focal_alpha=.25, focal_gamma=2.,
                 aux_loss_weight=.5,
                 **kwargs):
        super().__init__()
        self.geometry_loss_weight = geometry_loss_weight
        self.visibility_loss_weight = visibility_loss_weight
        self.visibility_dice_loss_weight = visibility_dice_loss_weight
        self.visibility_boundary_loss_weight = visibility_boundary_loss_weight
        self.objectness_loss_weight = objectness_loss_weight
        self.category_loss_weight = category_loss_weight
        self.dn_loss_weight = dn_loss_weight
        self.geometry_huber_beta = geometry_huber_beta
        self.geometry_normalization = float(geometry_normalization)
        if self.geometry_normalization <= 0:
            raise ValueError('geometry_normalization must be positive.')
        self.match_cost_geometry = match_cost_geometry
        self.match_cost_vis = match_cost_vis
        self.match_cost_vis_dice = match_cost_vis_dice
        self.match_cost_objectness = match_cost_objectness
        self.memory_loss_weight = memory_loss_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.aux_loss_weight = aux_loss_weight

    def _match(self, pred_x, pred_z, pred_vis, pred_objectness,
               gt_x, gt_z, gt_vis):
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
        vis_probability = pred_vis.sigmoid()[:, None]
        vis_targets = gt_vis.to(dtype=pred_vis.dtype)[None]
        vis_intersection = (vis_probability * vis_targets).sum(-1)
        vis_dice_cost = 1. - (2. * vis_intersection + 1.) / (
            vis_probability.sum(-1) + vis_targets.sum(-1) + 1.)
        objectness_cost = F.binary_cross_entropy_with_logits(
            pred_objectness[:, None].expand(-1, gt_vis.size(0)),
            torch.ones((pred_objectness.size(0), gt_vis.size(0)),
                       dtype=pred_objectness.dtype,
                       device=pred_objectness.device), reduction='none')
        cost = (self.match_cost_geometry * geometry_cost
                + self.match_cost_vis * vis_cost
                + self.match_cost_vis_dice * vis_dice_cost
                + self.match_cost_objectness * objectness_cost)
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
        loss = alpha * (1 - probability_target).pow(self.focal_gamma) * bce
        # Normalize by lanes rather than all queries.  Otherwise adding
        # background queries weakens every positive gradient.
        return loss.sum() / targets.sum().clamp_min(1.)

    @staticmethod
    def _visibility_loss_terms(pred_logits, target, pred_start=None,
                               pred_end=None):
        target = target.to(dtype=pred_logits.dtype)
        visible = target.bool()
        bce_terms = []
        if visible.any():
            bce_terms.append(F.binary_cross_entropy_with_logits(
                pred_logits[visible], torch.ones_like(pred_logits[visible])))
        invisible = ~visible
        if invisible.any():
            bce_terms.append(F.binary_cross_entropy_with_logits(
                pred_logits[invisible], torch.zeros_like(pred_logits[invisible])))
        bce = torch.stack(bce_terms).mean()

        probability = pred_logits.sigmoid()
        intersection = (probability * target).sum()
        dice = 1. - (2. * intersection + 1.) / (
            probability.sum() + target.sum() + 1.)

        boundary = pred_logits.sum() * 0
        if pred_start is not None and pred_end is not None and visible.any():
            visible_indices = torch.where(visible)[0].to(pred_logits.dtype)
            scale = max(pred_logits.numel() - 1, 1)
            predicted = torch.stack((pred_start, pred_end)) / scale
            expected = torch.stack((visible_indices[0], visible_indices[-1])) / scale
            boundary = F.smooth_l1_loss(
                predicted, expected, beta=.1)
        return bce, dice, boundary

    def _dn_losses(self, predictions, gt_lanes_3d, gt_visibility,
                   gt_lanes_category, gt_lanes_memory_mask):
        """Directly supervise noisy GT queries without Hungarian matching."""
        dn_predictions = predictions['dn_predictions']
        pred_x, pred_z = dn_predictions['pred_x'], dn_predictions['pred_z']
        pred_vis = dn_predictions['pred_vis']
        pred_vis_start = dn_predictions.get('pred_vis_start')
        pred_vis_end = dn_predictions.get('pred_vis_end')
        pred_objectness = dn_predictions['pred_cls']
        pred_category = dn_predictions['pred_category']
        valid_mask = predictions['dn_valid_mask'].bool()
        target_indices = predictions['dn_target_indices'].long()
        total_geometry = (pred_x.sum() + pred_z.sum()) * 0
        total_visibility = pred_vis.sum() * 0
        total_visibility_dice = pred_vis.sum() * 0
        total_visibility_boundary = pred_vis.sum() * 0
        total_objectness = pred_objectness.sum() * 0
        total_category = pred_category.sum() * 0
        count = 0
        for batch_idx in range(pred_x.size(0)):
            valid_queries = valid_mask[batch_idx]
            total_objectness = total_objectness + self._focal_loss(
                pred_objectness[batch_idx],
                valid_queries.to(dtype=pred_objectness.dtype))
            for query_idx, target_idx in enumerate(target_indices.tolist()):
                if not valid_queries[query_idx]:
                    continue
                valid = gt_visibility[batch_idx, target_idx].bool()
                weights = valid.float()
                if gt_lanes_memory_mask is not None:
                    memory = gt_lanes_memory_mask[batch_idx, target_idx].to(weights.dtype)
                    weights = weights * (1. + (self.memory_loss_weight - 1.) * memory)
                distance = torch.linalg.vector_norm(torch.stack((
                    pred_x[batch_idx, query_idx] - gt_lanes_3d[batch_idx, target_idx, :, 0],
                    pred_z[batch_idx, query_idx] - gt_lanes_3d[batch_idx, target_idx, :, 2]),
                    dim=-1), dim=-1) / self.geometry_normalization
                total_geometry = total_geometry + (
                    F.smooth_l1_loss(
                        distance, torch.zeros_like(distance), reduction='none',
                        beta=(self.geometry_huber_beta /
                              self.geometry_normalization))
                    * weights).sum() / weights.sum().clamp_min(1.)
                visibility, dice, boundary = self._visibility_loss_terms(
                    pred_vis[batch_idx, query_idx], valid,
                    (pred_vis_start[batch_idx, query_idx]
                     if pred_vis_start is not None else None),
                    (pred_vis_end[batch_idx, query_idx]
                     if pred_vis_end is not None else None))
                total_visibility = total_visibility + visibility
                total_visibility_dice = total_visibility_dice + dice
                total_visibility_boundary = total_visibility_boundary + boundary
                if gt_lanes_category is not None:
                    total_category = total_category + F.cross_entropy(
                        pred_category[batch_idx, query_idx].unsqueeze(0),
                        gt_lanes_category[batch_idx, target_idx].long().reshape(1))
                count += 1
        scale = self.dn_loss_weight / max(count, 1)
        return dict(
            loss_dn_geometry=total_geometry * self.geometry_loss_weight * scale,
            loss_dn_visibility=total_visibility * self.visibility_loss_weight * scale,
            loss_dn_visibility_dice=(total_visibility_dice
                                     * self.visibility_dice_loss_weight * scale),
            loss_dn_visibility_boundary=(
                total_visibility_boundary
                * self.visibility_boundary_loss_weight * scale),
            loss_dn_objectness=(total_objectness / max(pred_x.size(0), 1)
                                * self.objectness_loss_weight * self.dn_loss_weight),
            loss_dn_category=total_category * self.category_loss_weight * scale)

    def forward(self, predictions, gt_lanes_3d, gt_lanes_visibility,
                gt_lanes_category=None,
                gt_lanes_3d_enhanced=None, gt_lanes_visibility_enhanced=None,
                gt_lanes_category_enhanced=None,
                gt_lanes_memory_mask=None, **kwargs):
        if gt_lanes_3d_enhanced is not None and gt_lanes_visibility_enhanced is not None:
            gt_lanes_3d, gt_lanes_visibility = (
                gt_lanes_3d_enhanced, gt_lanes_visibility_enhanced)
            if gt_lanes_category_enhanced is not None:
                gt_lanes_category = gt_lanes_category_enhanced
        losses = self._prediction_losses(
            predictions, gt_lanes_3d, gt_lanes_visibility,
            gt_lanes_category, gt_lanes_memory_mask)
        for layer_idx, auxiliary in enumerate(
                predictions.get('aux_predictions', ())):
            auxiliary_losses = self._prediction_losses(
                auxiliary, gt_lanes_3d, gt_lanes_visibility,
                gt_lanes_category, gt_lanes_memory_mask)
            for name, value in auxiliary_losses.items():
                suffix = name.removeprefix('loss_')
                losses[f'loss_aux_{layer_idx}_{suffix}'] = (
                    value * self.aux_loss_weight)
        return losses

    def _prediction_losses(self, predictions, gt_lanes_3d,
                           gt_lanes_visibility, gt_lanes_category,
                           gt_lanes_memory_mask):
        pred_x, pred_z = predictions['pred_x'], predictions['pred_z']
        pred_vis, pred_objectness = predictions['pred_vis'], predictions['pred_cls']
        pred_vis_start = predictions.get('pred_vis_start')
        pred_vis_end = predictions.get('pred_vis_end')
        pred_category = predictions['pred_category']
        # Keep every prediction branch in the graph even for batches without
        # a visible/matchable lane; DDP requires the same parameters each step.
        total_geometry = (pred_x.sum() + pred_z.sum()) * 0
        total_visibility = pred_vis.sum() * 0
        total_visibility_dice = pred_vis.sum() * 0
        total_visibility_boundary = pred_vis.sum() * 0
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
                pred_objectness[batch_idx].detach(),
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
                    pred_z[batch_idx, query_idx] - gt_z[gt_idx]), dim=-1),
                    dim=-1) / self.geometry_normalization
                total_geometry = total_geometry + (
                    F.smooth_l1_loss(
                        distance, torch.zeros_like(distance), reduction='none',
                        beta=(self.geometry_huber_beta /
                              self.geometry_normalization))
                    * weights).sum() / weight_sum

                # A lane's visible and invisible anchors carry equal total
                # weight.  Background queries intentionally have no visibility
                # loss: objectness alone supervises their lane/background role.
                visibility, dice, boundary = self._visibility_loss_terms(
                    pred_vis[batch_idx, query_idx], valid,
                    (pred_vis_start[batch_idx, query_idx]
                     if pred_vis_start is not None else None),
                    (pred_vis_end[batch_idx, query_idx]
                     if pred_vis_end is not None else None))
                total_visibility = total_visibility + visibility
                total_visibility_dice = total_visibility_dice + dice
                total_visibility_boundary = total_visibility_boundary + boundary
                if gt_lanes_category is not None:
                    total_category = total_category + F.cross_entropy(
                        pred_category[batch_idx, query_idx].unsqueeze(0),
                        gt_lanes_category[batch_idx, gt_idx].long().reshape(1))
                match_count += 1
            total_objectness = total_objectness + self._focal_loss(
                pred_objectness[batch_idx], objectness_target)
        losses = dict(
            loss_geometry=(total_geometry / max(match_count, 1)
                           * self.geometry_loss_weight),
            loss_visibility=(total_visibility / max(match_count, 1)
                             * self.visibility_loss_weight),
            loss_visibility_dice=(total_visibility_dice / max(match_count, 1)
                                  * self.visibility_dice_loss_weight),
            loss_visibility_boundary=(
                total_visibility_boundary / max(match_count, 1)
                * self.visibility_boundary_loss_weight),
            loss_objectness=(total_objectness / max(pred_x.size(0), 1)
                             * self.objectness_loss_weight),
            loss_category=(total_category / max(match_count, 1)
                           * self.category_loss_weight))
        if 'dn_predictions' in predictions:
            losses.update(self._dn_losses(
                predictions, gt_lanes_3d, gt_lanes_visibility,
                gt_lanes_category, gt_lanes_memory_mask))
        return losses
