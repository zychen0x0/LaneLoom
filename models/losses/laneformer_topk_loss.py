"""Compact Top-k classification and regression losses for 3D lanes."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS, TASK_UTILS

from .topk_assigner import LaneFormerTopKAssigner


@MODELS.register_module()
class LaneFormerTopKLaneLoss(nn.Module):
    """Supervise two nearest anchors per GT and classify all others as background."""

    def __init__(self, assigner=None, pos_k=2, cls_loss_weight=1.0,
                 x_loss_weight=1.0, z_loss_weight=1.0,
                 visibility_loss_weight=1.0, aux_loss_weight=0.5,
                 smooth_l1_beta=0.5, focal_alpha=0.25, focal_gamma=2.0,
                 **kwargs):
        super().__init__()
        self.assigner = (
            TASK_UTILS.build(assigner) if assigner is not None
            else LaneFormerTopKAssigner(pos_k=pos_k))
        self.cls_loss_weight = float(cls_loss_weight)
        self.x_loss_weight = float(x_loss_weight)
        self.z_loss_weight = float(z_loss_weight)
        self.visibility_loss_weight = float(visibility_loss_weight)
        self.aux_loss_weight = float(aux_loss_weight)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)

    def _focal_loss(self, logits, target, positive_count):
        probability = logits.softmax(dim=-1)
        target_probability = probability.gather(1, target[:, None]).squeeze(1)
        cross_entropy = F.cross_entropy(logits, target, reduction='none')
        alpha = torch.where(
            target == 0,
            logits.new_tensor(1 - self.focal_alpha),
            logits.new_tensor(self.focal_alpha))
        loss = alpha * (1 - target_probability).pow(
            self.focal_gamma) * cross_entropy
        return loss.sum() / max(positive_count, 1)

    @staticmethod
    def _classification_logits(predictions):
        if 'pred_logits' in predictions:
            return predictions['pred_logits']
        foreground = (
            predictions['pred_category']
            + predictions['pred_cls'].unsqueeze(-1))
        background = foreground.new_zeros(*foreground.shape[:-1], 1)
        return torch.cat((background, foreground), dim=-1)

    def _assign(self, predictions, gt_lanes_3d, gt_visibility):
        anchor_points = predictions.get('anchor_points')
        if anchor_points is None:
            anchor_x = predictions['pred_x'].detach()
            anchor_z = predictions['pred_z'].detach()
        else:
            anchor_x = anchor_points[..., 0].detach()
            anchor_z = anchor_points[..., 2].detach()
        assignments = []
        for batch_idx in range(anchor_x.size(0)):
            assignments.append(self.assigner.assign(
                anchor_x[batch_idx], anchor_z[batch_idx],
                gt_lanes_3d[batch_idx, :, :, 0],
                gt_lanes_3d[batch_idx, :, :, 2],
                gt_visibility[batch_idx]))
        return assignments

    def _prediction_losses(self, predictions, gt_lanes_3d,
                           gt_visibility, gt_category, assignments):
        pred_x, pred_z = predictions['pred_x'], predictions['pred_z']
        pred_visibility = predictions['pred_vis']
        pred_logits = self._classification_logits(predictions)
        total_cls = pred_logits.sum() * 0
        total_x = pred_x.sum() * 0
        total_z = pred_z.sum() * 0
        total_visibility = pred_visibility.sum() * 0
        positive_count = sum(int((assignment >= 0).sum())
                             for assignment in assignments)
        visible_point_count = 0

        for batch_idx, assignment in enumerate(assignments):
            positive_queries = torch.where(assignment >= 0)[0]
            batch_positive_count = int(positive_queries.numel())
            class_target = torch.zeros(
                pred_logits.size(1), dtype=torch.long,
                device=pred_logits.device)
            if positive_queries.numel() > 0:
                matched_gt = assignment[positive_queries]
                class_target[positive_queries] = (
                    gt_category[batch_idx, matched_gt].long() + 1)
                visible = gt_visibility[batch_idx, matched_gt].to(pred_x.dtype)
                visible_point_count += int(visible.sum())
                target_x = gt_lanes_3d[batch_idx, matched_gt, :, 0]
                target_z = gt_lanes_3d[batch_idx, matched_gt, :, 2]
                total_x = total_x + (
                    F.smooth_l1_loss(
                        pred_x[batch_idx, positive_queries], target_x,
                        beta=self.smooth_l1_beta, reduction='none')
                    * visible).sum()
                total_z = total_z + (
                    F.smooth_l1_loss(
                        pred_z[batch_idx, positive_queries], target_z,
                        beta=self.smooth_l1_beta, reduction='none')
                    * visible).sum()
                total_visibility = total_visibility + F.smooth_l1_loss(
                    pred_visibility[batch_idx, positive_queries].sigmoid(),
                    visible, beta=self.smooth_l1_beta, reduction='sum')
            total_cls = total_cls + self._focal_loss(
                pred_logits[batch_idx], class_target, batch_positive_count)

        batch_size = max(pred_x.size(0), 1)
        visibility_count = max(positive_count * pred_x.size(-1), 1)
        return dict(
            loss_cls=total_cls / batch_size * self.cls_loss_weight,
            loss_x=(total_x / max(visible_point_count, 1)
                    * self.x_loss_weight),
            loss_z=(total_z / max(visible_point_count, 1)
                    * self.z_loss_weight),
            loss_visibility=(total_visibility / visibility_count
                             * self.visibility_loss_weight))

    def _auxiliary_assignments(self, predictions, assignments,
                               gt_lanes_3d, gt_visibility):
        return assignments

    def forward(self, predictions, gt_lanes_3d, gt_lanes_visibility,
                gt_lanes_category, gt_lanes_3d_enhanced=None,
                gt_lanes_visibility_enhanced=None,
                gt_lanes_category_enhanced=None, **kwargs):
        if (gt_lanes_3d_enhanced is not None
                and gt_lanes_visibility_enhanced is not None):
            gt_lanes_3d = gt_lanes_3d_enhanced
            gt_lanes_visibility = gt_lanes_visibility_enhanced
            if gt_lanes_category_enhanced is not None:
                gt_lanes_category = gt_lanes_category_enhanced

        assignments = self._assign(
            predictions, gt_lanes_3d, gt_lanes_visibility)
        losses = self._prediction_losses(
            predictions, gt_lanes_3d, gt_lanes_visibility,
            gt_lanes_category, assignments)
        for layer_idx, auxiliary in enumerate(
                predictions.get('aux_predictions', ())):
            auxiliary_assignments = self._auxiliary_assignments(
                auxiliary, assignments,
                gt_lanes_3d, gt_lanes_visibility)
            auxiliary_losses = self._prediction_losses(
                auxiliary, gt_lanes_3d, gt_lanes_visibility,
                gt_lanes_category, auxiliary_assignments)
            for name, value in auxiliary_losses.items():
                losses[f'loss_aux_{layer_idx}_{name.removeprefix("loss_")}'] = (
                    value * self.aux_loss_weight)
        return losses


@MODELS.register_module()
class LaneFormerOneToOneLaneLoss(LaneFormerTopKLaneLoss):
    """Keep the compact lane objectives with Hungarian one-to-one matching."""

    def __init__(self, assigner=None, **kwargs):
        if assigner is None:
            assigner = dict(type='LaneFormerHungarianAssigner')
        super().__init__(assigner=assigner, **kwargs)

    def _assign(self, predictions, gt_lanes_3d, gt_visibility):
        final_predictions = dict(predictions)
        final_predictions.pop('anchor_points', None)
        anchor_x = final_predictions['pred_x'].detach()
        anchor_z = final_predictions['pred_z'].detach()
        assignments = self.assigner.assign_batch(
            anchor_x, anchor_z,
            gt_lanes_3d[..., 0], gt_lanes_3d[..., 2],
            gt_visibility)
        return list(assignments.unbind(dim=0))

    def _auxiliary_assignments(self, predictions, assignments,
                               gt_lanes_3d, gt_visibility):
        return self._assign(predictions, gt_lanes_3d, gt_visibility)
