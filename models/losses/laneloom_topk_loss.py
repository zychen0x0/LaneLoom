"""Compact classification and regression losses for 3D lanes."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS, TASK_UTILS

from .topk_assigner import TopkAssigner


class LaneLossBase(nn.Module):
    """Shared lane classification, regression, and diagnostic objectives."""

    def __init__(self, assigner=None, pos_k=2, cls_loss_weight=1.0,
                 x_loss_weight=1.0, z_loss_weight=1.0,
                 visibility_loss_weight=1.0, aux_loss_weight=0.5,
                 smooth_l1_beta=0.5, focal_alpha=0.25, focal_gamma=2.0,
                 emit_diagnostics=False, diagnostic_score_threshold=.3,
                 diagnostic_visibility_threshold=.5, **kwargs):
        super().__init__()
        self.assigner = (
            TASK_UTILS.build(assigner) if assigner is not None
            else TopkAssigner(pos_k=pos_k))
        self.cls_loss_weight = float(cls_loss_weight)
        self.x_loss_weight = float(x_loss_weight)
        self.z_loss_weight = float(z_loss_weight)
        self.visibility_loss_weight = float(visibility_loss_weight)
        self.aux_loss_weight = float(aux_loss_weight)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.emit_diagnostics = bool(emit_diagnostics)
        self.diagnostic_score_threshold = float(diagnostic_score_threshold)
        self.diagnostic_visibility_threshold = float(
            diagnostic_visibility_threshold)

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

    def _assignment_diagnostics(self, predictions, assignments,
                                gt_visibility):
        reference = predictions['pred_cls']
        matched = 0
        unique_matched = 0
        for assignment in assignments:
            matched_gt = assignment[assignment >= 0]
            matched += int(matched_gt.numel())
            unique_matched += int(torch.unique(matched_gt).numel())
        valid_gt = int((gt_visibility > .5).any(dim=-1).sum())
        foreground_probability = reference.sigmoid()
        objectness_queries = int((
            foreground_probability >= self.diagnostic_score_threshold).sum())
        scores = foreground_probability
        visible = (
            predictions['pred_vis'].sigmoid()
            >= self.diagnostic_visibility_threshold).any(dim=-1)
        predicted = int((
            (scores >= self.diagnostic_score_threshold) & visible).sum())
        batch = max(reference.size(0), 1)
        queries = max(reference.numel(), 1)
        diagnostics = dict(
            diag_gt_match_recall=reference.new_tensor(
                unique_matched / max(valid_gt, 1)),
            diag_unmatched_gt=reference.new_tensor(
                max(valid_gt - unique_matched, 0) / batch),
            diag_duplicate_gt_assignments=reference.new_tensor(
                max(matched - unique_matched, 0) / batch),
            diag_positive_queries=reference.new_tensor(matched / batch),
            diag_positive_query_ratio=reference.new_tensor(
                matched / queries),
            diag_objectness_probability=foreground_probability.mean(),
            diag_objectness_queries=reference.new_tensor(
                objectness_queries / batch),
            diag_prediction_queries=reference.new_tensor(
                predicted / batch),
            diag_pred_gt_ratio=reference.new_tensor(
                predicted / max(valid_gt, 1)),
        )
        for modality in ('image', 'lidar'):
            name = f'diag_{modality}_valid_ratio'
            if name in predictions:
                diagnostics[name] = predictions[name]
        return diagnostics

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
        if self.emit_diagnostics:
            losses.update(self._assignment_diagnostics(
                predictions, assignments, gt_lanes_visibility))
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
            if self.emit_diagnostics:
                auxiliary_diagnostics = self._assignment_diagnostics(
                    auxiliary, auxiliary_assignments, gt_lanes_visibility)
                for name, value in auxiliary_diagnostics.items():
                    losses[
                        f'diag_aux_{layer_idx}_{name.removeprefix("diag_")}'] = (
                            value)
        topk_indices = predictions.get('layer0_topk_indices')
        if (self.emit_diagnostics and topk_indices is not None
                and predictions.get('aux_predictions')):
            auxiliary = predictions['aux_predictions'][0]
            selected_x = auxiliary['pred_x'].gather(
                1, topk_indices.unsqueeze(-1).expand(
                    -1, -1, auxiliary['pred_x'].size(-1)))
            selected_z = auxiliary['pred_z'].gather(
                1, topk_indices.unsqueeze(-1).expand(
                    -1, -1, auxiliary['pred_z'].size(-1)))
            visible = gt_lanes_visibility > .5
            valid_gt = visible.sum(dim=-1) >= 2
            distance = torch.linalg.vector_norm(torch.stack((
                selected_x.unsqueeze(2) - gt_lanes_3d[
                    ..., 0].unsqueeze(1),
                selected_z.unsqueeze(2) - gt_lanes_3d[
                    ..., 2].unsqueeze(1)), dim=-1), dim=-1)
            close_ratio = (
                ((distance < self.diagnostic_geometry_distance_threshold)
                 & visible.unsqueeze(1)).sum(dim=-1)
                / visible.sum(dim=-1).unsqueeze(1).clamp_min(1))
            covered = (
                close_ratio >= self.diagnostic_geometry_match_ratio
            ).any(dim=1) & valid_gt
            losses['diag_layer0_topk_gt_recall'] = (
                covered.sum().to(predictions['pred_cls'].dtype)
                / valid_gt.sum().clamp_min(1))
        return losses


@MODELS.register_module()
class TopkLaneLoss(LaneLossBase):
    """Supervise nearest anchors per GT and classify others as background."""


@MODELS.register_module()
class LaneLoss(LaneLossBase):
    """Keep the compact lane objectives with Hungarian one-to-one matching."""

    def __init__(self, assigner=None, objectness_loss_weight=1.0,
                 category_loss_weight=.25,
                 negative_objectness_weight=.1,
                 ambiguous_negative_weight=.1,
                 ambiguous_distance_threshold=.5,
                 ambiguous_match_ratio=.75,
                 diagnostic_geometry_distance_threshold=.3,
                 diagnostic_geometry_match_ratio=.8,
                 **kwargs):
        if assigner is None:
            assigner = dict(type='HungarianAssigner')
        self.objectness_loss_weight = float(objectness_loss_weight)
        self.category_loss_weight = float(category_loss_weight)
        self.negative_objectness_weight = float(negative_objectness_weight)
        self.ambiguous_negative_weight = float(ambiguous_negative_weight)
        self.ambiguous_distance_threshold = float(
            ambiguous_distance_threshold)
        self.ambiguous_match_ratio = float(ambiguous_match_ratio)
        self.diagnostic_geometry_distance_threshold = float(
            diagnostic_geometry_distance_threshold)
        self.diagnostic_geometry_match_ratio = float(
            diagnostic_geometry_match_ratio)
        if min(
                self.objectness_loss_weight,
                self.category_loss_weight,
                self.negative_objectness_weight,
                self.ambiguous_negative_weight) < 0:
            raise ValueError('Lane loss weights must be non-negative.')
        if self.ambiguous_distance_threshold <= 0:
            raise ValueError(
                'ambiguous_distance_threshold must be positive.')
        if self.diagnostic_geometry_distance_threshold <= 0:
            raise ValueError(
                'diagnostic_geometry_distance_threshold must be positive.')
        if not 0 < self.ambiguous_match_ratio <= 1:
            raise ValueError('ambiguous_match_ratio must be in (0, 1].')
        if not 0 < self.diagnostic_geometry_match_ratio <= 1:
            raise ValueError(
                'diagnostic_geometry_match_ratio must be in (0, 1].')
        super().__init__(
            assigner=assigner, emit_diagnostics=True, **kwargs)

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

    def required_query_mask(
            self, anchor_points, gt_lanes_3d, gt_visibility):
        assignments = self.assigner.assign_batch(
            anchor_points[..., 0].detach(),
            anchor_points[..., 2].detach(),
            gt_lanes_3d[..., 0], gt_lanes_3d[..., 2],
            gt_visibility)
        return assignments >= 0

    def required_prediction_mask(
            self, predictions, gt_lanes_3d, gt_visibility):
        assignments = self.assigner.assign_batch(
            predictions['pred_x'].detach(),
            predictions['pred_z'].detach(),
            gt_lanes_3d[..., 0], gt_lanes_3d[..., 2],
            gt_visibility)
        return assignments >= 0

    def _binary_focal_terms(self, logits, targets):
        probability = logits.sigmoid()
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        target_probability = torch.where(
            targets > .5, probability, 1 - probability)
        alpha = torch.where(
            targets > .5,
            logits.new_tensor(self.focal_alpha),
            logits.new_tensor(1 - self.focal_alpha))
        return alpha * (1 - target_probability).pow(
            self.focal_gamma) * bce

    def _ambiguous_negative_mask(
            self, predictions, gt_lanes_3d, gt_visibility):
        visible = gt_visibility > .5
        valid_gt = visible.sum(dim=-1) >= 2
        distance = torch.linalg.vector_norm(torch.stack((
            predictions['pred_x'].unsqueeze(2) - gt_lanes_3d[
                ..., 0].unsqueeze(1),
            predictions['pred_z'].unsqueeze(2) - gt_lanes_3d[
                ..., 2].unsqueeze(1)), dim=-1), dim=-1)
        close_ratio = (
            ((distance < self.ambiguous_distance_threshold)
             & visible.unsqueeze(1)).sum(dim=-1)
            / visible.sum(dim=-1).unsqueeze(1).clamp_min(1))
        return (
            (close_ratio >= self.ambiguous_match_ratio)
            & valid_gt.unsqueeze(1)).any(dim=-1)

    def _prediction_losses(self, predictions, gt_lanes_3d,
                           gt_visibility, gt_category, assignments):
        pred_x, pred_z = predictions['pred_x'], predictions['pred_z']
        pred_visibility = predictions['pred_vis']
        pred_objectness = predictions['pred_cls']
        pred_category = predictions['pred_category']
        total_objectness = pred_objectness.sum() * 0
        total_category = pred_category.sum() * 0
        total_x = pred_x.sum() * 0
        total_z = pred_z.sum() * 0
        total_visibility = pred_visibility.sum() * 0
        positive_count = 0
        visible_point_count = 0
        ambiguous = self._ambiguous_negative_mask(
            predictions, gt_lanes_3d, gt_visibility)

        for batch_idx, assignment in enumerate(assignments):
            positive_queries = torch.where(assignment >= 0)[0]
            positive = torch.zeros(
                pred_objectness.size(1), dtype=torch.bool,
                device=pred_objectness.device)
            positive[positive_queries] = True
            objectness_target = positive.to(pred_objectness.dtype)
            focal = self._binary_focal_terms(
                pred_objectness[batch_idx], objectness_target)
            positive_loss = torch.where(positive, focal, 0).sum()
            positive_loss = positive_loss / positive.sum().clamp_min(1)
            negative = ~positive
            negative_weight = torch.where(
                ambiguous[batch_idx] & negative,
                focal.new_tensor(self.ambiguous_negative_weight),
                focal.new_tensor(1.0))
            negative_loss = torch.where(
                negative, focal * negative_weight, 0).sum()
            negative_loss = (
                negative_loss / negative.sum().clamp_min(1))
            total_objectness = total_objectness + (
                positive_loss
                + self.negative_objectness_weight * negative_loss)

            if positive_queries.numel() == 0:
                continue
            matched_gt = assignment[positive_queries]
            positive_count += int(positive_queries.numel())
            total_category = total_category + F.cross_entropy(
                pred_category[batch_idx, positive_queries],
                gt_category[batch_idx, matched_gt].long(),
                reduction='sum')
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

        batch_size = max(pred_x.size(0), 1)
        visibility_count = max(positive_count * pred_x.size(-1), 1)
        return dict(
            loss_objectness=(
                total_objectness / batch_size
                * self.objectness_loss_weight),
            loss_category=(
                total_category / max(positive_count, 1)
                * self.category_loss_weight),
            loss_x=(total_x / max(visible_point_count, 1)
                    * self.x_loss_weight),
            loss_z=(total_z / max(visible_point_count, 1)
                    * self.z_loss_weight),
            loss_visibility=(total_visibility / visibility_count
                             * self.visibility_loss_weight))
