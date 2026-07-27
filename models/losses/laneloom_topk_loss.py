"""Compact classification and regression losses for 3D lanes."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS, TASK_UTILS

from ..valid_range_tube_iou import valid_range_tube_iou
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
                           gt_visibility, gt_category, assignments,
                           is_auxiliary=False):
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
                                gt_lanes_3d, gt_visibility):
        reference = predictions['pred_cls']
        matched = 0
        unique_matched = 0
        positive_scores = []
        negative_scores = []
        lane_scores = reference.sigmoid()
        for batch_idx, assignment in enumerate(assignments):
            matched_gt = assignment[assignment >= 0]
            matched += int(matched_gt.numel())
            unique_matched += int(torch.unique(matched_gt).numel())
            positive = assignment >= 0
            if positive.any():
                positive_scores.append(lane_scores[batch_idx, positive])
            if (~positive).any():
                negative_scores.append(lane_scores[batch_idx, ~positive])
        valid_gt = int((gt_visibility > .5).any(dim=-1).sum())
        lane_score_queries = int((
            lane_scores >= self.diagnostic_score_threshold).sum())
        scores = lane_scores
        visible = (
            predictions['pred_vis'].sigmoid()
            >= self.diagnostic_visibility_threshold).any(dim=-1)
        predicted = int((
            (scores >= self.diagnostic_score_threshold) & visible).sum())
        batch = max(reference.size(0), 1)
        queries = max(reference.numel(), 1)
        zero = reference.new_tensor(0.0)
        positive_lane_score = (
            torch.cat(positive_scores).mean()
            if positive_scores else zero)
        negative_lane_score = (
            torch.cat(negative_scores).mean()
            if negative_scores else zero)
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
            diag_lane_score_probability=lane_scores.mean(),
            diag_lane_score_queries=reference.new_tensor(
                lane_score_queries / batch),
            diag_positive_lane_score=positive_lane_score,
            diag_negative_lane_score=negative_lane_score,
            diag_lane_score_margin=(
                positive_lane_score - negative_lane_score),
            diag_prediction_queries=reference.new_tensor(
                predicted / batch),
            diag_pred_gt_ratio=reference.new_tensor(
                predicted / max(valid_gt, 1)),
        )
        for modality in ('image', 'lidar'):
            name = f'diag_{modality}_valid_ratio'
            if name in predictions:
                diagnostics[name] = predictions[name]
        for name, value in predictions.items():
            if name.startswith('diag_layer0_') and torch.is_tensor(value):
                diagnostics[name] = value
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
                predictions, assignments,
                gt_lanes_3d, gt_lanes_visibility))
        for layer_idx, auxiliary in enumerate(
                predictions.get('aux_predictions', ())):
            auxiliary_assignments = self._auxiliary_assignments(
                auxiliary, assignments,
                gt_lanes_3d, gt_lanes_visibility)
            auxiliary_losses = self._prediction_losses(
                auxiliary, gt_lanes_3d, gt_lanes_visibility,
                gt_lanes_category, auxiliary_assignments,
                is_auxiliary=True)
            for name, value in auxiliary_losses.items():
                losses[f'loss_aux_{layer_idx}_{name.removeprefix("loss_")}'] = (
                    value * self.aux_loss_weight)
            if self.emit_diagnostics:
                auxiliary_diagnostics = self._assignment_diagnostics(
                    auxiliary, auxiliary_assignments,
                    gt_lanes_3d, gt_lanes_visibility)
                for name, value in auxiliary_diagnostics.items():
                    losses[
                        f'diag_aux_{layer_idx}_{name.removeprefix("diag_")}'] = (
                            value)
        topk_indices = predictions.get('layer0_topk_indices')
        if (self.emit_diagnostics and topk_indices is not None
                and predictions.get('aux_predictions')):
            auxiliary = predictions['aux_predictions'][0]
            visible = gt_lanes_visibility > .5
            valid_gt = visible.sum(dim=-1) >= 2
            valid_gt_count = valid_gt.sum().clamp_min(1)

            def query_gt_matches(indices):
                selected_x = auxiliary['pred_x'].gather(
                    1, indices.unsqueeze(-1).expand(
                        -1, -1, auxiliary['pred_x'].size(-1)))
                selected_z = auxiliary['pred_z'].gather(
                    1, indices.unsqueeze(-1).expand(
                        -1, -1, auxiliary['pred_z'].size(-1)))
                distance = torch.linalg.vector_norm(torch.stack((
                    selected_x.unsqueeze(2) - gt_lanes_3d[
                        ..., 0].unsqueeze(1),
                    selected_z.unsqueeze(2) - gt_lanes_3d[
                        ..., 2].unsqueeze(1)), dim=-1), dim=-1)
                close_ratio = (
                    ((distance
                      < self.diagnostic_geometry_distance_threshold)
                     & visible.unsqueeze(1)).sum(dim=-1)
                    / visible.sum(dim=-1).unsqueeze(1).clamp_min(1))
                return (
                    close_ratio >= self.diagnostic_geometry_match_ratio
                ) & valid_gt.unsqueeze(1)

            predicted_matches = query_gt_matches(topk_indices)
            covered = predicted_matches.any(dim=1) & valid_gt
            losses['diag_layer0_topk_gt_recall'] = (
                covered.sum().to(predictions['pred_cls'].dtype)
                / valid_gt_count)
            queries_per_gt = predicted_matches.sum(dim=1)
            valid_queries_per_gt = queries_per_gt[valid_gt]
            losses['diag_layer0_topk_queries_per_gt'] = (
                valid_queries_per_gt.sum().to(
                    predictions['pred_cls'].dtype)
                / valid_gt_count)
            losses['diag_layer0_topk_queries_per_gt_max'] = (
                valid_queries_per_gt.max().to(
                    predictions['pred_cls'].dtype)
                if valid_queries_per_gt.numel()
                else predictions['pred_cls'].new_tensor(0.0))
            losses['diag_layer0_topk_uncovered_gt_ratio'] = (
                (valid_queries_per_gt == 0).sum().to(
                    predictions['pred_cls'].dtype)
                / valid_gt_count)
            selected_indices = predictions.get('selected_query_indices')
            if selected_indices is not None:
                training_covered = (
                    query_gt_matches(selected_indices).any(dim=1) & valid_gt)
                losses['diag_layer0_training_gt_recall'] = (
                    training_covered.sum().to(
                        predictions['pred_cls'].dtype)
                    / valid_gt_count)
        return losses


@MODELS.register_module()
class TopkLaneLoss(LaneLossBase):
    """Supervise nearest anchors per GT and classify others as background."""


@MODELS.register_module()
class LaneLoss(LaneLossBase):
    """Keep the compact lane objectives with Hungarian one-to-one matching."""

    def __init__(self, assigner=None, score_loss_weight=1.0,
                 objectness_loss_weight=None,
                 category_loss_weight=.25,
                 negative_objectness_weight=.1,
                 ambiguous_negative_weight=.1,
                 auxiliary_ambiguous_negative_weight=None,
                 ambiguous_distance_threshold=.5,
                 ambiguous_match_ratio=.75,
                 diagnostic_geometry_distance_threshold=.3,
                 diagnostic_geometry_match_ratio=.8,
                 tube_loss_weight=.5, tube_radius=.5,
                 tube_z_weight=1., tube_epsilon=1e-6,
                 **kwargs):
        if assigner is None:
            assigner = dict(type='HungarianAssigner')
        if objectness_loss_weight is not None:
            score_loss_weight = objectness_loss_weight
        self.score_loss_weight = float(score_loss_weight)
        self.category_loss_weight = float(category_loss_weight)
        self.negative_objectness_weight = float(negative_objectness_weight)
        self.ambiguous_negative_weight = float(ambiguous_negative_weight)
        if auxiliary_ambiguous_negative_weight is None:
            auxiliary_ambiguous_negative_weight = ambiguous_negative_weight
        self.auxiliary_ambiguous_negative_weight = float(
            auxiliary_ambiguous_negative_weight)
        self.ambiguous_distance_threshold = float(
            ambiguous_distance_threshold)
        self.ambiguous_match_ratio = float(ambiguous_match_ratio)
        self.diagnostic_geometry_distance_threshold = float(
            diagnostic_geometry_distance_threshold)
        self.diagnostic_geometry_match_ratio = float(
            diagnostic_geometry_match_ratio)
        self.tube_loss_weight = float(tube_loss_weight)
        self.tube_radius = float(tube_radius)
        self.tube_z_weight = float(tube_z_weight)
        self.tube_epsilon = float(tube_epsilon)
        if min(
                self.score_loss_weight,
                self.category_loss_weight,
                self.negative_objectness_weight,
                self.ambiguous_negative_weight,
                self.auxiliary_ambiguous_negative_weight,
                self.tube_loss_weight) < 0:
            raise ValueError('Lane loss weights must be non-negative.')
        if self.tube_radius <= 0:
            raise ValueError('tube_radius must be positive.')
        if self.tube_z_weight < 0:
            raise ValueError('tube_z_weight must be non-negative.')
        if self.tube_epsilon <= 0:
            raise ValueError('tube_epsilon must be positive.')
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

    def _assignment_diagnostics(
            self, predictions, assignments,
            gt_lanes_3d, gt_visibility):
        diagnostics = super()._assignment_diagnostics(
            predictions, assignments, gt_lanes_3d, gt_visibility)
        tube_ious = []
        for batch_idx, assignment in enumerate(assignments):
            positive_queries = torch.where(assignment >= 0)[0]
            if positive_queries.numel() == 0:
                continue
            matched_gt = assignment[positive_queries]
            tube_ious.append(valid_range_tube_iou(
                predictions['pred_x'][batch_idx, positive_queries],
                predictions['pred_z'][batch_idx, positive_queries],
                predictions['pred_vis'][
                    batch_idx, positive_queries].sigmoid(),
                gt_lanes_3d[batch_idx, matched_gt, :, 0],
                gt_lanes_3d[batch_idx, matched_gt, :, 2],
                gt_visibility[batch_idx, matched_gt],
                tube_radius=self.tube_radius,
                z_weight=self.tube_z_weight,
                epsilon=self.tube_epsilon))
        diagnostics['diag_positive_tube_iou'] = (
            torch.cat(tube_ious).mean()
            if tube_ious else predictions['pred_cls'].new_tensor(0.0))
        return diagnostics

    def _prediction_losses(self, predictions, gt_lanes_3d,
                           gt_visibility, gt_category, assignments,
                           is_auxiliary=False):
        pred_x, pred_z = predictions['pred_x'], predictions['pred_z']
        pred_visibility = predictions['pred_vis']
        pred_score = predictions['pred_cls']
        pred_category = predictions['pred_category']
        total_score = pred_score.sum() * 0
        total_category = pred_category.sum() * 0
        total_x = pred_x.sum() * 0
        total_z = pred_z.sum() * 0
        total_visibility = pred_visibility.sum() * 0
        total_tube = (
            pred_x.sum() + pred_z.sum() + pred_visibility.sum()) * 0
        positive_count = 0
        visible_point_count = 0
        ambiguous = self._ambiguous_negative_mask(
            predictions, gt_lanes_3d, gt_visibility)
        ambiguous_negative_weight = (
            self.auxiliary_ambiguous_negative_weight
            if is_auxiliary else self.ambiguous_negative_weight)

        for batch_idx, assignment in enumerate(assignments):
            positive_queries = torch.where(assignment >= 0)[0]
            positive = torch.zeros(
                pred_score.size(1), dtype=torch.bool,
                device=pred_score.device)
            positive[positive_queries] = True
            score_target = pred_score.new_zeros(pred_score.size(1))
            if positive_queries.numel() > 0:
                matched_gt = assignment[positive_queries]
                target_x = gt_lanes_3d[batch_idx, matched_gt, :, 0]
                target_z = gt_lanes_3d[batch_idx, matched_gt, :, 2]
                visible = gt_visibility[
                    batch_idx, matched_gt].to(pred_x.dtype)
                tube_iou = valid_range_tube_iou(
                    pred_x[batch_idx, positive_queries],
                    pred_z[batch_idx, positive_queries],
                    pred_visibility[
                        batch_idx, positive_queries].sigmoid(),
                    target_x, target_z, visible,
                    tube_radius=self.tube_radius,
                    z_weight=self.tube_z_weight,
                    epsilon=self.tube_epsilon)
                score_target[positive_queries] = tube_iou.detach()
                total_tube = total_tube + (1.0 - tube_iou).sum()
            score_terms = F.binary_cross_entropy_with_logits(
                pred_score[batch_idx], score_target, reduction='none')
            positive_loss = torch.where(
                positive, score_terms, 0).sum()
            positive_loss = (
                positive_loss / positive.sum().clamp_min(1))
            negative = ~positive
            negative_weight = torch.where(
                ambiguous[batch_idx] & negative,
                score_terms.new_tensor(ambiguous_negative_weight),
                score_terms.new_tensor(1.0))
            negative_loss = torch.where(
                negative, score_terms * negative_weight, 0).sum()
            negative_loss = (
                negative_loss / negative.sum().clamp_min(1))
            total_score = total_score + (
                positive_loss
                + self.negative_objectness_weight * negative_loss)

            if positive_queries.numel() == 0:
                continue
            positive_count += int(positive_queries.numel())
            total_category = total_category + F.cross_entropy(
                pred_category[batch_idx, positive_queries],
                gt_category[batch_idx, matched_gt].long(),
                reduction='sum')
            visible_point_count += int(visible.sum())
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
            loss_score=(
                total_score / batch_size * self.score_loss_weight),
            loss_category=(
                total_category / max(positive_count, 1)
                * self.category_loss_weight),
            loss_x=(total_x / max(visible_point_count, 1)
                    * self.x_loss_weight),
            loss_z=(total_z / max(visible_point_count, 1)
                    * self.z_loss_weight),
            loss_visibility=(total_visibility / visibility_count
                             * self.visibility_loss_weight),
            loss_tube=(
                total_tube / max(positive_count, 1)
                * self.tube_loss_weight))
