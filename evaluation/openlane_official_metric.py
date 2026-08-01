"""MMEngine adapter for the official OpenLane 3D lane evaluator."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

from ..models.head.curve_nms import mean_distance_curve_nms
from .lane_evaluation import OpenLaneLaneEvaluator


@METRICS.register_module()
class OpenLaneOfficialMetric(BaseMetric):
    """Evaluate lane annotations with the official OpenLane protocol.

    Current-frame raw annotations remain the default.  Temporal experiments
    may instead evaluate the pose-aligned, fixed-anchor supplement emitted by
    the dataloader.  That keeps the matching protocol identical, but the
    resulting metric is not the official benchmark target because its GT set
    has been augmented.
    """

    default_prefix = 'OpenLaneOfficial'

    def __init__(self,
                 score_threshold: float = 0.5,
                 visibility_threshold: float = 0.5,
                 mean_distance_nms_threshold: float | None = None,
                 use_temporal_gt_supplement: bool = False,
                 collect_device: str = 'cpu',
                 prefix: str | None = None,
                 collect_dir: str | None = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix,
                         collect_dir=collect_dir)
        self.score_threshold = float(score_threshold)
        self.visibility_threshold = float(visibility_threshold)
        self.mean_distance_nms_threshold = (
            None if mean_distance_nms_threshold is None
            else float(mean_distance_nms_threshold))
        if (self.mean_distance_nms_threshold is not None
                and (not np.isfinite(self.mean_distance_nms_threshold)
                     or self.mean_distance_nms_threshold <= 0)):
            raise ValueError(
                'mean_distance_nms_threshold must be finite positive.')
        self.use_temporal_gt_supplement = bool(
            use_temporal_gt_supplement)
        self.evaluator = OpenLaneLaneEvaluator()

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            self.results.append(self._evaluate_sample(sample))

    def _evaluate_sample(self, sample):
        if (not self.use_temporal_gt_supplement
                and sample.get('raw_label') is None):
            raise KeyError(
                'OpenLaneOfficialMetric requires the current-frame raw_label.')
        pred_x = self._numpy(sample['pred_x']).astype(np.float64)
        pred_z = self._numpy(sample['pred_z']).astype(np.float64)
        pred_visibility = self._numpy(
            sample['pred_visibility']).astype(np.float64)
        pred_scores = self._numpy(sample['pred_scores']).astype(np.float64)
        pred_categories = self._numpy(
            sample['pred_category']).astype(np.int64)
        anchor_y = self._numpy(sample['anchor_y_steps']).astype(np.float64)

        selected = pred_scores >= self.score_threshold
        selected &= (
            (pred_visibility >= self.visibility_threshold).sum(axis=1) > 1)
        prediction_count_before_nms = int(selected.sum())
        if sample.get('pred_nms_keep_queries') is not None:
            selected &= self._numpy(
                sample['pred_nms_keep_queries']).astype(bool)
        if self.mean_distance_nms_threshold is not None:
            distance_keep = mean_distance_curve_nms(
                torch.from_numpy(pred_x).unsqueeze(0),
                torch.from_numpy(pred_z).unsqueeze(0),
                torch.from_numpy(pred_visibility).unsqueeze(0),
                torch.from_numpy(pred_scores).unsqueeze(0),
                distance_threshold=self.mean_distance_nms_threshold,
                visibility_threshold=self.visibility_threshold,
                candidate_mask=torch.from_numpy(selected).unsqueeze(0))
            selected &= distance_keep[0].numpy()

        lanes = []
        categories = []
        for query_index in np.flatnonzero(selected):
            visible = (
                pred_visibility[query_index] >= self.visibility_threshold)
            lanes.append(np.column_stack((
                pred_x[query_index, visible],
                anchor_y[visible],
                pred_z[query_index, visible],
            )))
            categories.append(int(pred_categories[query_index]))

        if self.use_temporal_gt_supplement:
            required_keys = (
                'gt_lanes_3d_enhanced',
                'gt_lanes_visibility_enhanced',
                'gt_lanes_category_enhanced',
            )
            missing = [
                key for key in required_keys if sample.get(key) is None]
            if missing:
                raise KeyError(
                    'Temporal GT evaluation requires: '
                    + ', '.join(missing))
            gt_lanes = self._numpy(
                sample['gt_lanes_3d_enhanced']).astype(np.float64)
            gt_visibility = self._numpy(
                sample['gt_lanes_visibility_enhanced']).astype(np.float64)
            gt_categories = self._numpy(
                sample['gt_lanes_category_enhanced']).astype(np.int64)
            official = self.evaluator.evaluate_ground_lanes(
                lanes,
                categories,
                list(gt_lanes),
                [
                    (visibility >= self.visibility_threshold).astype(
                        np.float64)
                    for visibility in gt_visibility
                ],
                gt_categories.tolist())
        else:
            official = self.evaluator.evaluate(
                lanes, categories, sample['raw_label'])
        compact = {
            name: int(official[name])
            for name in (
                'recall_matches', 'precision_matches', 'category_matches',
                'gt_count', 'pred_count', 'match_count')
        }
        compact['pred_count_before_nms'] = prediction_count_before_nms
        compact['nms_suppressed'] = (
            prediction_count_before_nms - int(selected.sum()))
        for name in (
                'x_error_close', 'x_error_far',
                'z_error_close', 'z_error_far'):
            values = np.asarray(official[name], dtype=np.float64)
            values = values[values > -1 + 1e-6]
            compact[f'{name}_sum'] = float(values.sum())
            compact[f'{name}_count'] = int(values.size)
        return compact

    def compute_metrics(self, results):
        summary = {
            'recall_matches': 0,
            'precision_matches': 0,
            'category_matches': 0,
            'gt_count': 0,
            'pred_count': 0,
            'match_count': 0,
            'pred_count_before_nms': 0,
            'nms_suppressed': 0,
        }
        for axis in ('x', 'z'):
            for distance in ('close', 'far'):
                summary[f'{axis}_error_{distance}_sum'] = 0.
                summary[f'{axis}_error_{distance}_count'] = 0
        for result in results:
            for name, value in result.items():
                summary[name] += value

        recall = (
            summary['recall_matches'] / summary['gt_count']
            if summary['gt_count'] else 0.)
        precision = (
            summary['precision_matches'] / summary['pred_count']
            if summary['pred_count'] else 0.)
        f1 = (
            2. * recall * precision / (recall + precision)
            if recall + precision else 0.)
        category_accuracy = (
            summary['category_matches'] / summary['match_count']
            if summary['match_count'] else 0.)

        metrics = dict(
            F1=f1,
            Recall=recall,
            Precision=precision,
            **{'Category Accuracy': category_accuracy},
            num_gt=float(summary['gt_count']),
            num_pred=float(summary['pred_count']),
            num_tp=float(summary['match_count']),
            num_matches=float(summary['match_count']),
            num_recall_matches=float(summary['recall_matches']),
            num_precision_matches=float(summary['precision_matches']),
            pred_per_frame=(
                summary['pred_count'] / max(len(results), 1)),
            num_pred_before_nms=float(summary['pred_count_before_nms']),
            num_nms_suppressed=float(summary['nms_suppressed']),
            nms_suppression_ratio=(
                summary['nms_suppressed']
                / max(summary['pred_count_before_nms'], 1)),
        )
        for axis in ('x', 'z'):
            for source_name, output_name in (
                    ('close', 'near'), ('far', 'far')):
                total = summary[f'{axis}_error_{source_name}_sum']
                count = summary[f'{axis}_error_{source_name}_count']
                metrics[f'{axis}_error_{output_name}'] = (
                    total / count if count else 0.)
        return metrics
