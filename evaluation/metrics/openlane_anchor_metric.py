"""Independent MMEngine evaluation for LaneFormer's OpenLane anchors.

The metric evaluates the model-native vehicle/ground-coordinate representation:
``x, z`` sampled at fixed forward ``y`` anchors.  It deliberately does not
import an evaluator from another repository or convert the annotations a second
time.  A predicted and a labelled lane are one true positive only when a
Hungarian-assigned pair agrees within ``distance_threshold`` on at least
``match_ratio_threshold`` of *both* lanes' visible anchors.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from scipy.optimize import linear_sum_assignment


@METRICS.register_module()
class OpenLaneAnchorMetric(BaseMetric):
    """Evaluate 3D lane sets for the selected OpenLane split.

    Args:
        score_threshold: Queries below this confidence are not predictions.
        visibility_threshold: Per-anchor probability threshold.
        distance_threshold: Maximum 3D ``(x, z)`` distance for an anchor match.
        match_ratio_threshold: Required matched-anchor ratio on both lanes.
        near_distance: Boundary between near and far geometry errors in metres.
        use_temporal_gt_supplement: When available, evaluate against the
            dataloader's pose-aligned temporal GT supplement rather than the
            current-frame-only GT.  This should match temporal-GT training.
    """

    default_prefix = 'OpenLane'

    def __init__(self,
                 score_threshold: float = 0.3,
                 visibility_threshold: float = 0.5,
                 distance_threshold: float = 0.5,
                 match_ratio_threshold: float = 0.75,
                 near_distance: float = 40.0,
                 use_temporal_gt_supplement: bool = True,
                 collect_device: str = 'cpu',
                 prefix: str | None = None,
                 collect_dir: str | None = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix,
                         collect_dir=collect_dir)
        self.score_threshold = float(score_threshold)
        self.visibility_threshold = float(visibility_threshold)
        self.distance_threshold = float(distance_threshold)
        self.match_ratio_threshold = float(match_ratio_threshold)
        self.near_distance = float(near_distance)
        self.use_temporal_gt_supplement = bool(use_temporal_gt_supplement)

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """Store compact per-frame counts; MMEngine collects them across ranks."""
        for sample in data_samples:
            self.results.append(self._evaluate_sample(sample))

    def _evaluate_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        pred_x = self._numpy(sample['pred_x']).astype(np.float64)
        pred_z = self._numpy(sample['pred_z']).astype(np.float64)
        pred_vis = self._numpy(sample['pred_visibility']).astype(np.float64)
        pred_scores = self._numpy(sample['pred_scores']).astype(np.float64)
        pred_category = self._numpy(sample['pred_category']).astype(np.int64)
        use_supplement = (
            self.use_temporal_gt_supplement and
            sample.get('gt_lanes_3d_enhanced') is not None and
            sample.get('gt_lanes_visibility_enhanced') is not None)
        target_suffix = '_enhanced' if use_supplement else ''
        gt_lanes = self._numpy(
            sample[f'gt_lanes_3d{target_suffix}']).astype(np.float64)
        gt_vis = self._numpy(
            sample[f'gt_lanes_visibility{target_suffix}']).astype(np.float64)
        category_key = (
            'gt_lanes_category_enhanced'
            if use_supplement and sample.get('gt_lanes_category_enhanced') is not None
            else 'gt_lanes_category')
        gt_category = self._numpy(sample[category_key]).astype(np.int64)
        anchor_y = self._numpy(sample['anchor_y_steps']).astype(np.float64)

        prediction_mask = pred_scores >= self.score_threshold
        prediction_mask &= (pred_vis >= self.visibility_threshold).any(axis=1)
        gt_mask = (gt_vis >= self.visibility_threshold).any(axis=1)
        pred_indices = np.flatnonzero(prediction_mask)
        gt_indices = np.flatnonzero(gt_mask)

        summary = self._empty_summary()
        summary['gt'] = int(gt_indices.size)
        summary['pred'] = int(pred_indices.size)
        if gt_indices.size and pred_indices.size:
            cost = np.empty((gt_indices.size, pred_indices.size), dtype=np.float64)
            pair_data: dict[tuple[int, int], dict[str, Any]] = {}
            for row, gt_idx in enumerate(gt_indices):
                for col, pred_idx in enumerate(pred_indices):
                    pair = self._pair_statistics(
                        gt_lanes[gt_idx], gt_vis[gt_idx], pred_x[pred_idx],
                        pred_z[pred_idx], pred_vis[pred_idx], anchor_y)
                    cost[row, col] = pair['cost']
                    pair_data[row, col] = pair
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows.tolist(), cols.tolist()):
                pair = pair_data[row, col]
                if pair['is_tp']:
                    gt_idx = gt_indices[row]
                    pred_idx = pred_indices[col]
                    summary['tp'] += 1
                    summary['category_count'] += 1
                    summary['category_correct'] += int(
                        pred_category[pred_idx] == gt_category[gt_idx])
                    for name in ('x_near_sum', 'z_near_sum', 'near_count',
                                 'x_far_sum', 'z_far_sum', 'far_count'):
                        summary[name] += pair[name]

        return summary

    def _pair_statistics(self, gt_lane: np.ndarray, gt_vis: np.ndarray,
                         pred_x: np.ndarray, pred_z: np.ndarray,
                         pred_vis: np.ndarray, anchor_y: np.ndarray) -> dict[str, Any]:
        gt_valid = gt_vis >= self.visibility_threshold
        pred_valid = pred_vis >= self.visibility_threshold
        union = gt_valid | pred_valid
        shared = gt_valid & pred_valid
        distances = np.full(anchor_y.shape, self.distance_threshold, dtype=np.float64)
        distances[shared] = np.hypot(
            gt_lane[shared, 0] - pred_x[shared],
            gt_lane[shared, 2] - pred_z[shared])
        matched = shared & (distances < self.distance_threshold)
        gt_count = int(gt_valid.sum())
        pred_count = int(pred_valid.sum())
        gt_ratio = float(matched.sum() / max(gt_count, 1))
        pred_ratio = float(matched.sum() / max(pred_count, 1))
        is_tp = (gt_ratio >= self.match_ratio_threshold and
                 pred_ratio >= self.match_ratio_threshold)
        x_error = np.abs(gt_lane[:, 0] - pred_x)
        z_error = np.abs(gt_lane[:, 2] - pred_z)
        near = matched & (anchor_y <= self.near_distance)
        far = matched & (anchor_y > self.near_distance)
        return dict(
            cost=float(distances[union].mean()) if union.any() else self.distance_threshold,
            is_tp=is_tp,
            x_near_sum=float(x_error[near].sum()),
            z_near_sum=float(z_error[near].sum()),
            near_count=int(near.sum()),
            x_far_sum=float(x_error[far].sum()),
            z_far_sum=float(z_error[far].sum()),
            far_count=int(far.sum()))

    @staticmethod
    def _empty_summary() -> dict[str, float | int]:
        return dict(gt=0, pred=0, tp=0, category_count=0, category_correct=0,
                    x_near_sum=0.0, z_near_sum=0.0,
                    near_count=0, x_far_sum=0.0, z_far_sum=0.0, far_count=0)

    def compute_metrics(self, results: list[dict[str, Any]]) -> dict[str, float]:
        summary = self._empty_summary()
        for result in results:
            for name, value in result.items():
                summary[name] += value

        gt, pred, tp = int(summary['gt']), int(summary['pred']), int(summary['tp'])
        recall = tp / gt if gt else 0.0
        precision = tp / pred if pred else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        near_count, far_count = int(summary['near_count']), int(summary['far_count'])
        category_count = int(summary['category_count'])
        return dict(
            F1=f1,
            Recall=recall,
            Precision=precision,
            **{'Category Accuracy': (
                int(summary['category_correct']) / category_count
                if category_count else 0.0)},
            x_error_near=(float(summary['x_near_sum']) / near_count if near_count else 0.0),
            z_error_near=(float(summary['z_near_sum']) / near_count if near_count else 0.0),
            x_error_far=(float(summary['x_far_sum']) / far_count if far_count else 0.0),
            z_error_far=(float(summary['z_far_sum']) / far_count if far_count else 0.0),
            num_gt=float(gt),
            num_pred=float(pred),
            num_tp=float(tp),
        )
