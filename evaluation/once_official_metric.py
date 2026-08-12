"""MMEngine adapter for the released ONCE-3DLanes evaluation protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import cv2
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString


def _camera_lane_from_ground(x, forward, height):
    """LaneLoom (x, forward, height) -> ONCE camera (x, y, z)."""
    return np.column_stack((x, -height, forward)).astype(np.float64)


class _EvaluationCounts(NamedTuple):
    gt_count: int
    pred_count: int
    true_positive: int
    distance_sum: float


class _ReleasedONCEProtocol:
    """Dependency-modernized equivalent of once_3dlanes_benchmark/eval.py.

    The released evaluator uses BEV raster IoU for Hungarian assignment and a
    0.3 m one-way line distance as the true-positive gate.  Its ``iou_thresh``
    configuration value is not applied by the published code, so this adapter
    intentionally preserves that behavior for score comparability.
    """

    def __init__(self, side_range=(-10.0, 10.0), fwd_range=(0.0, 50.0),
                 resolution=0.05, lane_width_pixels=30,
                 distance_threshold=0.3):
        self.side_range = tuple(float(value) for value in side_range)
        self.fwd_range = tuple(float(value) for value in fwd_range)
        self.resolution = float(resolution)
        self.lane_width_pixels = int(lane_width_pixels)
        self.distance_threshold = float(distance_threshold)
        self.width = int(
            (self.side_range[1] - self.side_range[0]) / self.resolution)
        self.height = int(
            (self.fwd_range[1] - self.fwd_range[0]) / self.resolution)

    @staticmethod
    def sort_camera_lane(lane):
        lane = np.asarray(lane, dtype=np.float64)
        if lane.shape[0] >= 2 and lane[0, 2] > lane[1, 2]:
            lane = lane[::-1]
        return lane

    def rasterize(self, lane):
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        lane = np.asarray(lane, dtype=np.float64)
        lane = lane[lane[:, 2] < 10.0]
        if lane.shape[0] < 2:
            return image
        x_pixels = (lane[:, 0] / self.resolution).astype(np.int32)
        y_pixels = (-lane[:, 2] / self.resolution).astype(np.int32)
        x_pixels += int(self.side_range[1] / self.resolution)
        y_pixels += int(self.fwd_range[1] / self.resolution)
        for index in range(lane.shape[0] - 1):
            cv2.line(
                image,
                (int(x_pixels[index]), int(y_pixels[index])),
                (int(x_pixels[index + 1]), int(y_pixels[index + 1])),
                255,
                self.lane_width_pixels)
        return image

    @staticmethod
    def one_way_distance(gt_lane, pred_lane):
        try:
            source = LineString(np.asarray(gt_lane, dtype=np.float64))
            target = LineString(np.asarray(pred_lane, dtype=np.float64))
            samples = np.arange(0.05, 1.0, 0.1)
            return float(np.mean([
                source.interpolate(sample, normalized=True).distance(target)
                for sample in samples
            ]))
        except (ValueError, TypeError):
            return float('inf')

    def pair_matrices(self, gt_lanes, pred_lanes):
        iou = np.zeros((len(gt_lanes), len(pred_lanes)), dtype=np.float64)
        distance = np.full_like(iou, np.inf)
        gt_rasters = [self.rasterize(lane) for lane in gt_lanes]
        pred_rasters = [self.rasterize(lane) for lane in pred_lanes]
        for gt_index, gt_lane in enumerate(gt_lanes):
            for pred_index, pred_lane in enumerate(pred_lanes):
                first_image = gt_rasters[gt_index]
                second_image = pred_rasters[pred_index]
                union = cv2.bitwise_or(first_image, second_image)
                union_sum = float(union.sum())
                if union_sum:
                    intersection = float(
                        first_image.sum() + second_image.sum() - union_sum)
                    iou[gt_index, pred_index] = intersection / union_sum
                distance[gt_index, pred_index] = self.one_way_distance(
                    gt_lane, pred_lane)
        return iou, distance

    def evaluate_threshold(self, result, threshold):
        gt_count = len(result['gt_lanes'])
        selected = np.flatnonzero(result['pred_scores'] > threshold)
        pred_count = int(selected.size)
        if gt_count == 0 or pred_count == 0:
            return _EvaluationCounts(gt_count, pred_count, 0, 0.0)
        costs = 1.0 - result['iou_matrix'][:, selected]
        gt_indices, selected_indices = linear_sum_assignment(costs)
        pred_indices = selected[selected_indices]
        matched_distances = result['distance_matrix'][gt_indices, pred_indices]
        true_positive = matched_distances < self.distance_threshold
        return _EvaluationCounts(
            gt_count,
            pred_count,
            int(true_positive.sum()),
            float(matched_distances[true_positive].sum()),
        )


@METRICS.register_module()
class ONCEOfficialMetric(BaseMetric):
    """Report ONCE F1, precision, recall and CDE in official camera axes."""

    default_prefix = 'ONCEOfficial'

    def __init__(self, score_threshold=0.5, score_thresholds=None,
                 visibility_threshold=0.5, distance_threshold=0.3,
                 prediction_dir=None, collect_device='cpu', prefix=None,
                 collect_dir=None):
        super().__init__(collect_device=collect_device, prefix=prefix,
                         collect_dir=collect_dir)
        self.score_threshold = float(score_threshold)
        thresholds = (
            [self.score_threshold]
            if score_thresholds is None else
            [float(value) for value in score_thresholds])
        thresholds.append(self.score_threshold)
        self.score_thresholds = tuple(sorted(set(thresholds)))
        if any(not 0.0 <= value < 1.0 for value in self.score_thresholds):
            raise ValueError('ONCE score thresholds must be in [0, 1).')
        self.visibility_threshold = float(visibility_threshold)
        self.prediction_dir = (
            None if prediction_dir is None
            else Path(prediction_dir).expanduser().resolve())
        self.protocol = _ReleasedONCEProtocol(
            distance_threshold=distance_threshold)

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _prepare_result(self, sample):
        raw_label = sample.get('raw_label')
        if raw_label is None:
            raise KeyError('ONCEOfficialMetric requires raw_label.')
        gt_lanes = [
            self.protocol.sort_camera_lane(lane)
            for lane in raw_label.get('lanes', [])
            if np.asarray(lane).ndim == 2 and len(lane) >= 2
        ]

        pred_x = self._numpy(sample['pred_x']).astype(np.float64)
        pred_z = self._numpy(sample['pred_z']).astype(np.float64)
        visibility = self._numpy(
            sample['pred_visibility']).astype(np.float64)
        scores = self._numpy(sample['pred_scores']).astype(np.float64)
        anchor_y = self._numpy(
            sample['anchor_y_steps']).astype(np.float64)
        keep = np.ones(scores.shape, dtype=bool)
        if sample.get('pred_nms_keep_queries') is not None:
            keep &= self._numpy(
                sample['pred_nms_keep_queries']).astype(bool)
        keep &= scores > min(self.score_thresholds)
        keep &= (
            visibility >= self.visibility_threshold).sum(axis=1) >= 2

        pred_lanes = []
        pred_scores = []
        for query_index in np.flatnonzero(keep):
            visible = (
                visibility[query_index] >= self.visibility_threshold)
            lane = _camera_lane_from_ground(
                pred_x[query_index, visible],
                anchor_y[visible],
                pred_z[query_index, visible])
            pred_lanes.append(self.protocol.sort_camera_lane(lane))
            pred_scores.append(float(scores[query_index]))
        iou_matrix, distance_matrix = self.protocol.pair_matrices(
            gt_lanes, pred_lanes)
        return dict(
            scene_name=str(sample.get('scene_name', '')),
            frame_key=str(sample.get('frame_key', '')),
            gt_lanes=gt_lanes,
            pred_lanes=pred_lanes,
            pred_scores=np.asarray(pred_scores, dtype=np.float64),
            iou_matrix=iou_matrix,
            distance_matrix=distance_matrix,
        )

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            self.results.append(self._prepare_result(sample))

    @staticmethod
    def _summarize(counts):
        gt_count = sum(item.gt_count for item in counts)
        pred_count = sum(item.pred_count for item in counts)
        true_positive = sum(item.true_positive for item in counts)
        distance_sum = sum(item.distance_sum for item in counts)
        precision = true_positive / pred_count if pred_count else 0.0
        recall = true_positive / gt_count if gt_count else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0)
        cde = distance_sum / true_positive if true_positive else 0.0
        return dict(
            F1=f1,
            Precision=precision,
            Recall=recall,
            CDE=cde,
            num_gt=float(gt_count),
            num_pred=float(pred_count),
            num_tp=float(true_positive),
        )

    def _write_predictions(self, results):
        if self.prediction_dir is None:
            return
        export_threshold = min(self.score_thresholds)
        for result in results:
            output_path = (
                self.prediction_dir / result['scene_name'] / 'cam01'
                / f"{result['frame_key']}.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            lanes = []
            for lane, score in zip(
                    result['pred_lanes'], result['pred_scores']):
                if score <= export_threshold:
                    continue
                lanes.append(dict(
                    points=np.asarray(lane).tolist(),
                    score=round(float(score), 6)))
            output_path.write_text(
                json.dumps({'lanes': lanes}), encoding='utf-8')

    def compute_metrics(self, results):
        threshold_metrics = {}
        for threshold in self.score_thresholds:
            counts = [
                self.protocol.evaluate_threshold(result, threshold)
                for result in results
            ]
            threshold_metrics[threshold] = self._summarize(counts)

        primary = dict(threshold_metrics[self.score_threshold])
        best_threshold = self.score_thresholds[0]
        for threshold in self.score_thresholds[1:]:
            if (threshold_metrics[threshold]['F1']
                    > threshold_metrics[best_threshold]['F1']):
                best_threshold = threshold
        primary.update(
            score_threshold=self.score_threshold,
            best_F1=threshold_metrics[best_threshold]['F1'],
            best_Precision=threshold_metrics[best_threshold]['Precision'],
            best_Recall=threshold_metrics[best_threshold]['Recall'],
            best_CDE=threshold_metrics[best_threshold]['CDE'],
            best_score_threshold=best_threshold,
        )
        for threshold, values in threshold_metrics.items():
            for name in ('F1', 'Precision', 'Recall', 'CDE'):
                primary[f'{name}@{threshold:g}'] = values[name]
        self._write_predictions(results)
        return primary
