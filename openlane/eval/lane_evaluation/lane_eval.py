# Copyright (c) 2022 The OpenLane Dataset Authors.
# Licensed under the Apache License, Version 2.0.
# Modified for LaneLoom's in-process MMEngine integration.
"""In-process adaptation of OpenLane's official 3D lane evaluator."""

from __future__ import annotations

import numpy as np

from .min_cost_flow import solve_min_cost_flow
from .utils import (
    prune_3d_lane_by_range,
    prune_3d_lane_by_visibility,
    resample_laneline_in_y,
)


class OpenLaneLaneEvaluator:
    """Evaluate ground-coordinate lanes with the official OpenLane protocol."""

    def __init__(self):
        self.top_view_region = np.array([
            [-10., 103.],
            [10., 103.],
            [-10., 3.],
            [10., 3.],
        ])
        self.x_min = float(self.top_view_region[0, 0])
        self.x_max = float(self.top_view_region[1, 0])
        self.y_min = float(self.top_view_region[2, 1])
        self.y_max = float(self.top_view_region[0, 1])
        self.y_samples = np.linspace(
            self.y_min, self.y_max, num=100, endpoint=False)
        self.dist_th = 1.5
        self.ratio_th = 0.75
        self.close_range = 40.

    @staticmethod
    def ground_truth_from_raw_label(raw_label):
        """Convert one official annotation JSON object to ground coordinates."""
        extrinsic = np.asarray(raw_label['extrinsic'], dtype=np.float64).copy()
        vehicle_to_ground = np.array([
            [0., 1., 0.],
            [-1., 0., 0.],
            [0., 0., 1.],
        ])
        ground_to_camera = np.array([
            [1., 0., 0.],
            [0., 0., 1.],
            [0., -1., 0.],
        ])
        extrinsic[:3, :3] = (
            np.linalg.inv(vehicle_to_ground)
            @ extrinsic[:3, :3]
            @ vehicle_to_ground
            @ ground_to_camera)
        extrinsic[:2, 3] = 0.
        camera_representation = np.linalg.inv(np.array([
            [0., 0., 1., 0.],
            [-1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., 0., 1.],
        ]))

        lanes = []
        visibility = []
        categories = []
        for packed_lane in raw_label.get('lane_lines', []):
            lane = np.asarray(packed_lane['xyz'], dtype=np.float64)
            lane_h = np.vstack((lane, np.ones((1, lane.shape[1]))))
            ground_lane = (
                extrinsic @ camera_representation @ lane_h)[:3].T
            lanes.append(ground_lane)
            visibility.append(np.asarray(
                packed_lane['visibility'], dtype=np.float64))
            categories.append(int(packed_lane['category']))
        return lanes, visibility, categories

    def evaluate(self, pred_lanes, pred_categories, raw_label):
        gt_lanes, gt_visibility, gt_categories = (
            self.ground_truth_from_raw_label(raw_label))
        return self.evaluate_ground_lanes(
            pred_lanes, pred_categories,
            gt_lanes, gt_visibility, gt_categories)

    def evaluate_ground_lanes(self, pred_lanes, pred_categories,
                              gt_lanes, gt_visibility, gt_categories):
        """Run the official per-frame matching and error computation."""
        pred_lanes = [
            np.asarray(lane, dtype=np.float64).copy() for lane in pred_lanes]
        pred_categories = [int(category) for category in pred_categories]
        gt_lanes = [
            prune_3d_lane_by_visibility(
                np.asarray(lane, dtype=np.float64),
                np.asarray(gt_visibility[index]))
            for index, lane in enumerate(gt_lanes)
        ]
        gt_categories = [int(category) for category in gt_categories]

        gt_lanes, gt_categories = self._prepare_lanes(
            gt_lanes, gt_categories)
        pred_lanes, pred_categories = self._prepare_lanes(
            pred_lanes, pred_categories)
        gt_lanes, gt_visibility_mat, gt_categories = self._resample_lanes(
            gt_lanes, gt_categories)
        pred_lanes, pred_visibility_mat, pred_categories = (
            self._resample_lanes(pred_lanes, pred_categories))

        gt_count = len(gt_lanes)
        pred_count = len(pred_lanes)
        summary = dict(
            recall_matches=0,
            precision_matches=0,
            category_matches=0,
            gt_count=gt_count,
            pred_count=pred_count,
            match_count=0,
            x_error_close=[],
            x_error_far=[],
            z_error_close=[],
            z_error_far=[],
        )
        if gt_count == 0 or pred_count == 0:
            return summary

        pair_data, adj_mat, cost_mat = self._pair_matrices(
            gt_lanes, gt_visibility_mat, pred_lanes, pred_visibility_mat)
        for gt_index, pred_index, cost in solve_min_cost_flow(
                adj_mat, cost_mat):
            if cost >= self.dist_th * self.y_samples.size:
                continue
            pair = pair_data[gt_index, pred_index]
            summary['match_count'] += 1
            summary['recall_matches'] += int(
                pair['num_matches']
                / np.sum(gt_visibility_mat[gt_index]) >= self.ratio_th)
            summary['precision_matches'] += int(
                pair['num_matches']
                / np.sum(pred_visibility_mat[pred_index]) >= self.ratio_th)
            summary['category_matches'] += int(
                pred_categories[pred_index] == gt_categories[gt_index]
                or (pred_categories[pred_index] == 20
                    and gt_categories[gt_index] == 21))
            for name in (
                    'x_error_close', 'x_error_far',
                    'z_error_close', 'z_error_far'):
                summary[name].append(pair[name])
        return summary

    def _prepare_lanes(self, lanes, categories):
        prepared_lanes = []
        prepared_categories = []
        for lane, category in zip(lanes, categories):
            if lane.shape[0] <= 1:
                continue
            if not (
                    lane[0, 1] < self.y_samples[-1]
                    and lane[-1, 1] > self.y_samples[0]):
                continue
            lane = prune_3d_lane_by_range(lane, self.x_min, self.x_max)
            if lane.shape[0] <= 1:
                continue
            prepared_lanes.append(lane)
            prepared_categories.append(category)
        return prepared_lanes, prepared_categories

    def _resample_lanes(self, lanes, categories):
        sampled_lanes = []
        visibility_rows = []
        sampled_categories = []
        for lane, category in zip(lanes, categories):
            min_y = np.min(lane[:, 1])
            max_y = np.max(lane[:, 1])
            x_values, z_values, visibility = resample_laneline_in_y(
                lane, self.y_samples, out_vis=True)
            visibility = np.logical_and(
                x_values >= self.x_min,
                np.logical_and(
                    x_values <= self.x_max,
                    np.logical_and(
                        self.y_samples >= min_y,
                        self.y_samples <= max_y)))
            if np.sum(visibility) <= 1:
                continue
            sampled_lanes.append(np.column_stack((x_values, z_values)))
            visibility_rows.append(visibility)
            sampled_categories.append(category)
        visibility_mat = np.asarray(
            visibility_rows, dtype=np.float64).reshape(-1, self.y_samples.size)
        return sampled_lanes, visibility_mat, sampled_categories

    def _pair_matrices(self, gt_lanes, gt_visibility,
                       pred_lanes, pred_visibility):
        gt_count = len(gt_lanes)
        pred_count = len(pred_lanes)
        adj_mat = np.ones((gt_count, pred_count), dtype=np.int64)
        cost_mat = np.full(
            (gt_count, pred_count), 1000, dtype=np.int64)
        pair_data = {}
        close_index = np.where(self.y_samples > self.close_range)[0][0]
        for gt_index in range(gt_count):
            for pred_index in range(pred_count):
                x_dist = np.abs(
                    gt_lanes[gt_index][:, 0]
                    - pred_lanes[pred_index][:, 0])
                z_dist = np.abs(
                    gt_lanes[gt_index][:, 1]
                    - pred_lanes[pred_index][:, 1])
                both_visible = np.logical_and(
                    gt_visibility[gt_index] >= .5,
                    pred_visibility[pred_index] >= .5)
                both_invisible = np.logical_and(
                    gt_visibility[gt_index] < .5,
                    pred_visibility[pred_index] < .5)
                other = np.logical_not(
                    np.logical_or(both_visible, both_invisible))
                euclidean = np.hypot(x_dist, z_dist)
                euclidean[both_invisible] = 0.
                euclidean[other] = self.dist_th
                cost = np.sum(euclidean)
                cost_mat[gt_index, pred_index] = (
                    1 if 0 < cost < 1 else int(cost))
                pair_data[gt_index, pred_index] = dict(
                    num_matches=(
                        np.sum(euclidean < self.dist_th)
                        - np.sum(both_invisible)),
                    x_error_close=self._visible_error(
                        x_dist[:close_index], both_visible[:close_index]),
                    z_error_close=self._visible_error(
                        z_dist[:close_index], both_visible[:close_index]),
                    x_error_far=self._visible_error(
                        x_dist[close_index:], both_visible[close_index:]),
                    z_error_far=self._visible_error(
                        z_dist[close_index:], both_visible[close_index:]),
                )
        return pair_data, adj_mat, cost_mat

    @staticmethod
    def _visible_error(distance, visible):
        count = np.sum(visible)
        if count == 0:
            return -1.
        return float(np.sum(distance * visible) / count)
