"""CPU checks for the E4 coverage computation on synthetic lanes."""

import importlib
import sys

import pytest
import torch

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from test_modules import _install_stand_ins  # noqa: E402

_install_stand_ins()
coverage = importlib.import_module('LaneLoom.ablation_extras.tools.e4_q0_coverage')


def _lane(x0, slope=0.0, bend=0.0):
    y = torch.arange(2.0, 101.0, 2.0)
    x = x0 + slope * y + bend * (y / 100.0) ** 2
    return torch.stack((x, y, torch.zeros_like(y)), dim=-1)


def test_coverage_counts_close_query_only():
    gt = torch.stack((_lane(1.8), _lane(5.5, bend=4.0), torch.zeros(50, 3)))
    visibility = torch.zeros(3, 50)
    visibility[0] = 1
    visibility[1, :20] = 1
    queries = torch.stack((_lane(1.9), _lane(9.0)))
    rows = coverage.coverage_for_sample(queries, gt, visibility, 0.3, 0.5, 0.8)
    assert len(rows) == 2
    assert rows[0]['covered'] and rows[0]['covered_loose']
    assert not rows[1]['covered']
    assert rows[0]['best_percentile_m'] == pytest.approx(0.1, abs=1e-5)
    assert rows[0]['chord_deviation_m'] < 1e-3
    assert rows[1]['visible_length_m'] == pytest.approx(38.0)
    assert rows[0]['near_lateral_m'] == pytest.approx(1.8)


def test_ratio_threshold_and_grouping():
    gt = _lane(0.0).unsqueeze(0)
    visibility = torch.ones(1, 50)
    query = _lane(0.0).clone()
    query[:15, 0] += 1.0  # 30% of anchors off by 1 m
    rows = coverage.coverage_for_sample(query[None], gt, visibility, 0.3, 0.5, 0.8)
    assert not rows[0]['covered']
    query[:10, 0] -= 1.0  # now only 10% off
    rows = coverage.coverage_for_sample(query[None], gt, visibility, 0.3, 0.5, 0.8)
    assert rows[0]['covered']
    summary = coverage.aggregate(rows * 3, 'chord_deviation_m',
                                 coverage.CURVATURE_BINS)
    assert summary[0]['lanes'] == 3 and summary[0]['coverage_pct'] == 100.0



def test_initial_curves_uses_all_and_retained_queries():
    class Head:
        def _anchor_prior(self, img_feats, pts_feats, img_metas):
            return {'anchor_points': torch.zeros(2, 505, 50, 3)}

        def forward_bev(self, img_feats, pts_feats, img_metas):
            return 'bev'

        def _lidar_occupancy(self, pts_feats, bev):
            return None

        def _query_feature_sampler(self, img_feats, bev, metas, occupancy):
            return 'sampler'

        def _predict(self, bev, anchor_prior, query_feature_sampler):
            assert bev == 'bev' and query_feature_sampler == 'sampler'
            return {'anchor_points': anchor_prior['anchor_points'][:, :64]}

    class Model:
        lane_head = Head()

        def data_preprocessor(self, data, training):
            assert training is False
            return data

        def _encode_batch(self, batch):
            return 'img', 'pts', 'metas', None, None

    batch, q0_all, q0_kept = coverage.initial_curves(Model(), {'k': 1})
    assert q0_all.shape[1] == 505 and q0_kept.shape[1] == 64
    assert batch == {'k': 1}


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))


def test_straight_lines_cover_straight_lane_but_not_curve():
    anchor_y = torch.arange(2.0, 101.0, 2.0)
    lines = coverage.straight_lines(anchor_y, -25.0, 25.0, 101)
    assert lines.shape == (101, 50, 3)
    assert lines[50, 0, 0] == pytest.approx(0.0)
    assert lines[51, 0, 0] == pytest.approx(0.5)
    gt = torch.stack((_lane(1.5), _lane(1.5, bend=6.0)))
    rows = coverage.coverage_for_sample(lines, gt, torch.ones(2, 50),
                                        0.3, 0.5, 0.8)
    assert rows[0]['covered'] and not rows[1]['covered']
