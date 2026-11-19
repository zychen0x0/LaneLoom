"""CPU check for the E3 stage breakdown call sequence."""

import importlib
import sys

import torch

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from test_modules import _install_stand_ins  # noqa: E402

_install_stand_ins()
fps = importlib.import_module('LaneLoom.ablation_extras.tools.e3_benchmark_fps')


def test_stage_times_calls_each_stage_once():
    calls = []

    class Head:
        def predict(self, img_feats, pts_feats, img_metas, points, prev_bev):
            calls.append(('head', img_feats, pts_feats, prev_bev))

    class Model:
        lane_head = Head()

        def data_preprocessor(self, data, training):
            assert training is False
            return {'img': torch.zeros(1, 3, 1, 3, 8, 8),
                    'img_metas': [{}], 'points': torch.zeros(5, 6)}

        def extract_temporal_img_feat(self, img, metas):
            calls.append('img')
            return 'img_feats'

        def extract_pts_feat(self, points, img_feats, img_metas, images):
            assert isinstance(points, list) and img_feats == 'img_feats'
            calls.append('pts')
            return 'pts_feats'

    stages, frames = fps.stage_times(Model(), {})
    assert frames == 3
    assert set(stages) == {'preprocess', 'image_encoder', 'lidar_encoder',
                           'lane_head'}
    assert all(value >= 0 for value in stages.values())
    assert calls == ['img', 'pts', ('head', 'img_feats', 'pts_feats', None)]
