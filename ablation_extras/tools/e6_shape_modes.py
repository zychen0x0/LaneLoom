"""E6: how many distinct lane shapes does one OpenLane frame contain?

For every annotated frame, the ground-truth lanes are grouped into shape
modes with the same complete-link rule that the curve query generator uses
for its mode loss: two lanes share a mode when, after removing their mean
lateral and vertical offsets, they agree within the configured point,
heading and pitch tolerances. The script reports the distribution of the
number of modes per frame and the share of frames whose lanes are covered
by at most k modes. Only label files are read; no images, LiDAR or GPU.

Usage (from the repository root, PYTHONPATH set to its parent):
    python ablation_extras/tools/e6_shape_modes.py \\
        configs/main/laneloom_openlane_image_lidar_temporal.py --split train

Outputs: ablation_extras/return_back/e6_shape_modes_<split>.json and .md
"""

import argparse
from collections import Counter
import json

import torch
from mmengine.config import Config
from mmdet3d.registry import DATASETS, MODELS

from LaneLoom.ablation_extras.tools._runtime import RETURN_ROOT, Progress, now
from LaneLoom.engine import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('config')
    parser.add_argument('--split', choices=('train', 'val'), default='train')
    parser.add_argument('--load-interval', type=int, default=1)
    return parser.parse_args()


def frame_modes(prior, lanes, visibility):
    """Number of shape modes and of mode-eligible lanes in one frame."""
    eligible = (visibility > 0.5).sum(dim=-1) >= prior.min_mode_points
    if not eligible.any():
        return 0, 0
    _, active, _ = prior._gt_shape_modes(lanes[None], visibility[None])
    return int(active.sum()), int(eligible.sum())


def summarize(mode_counts, lane_counts, max_k=8):
    frames = [m for m in mode_counts if m > 0]
    histogram = Counter(frames)
    total = max(len(frames), 1)
    cumulative = {
        f'<= {k}': round(100.0 * sum(v for m, v in histogram.items()
                                     if m <= k) / total, 2)
        for k in range(1, max_k + 1)}
    return dict(
        frames_with_lanes=len(frames),
        mean_modes_per_frame=round(sum(frames) / total, 3),
        mean_eligible_lanes_per_frame=round(
            sum(lane_counts) / max(len(lane_counts), 1), 3),
        histogram={str(k): histogram[k] for k in sorted(histogram)},
        cumulative_percent=cumulative)


def main():
    args = parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    loader_cfg = (cfg.train_dataloader if args.split == 'train'
                  else cfg.val_dataloader)
    dataset_cfg = loader_cfg.dataset
    dataset_cfg.dataset_cfg['load_interval'] = args.load_interval
    dataset = DATASETS.build(dataset_cfg)
    prior_cfg = dict(cfg.model.lane_head.anchor_prior)
    prior_cfg['type'] = 'LiDARShapeTokenPrior'
    prior = MODELS.build(prior_cfg).eval()

    print(f'[{now()}] E6 start: {len(dataset.label_files)} {args.split} '
          'label files', flush=True)
    progress = Progress(len(dataset.label_files), 'E6 frames', every=2000)
    mode_counts, lane_counts = [], []
    with torch.no_grad():
        for index, path in enumerate(dataset.label_files):
            packed = dataset._parse_openlane_gt(dataset._read_json(path))
            modes, lanes = frame_modes(
                prior, packed['gt_lanes_3d'], packed['gt_lanes_visibility'])
            mode_counts.append(modes)
            lane_counts.append(lanes)
            progress.update(index + 1)

    result = dict(split=args.split, load_interval=args.load_interval,
                  config=args.config, **summarize(mode_counts, lane_counts))
    RETURN_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = RETURN_ROOT / f'e6_shape_modes_{args.split}.json'
    json_path.write_text(json.dumps(result, indent=2) + '\n')
    lines = ['| Modes per frame <= k | ' + ' | '.join(
        str(k) for k in range(1, 9)) + ' |',
        '|---|' + '---|' * 8,
        '| Frames (%) | ' + ' | '.join(
            str(v) for v in result['cumulative_percent'].values()) + ' |']
    table = '\n'.join(lines) + '\n'
    (RETURN_ROOT / f'e6_shape_modes_{args.split}.md').write_text(table)
    print(table)
    print(f'[{now()}] E6 SUCCESS. Send this file to Claude: {json_path}',
          flush=True)


if __name__ == '__main__':
    main()
