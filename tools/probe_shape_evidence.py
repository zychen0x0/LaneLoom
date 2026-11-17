"""Measure how the epoch 43 template head responds to real point-cloud evidence.

No training or GT enters prediction. Input interventions preserve painted RGB
with each point; geometric interventions therefore do not accidentally repaint
the moved points from unrelated camera pixels. Reports go to stdout only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from LaneLoom.vis.output import visualize_query_stages as vis


def compare_ply_input(model, batch, ply_path, base):
    """Compare the user's single-frame FOV export with normal temporal input."""
    from plyfile import PlyData
    from scipy.spatial import cKDTree

    device = next(model.parameters()).device
    encoder, prior = model.pts_bev_encoder, model.lane_head.anchor_prior
    vertices = PlyData.read(ply_path)['vertex'].data
    xyz = np.column_stack([vertices[k] for k in ('x', 'y', 'z')]).astype(np.float32)
    rgb = np.column_stack([vertices[k] for k in ('red', 'green', 'blue')]).astype(np.float32)
    norm = batch['img_metas'][0]['current']['img_norm_cfg']
    rgb = (rgb - np.asarray(norm['mean'])) / np.asarray(norm['std'])
    current_queue = batch['img'].shape[1] - 1
    values = np.column_stack((xyz, vertices['intensity'], vertices['elongation'],
                              rgb, np.full(len(xyz), current_queue))).astype(np.float32)
    p = torch.from_numpy(values).to(device)
    raw = batch['lidar'][0].to(device)
    current = raw[raw[:, -1] == current_queue]
    inside = ((xyz >= np.asarray(encoder.point_cloud_range[:3]))
              & (xyz <= np.asarray(encoder.point_cloud_range[3:]))).all(-1)
    distances, _ = cKDTree(current[:, :3].cpu().numpy()).query(xyz[inside])
    print('PLY ALIGNMENT', ply_path, 'points', len(xyz), 'inside model ROI', int(inside.sum()),
          'current-frame nearest XYZ distance max/median', float(distances.max()), float(np.median(distances)))
    if np.max(distances) > .01:
        raise ValueError('PLY does not align with the current-frame model points within 1 cm.')
    images = batch['img'].to(device)
    for name, points in (('current_frame_all_points', current),
                         ('user_ply_single_frame_fov', p)):
        if name == 'user_ply_single_frame_fov':
            with patch.object(encoder, 'paint_points', return_value=[points]):
                bev = encoder([points])
        else:
            bev = encoder([points], images=images, img_metas=batch['img_metas'])
        shapes = prior(bev)['shape_points'][0].cpu()
        print('PLY INPUT COMPARISON', name, 'points before ROI', len(points),
              matched_change(base, shapes), 'template x at 60m', shapes[:, 11, 0].tolist(), flush=True)


def intervene(points, mode):
    p = points.clone()
    if mode == 'flat_z':
        p[:, 2] = 0
    elif mode == 'zero_rgb':
        p[:, 5:8] = 0
    elif mode == 'shuffle_rgb':
        generator = torch.Generator(device=p.device).manual_seed(0)
        p[:, 5:8] = p[torch.randperm(len(p), generator=generator, device=p.device), 5:8]
    elif mode == 'near_only':
        p = p[p[:, 1] < 40]
    elif mode == 'far_only':
        p = p[p[:, 1] >= 40]
    elif mode == 'shear_left':
        p[:, 0] -= .05 * p[:, 1]
    elif mode == 'shear_right':
        p[:, 0] += .05 * p[:, 1]
    elif mode == 'grade_up':
        p[:, 2] += .01 * p[:, 1]
    elif mode in ('bend_left', 'bend_right'):
        p[:, 0] += (-5. if mode == 'bend_left' else 5.) * (p[:, 1] / 100).square()
    elif mode != 'original':
        raise ValueError(mode)
    return p


def matched_change(base, altered, expected=None):
    """Match unordered templates on mean xz distance; also report terminal changes."""
    target = base if expected is None else expected
    costs = torch.linalg.vector_norm(
        altered[:, None, :, [0, 2]] - target[None, :, :, [0, 2]], dim=-1).mean(-1)
    rows, columns = linear_sum_assignment(costs.numpy())
    ordered = torch.empty_like(altered)
    ordered[columns] = altered[rows]
    difference = ordered - base
    checkpoints = [int((base[0, :, 1] - y).abs().argmin()) for y in (20, 40, 60, 100)]
    t = base[0, :, 1] / 100
    predicted_dx = difference[:, :, 0].mean(0)
    expected_dx = (target - base)[:, :, 0].mean(0)
    predicted_nonlin = predicted_dx - t * (predicted_dx @ t) / (t @ t)
    expected_nonlin = expected_dx - t * (expected_dx @ t) / (t @ t)
    denominator = expected_nonlin.square().sum()
    nonlin_gain = float(predicted_nonlin @ expected_nonlin / denominator) if denominator > 1e-6 else None
    return dict(mean_xz_change_m=float(torch.linalg.vector_norm(difference[..., [0, 2]], dim=-1).mean()),
                mean_dx_at_20_40_60_100=[round(float(x), 3) for x in difference[:, checkpoints, 0].mean(0)],
                nonlinear_x_response_gain=nonlin_gain,
                dx100_m=[round(float(x), 3) for x in difference[:, -1, 0]],
                dz100_m=[round(float(x), 3) for x in difference[:, -1, 2]],
                expected_error_m=float(costs[rows, columns].mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', nargs='+', type=Path, default=vis.DEFAULT_IMAGES)
    parser.add_argument('--checkpoint', type=Path, default=vis.DEFAULT_CHECKPOINT)
    parser.add_argument('--ply', type=Path, help='Compare one matching single-frame painted PLY; requires one --images path.')
    args = parser.parse_args()
    if args.ply is not None and len(args.images) != 1:
        parser.error('--ply requires exactly one --images path.')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = Config.fromstring(saved['meta']['cfg'], '.py')
    del saved
    device = torch.device('cuda:0')
    model = vis.infer._build_model(cfg, args.checkpoint, device)
    encoder = model.pts_bev_encoder
    prior = model.lane_head.anchor_prior
    raw_rows, active_rows = [], []
    with torch.inference_mode():
        for image in args.images:
            batch = vis.infer._build_sample(cfg, image)
            raw = [p.to(device) for p in batch['lidar']]
            images = batch['img'].to(device)
            painted = encoder.paint_points(raw, images=images, img_metas=batch['img_metas'])[0]
            original_bev = encoder(raw, images=images, img_metas=batch['img_metas'])
            base = prior(original_bev)['shape_points'][0].cpu()
            if args.ply is not None:
                compare_ply_input(model, batch, args.ply, base)
            context = prior.feature_encoder(original_bev['features'])
            tokens = context[:, None] + prior.shape_tokens.weight[None]
            preactivation = prior.template_head[0](tokens)
            raw_templates = prior.template_head(tokens)[0]
            active = preactivation[0] > 0
            raw_rows.append(raw_templates.cpu())
            active_rows.append(active.cpu())
            print('\nFRAME', image.stem, 'POINTS', len(painted),
                  'BEV', tuple(original_bev['features'].shape), flush=True)
            weight = encoder.point_mlp[0].weight
            print('POINT MLP input RGB weight norm', float(weight[:, 5:8].norm()),
                  'all input weight norm', float(weight.norm()))
            print('PILLAR BANDS: y-range / points / occupied cells / singleton fraction / z-span mean / nonzero RGB fraction', flush=True)
            pillars = encoder.pillarize([painted])
            for start in range(0, 100, 20):
                n = int(((painted[:, 1] >= start) & (painted[:, 1] < start + 20)).sum())
                cy = (pillars['coordinates'][:, 1].float() + .5) * encoder.pillar_size[1]
                keep = (cy >= start) & (cy < start + 20)
                stats = pillars['statistics'][keep]
                singleton = float((torch.expm1(stats[:, 0]) < 1.5).float().mean()) if len(stats) else None
                span = float((stats[:, 2] - stats[:, 1]).mean()) if len(stats) else None
                band = painted[(painted[:, 1] >= start) & (painted[:, 1] < start + 20)]
                rgb_fraction = float(band[:, 5:8].abs().sum(-1).gt(0).float().mean()) if len(band) else None
                print((start, start + 20), n, len(stats), singleton, span, 'RGB fraction', rgb_fraction)
            for mode in ('original', 'zero_rgb', 'shuffle_rgb', 'flat_z', 'near_only',
                         'far_only', 'shear_left', 'shear_right', 'grade_up',
                         'bend_left', 'bend_right'):
                p = intervene(painted, mode)
                # Only bypass RGB re-projection; the complete trained pillar and
                # sparse BEV encoder still runs on the intervened point cloud.
                with patch.object(encoder, 'paint_points', return_value=[p]):
                    bev = encoder([p])
                altered = prior(bev)['shape_points'][0].cpu()
                if mode == 'original':
                    # Sparse scatter accumulations have small CUDA rounding
                    # differences between identical replays; quantify this noise.
                    torch.testing.assert_close(bev['features'], original_bev['features'], atol=1e-3, rtol=1e-3)
                    torch.testing.assert_close(altered, base, atol=1e-3, rtol=1e-3)
                expected = base.clone()
                if mode in ('shear_left', 'shear_right'):
                    expected[:, :, 0] += (-.05 if mode == 'shear_left' else .05) * expected[:, :, 1]
                elif mode == 'grade_up':
                    expected[:, :, 2] += .01 * expected[:, :, 1]
                elif mode in ('bend_left', 'bend_right'):
                    expected[:, :, 0] += (-5. if mode == 'bend_left' else 5.) * (expected[:, :, 1] / 100).square()
                valid = ((p[:, 0] >= -25) & (p[:, 0] <= 25)
                         & (p[:, 1] >= 0) & (p[:, 1] <= 100)
                         & (p[:, 2] >= -5) & (p[:, 2] <= 5))
                context_delta = prior.feature_encoder(bev['features']) - context
                print('PROBE', mode, 'points_kept', int(valid.sum()),
                      'context_delta_norm', round(float(context_delta.norm()), 4),
                      matched_change(base, altered, expected), flush=True)
            del original_bev, painted, images, raw
    raw_rows = torch.stack(raw_rows)
    centered = raw_rows - raw_rows.mean(1, keepdim=True)
    print('\nTEMPLATE COUPLING: each scene uses one shared 64-D context')
    for i, active in enumerate(active_rows):
        difference = (active[:, None] != active[None]).float().mean(-1)
        print(args.images[i].stem, 'mean ReLU mask disagreement among template pairs', float(difference.mean()))
    print('Across-scene raw-template variation norm', float((raw_rows - raw_rows.mean(0)).norm()),
          'across-scene centered-template variation norm', float((centered - centered.mean(0)).norm()))


if __name__ == '__main__':
    main()
