"""Evaluate Q0 on raw official GT, with no target annotations in prediction."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from mmengine.config import Config
from scipy.optimize import linear_sum_assignment, linprog
from scipy.spatial import cKDTree
from mmdet3d.registry import DATASETS

from LaneLoom.datasets.openlane.dataset import openlane_collate
from LaneLoom.engine import register_all_modules
from LaneLoom.tools.train_q0_prior import Q0Pretrainer
from LaneLoom.tools.q0_scenarios import read_validation_scenarios
from LaneLoom.vis.output import infer_openlane_sample as infer
from LaneLoom.vis.output import visualize_query_stages as visual


def cubic_basis(y, length=100.):
    t = np.asarray(y, dtype=np.float64) / length
    return np.stack(((1 - t)**3, 3*(1 - t)**2*t, 3*(1 - t)*t*t, t**3), -1)


def raw_lanes(label, calibration):
    result = []
    for index, lane in enumerate(label['lane_lines']):
        xyz = np.asarray(lane['xyz'], dtype=np.float64)
        points = (calibration['lane_cam_to_ground'] @ np.vstack((xyz, np.ones(xyz.shape[1]))))[:3].T
        visible = np.asarray(lane.get('visibility', np.ones(len(points)))) > 0
        keep = (visible & np.isfinite(points).all(-1) & (points[:, 0] >= -25)
                & (points[:, 0] <= 25) & (points[:, 1] >= 0) & (points[:, 1] <= 100))
        if keep.sum() < 2:
            continue
        points = points[keep]
        points = points[np.argsort(points[:, 1])]
        result.append(dict(json_index=index, points=points, original_visible=int(visible.sum())))
    return result


def summarize_distances(distances, lanes, ratio, threshold=.3):
    """Maximum covered GT count with a distinct Q0 candidate for every lane."""
    costs, fractions = [], []
    for d in distances:
        percentile = np.partition(d, math.ceil(ratio * d.shape[-1]) - 1, axis=-1)
        costs.append(percentile[..., math.ceil(ratio * d.shape[-1]) - 1].reshape(-1))
        fractions.append((d < threshold).mean(-1).reshape(-1))
    cost = np.stack(costs)
    # Failed matches dominate tie-breaking: maximize coverage before distance.
    rows, cols = linear_sum_assignment((cost >= threshold) * 1e6 + np.minimum(cost, 1e3))
    output = []
    for row, col in zip(rows, cols):
        output.append(dict(json_index=lanes[row]['json_index'], query_id=int(col),
                           template=int(col // 101), seed=int(col % 101),
                           percentile_m=float(cost[row, col]), fraction=float(fractions[row][col]),
                           covered=bool(cost[row, col] < threshold)))
    return dict(gt=len(lanes), covered=sum(l['covered'] for l in output), lanes=output)


def coverage_goal_report(data, expected_frames, expected_total=74, required_metrics=None):
    """Require the complete audited evaluation set, never a passing subset."""
    metrics = ('paired_xz_p80', 'paired_xz_p75_03', 'paired_xz_p80_05', 'paired_xz_p75_05')
    required_metrics = tuple(metrics if required_metrics is None else required_metrics)
    if not required_metrics or any(metric not in metrics for metric in required_metrics):
        raise ValueError('Coverage goal must select known, nonempty matching criteria')
    samples = data['samples']
    if ({s['frame'] for s in samples} != set(expected_frames)
            or len(samples) != len(expected_frames) or not expected_frames
            or sum(s['raw_gt'] for s in samples) != expected_total):
        raise ValueError(f'Coverage goal requires all original {len(expected_frames)} frames and {expected_total} raw GT lanes')
    results = {}
    for metric in metrics:
        counts = {}
        for model in ('baseline', 'spatial'):
            for sample in samples:
                if sample[model][metric]['gt'] != sample['raw_gt']:
                    raise ValueError('A GT lane was dropped from a coverage metric')
            counts[model] = sum(s[model][metric]['covered'] for s in samples)
        results[metric] = dict(**counts, total=expected_total,
                               passed=counts['spatial'] / expected_total > .9 and counts['spatial'] > counts['baseline'])
    return dict(passed=all(results[metric]['passed'] for metric in required_metrics),
                required_metrics=list(required_metrics), criteria=results)


def validate_validation_training_manifest(checkpoint, protocol, scenes):
    """Verify the saved training manifest, not just a claimed zero overlap."""
    manifest = Path(checkpoint).parent / 'training_samples.txt'
    payload = manifest.read_bytes()
    if hashlib.sha256(payload).hexdigest() != protocol['training_labels_sha256']:
        raise ValueError('Training manifest does not match the checkpoint')
    paths = [Path(line) for line in payload.decode().splitlines() if line]
    validation_segments = {s['segment'] for s in scenes}
    if (len(paths) != protocol['training_labels']
            or any(not p.parts or p.parts[0] != 'training' for p in paths)
            or any(p.parent.name in validation_segments for p in paths)):
        raise ValueError('Validation scene/segment leaked into training')


def grouped_coverage(samples):
    result = {}
    for group in sorted({s['group'] for s in samples}):
        rows = [s for s in samples if s['group'] == group]
        total = sum(s['raw_gt'] for s in rows)
        metrics = {}
        for key in ('paired_xz_p80', 'paired_xz_p75_03', 'paired_xz_p80_05', 'paired_xz_p75_05'):
            counts = {name: sum(s[name][key]['covered'] for s in rows)
                      for name in ('baseline', 'spatial')}
            metrics[key] = dict(**counts, total=total)
        result[group] = dict(frames=len(rows), raw_gt=total, metrics=metrics)
    return result


def validate_comparison_protocols(old, new, allow_lane_field_loss_difference=False):
    """Keep data/budget checks strict when explicitly ablating auxiliary labels."""
    before, after = dict(old), dict(new)
    for value in (before, after):
        value.pop('model_type')
        value.pop('q0_model_sha256', None)
        if allow_lane_field_loss_difference:
            value.pop('lane_field_loss_weight', None)
            if 'supervision' in value:
                value['supervision'] = value['supervision'].removesuffix(
                    '; sparse multi-label lane-position and visible-point height field')
    if before != after:
        raise ValueError('Training protocols differ')


def validate_comparison_input_frames(old_cfg, new_cfg):
    """Do not confuse a head comparison with a change in spatial input support."""
    old, new = old_cfg.model.pts_bev_encoder, new_cfg.model.pts_bev_encoder
    if (tuple(old.point_cloud_range) != tuple(new.point_cloud_range)
            or old.get('align_to_lane_ground', False) != new.get('align_to_lane_ground', False)):
        raise ValueError('Comparison requires the same BEV coordinate frame and point-cloud range')


def evaluate_lane_field(prior, lanes, pc_range):
    """Inspect intermediate evidence separately from actual Q0 coverage."""
    if 'lane_field_logits' not in prior:
        return None
    logits, height = prior['lane_field_logits'], prior['lane_height_field']
    mode = prior.get('lane_field_mode', 'balanced_sigmoid')
    probability_map = logits.softmax(-1) if mode == 'row_distribution' else logits.sigmoid()
    points = logits.new_tensor(np.concatenate([lane['points'] for lane in lanes]))
    x0, y0, _, x1, y1, _ = pc_range
    grid = torch.stack((2*(points[:, 0]-x0)/(x1-x0)-1,
                        2*(points[:, 1]-y0)/(y1-y0)-1), -1)[None, None]
    field = torch.cat((probability_map, height), 1)
    sampled = torch.nn.functional.grid_sample(
        field, grid, padding_mode='border', align_corners=False)[0, :, 0]
    row_average = torch.nn.functional.grid_sample(
        probability_map.mean(-1, keepdim=True), grid,
        padding_mode='border', align_corners=False)[0, 0, 0]
    # A broad road-area response can have a high score at every GT point.
    # Check actual lateral peaks as a separate diagnostic, never as Q0 recall.
    width = logits.size(-1)
    column = (torch.arange(width, device=logits.device, dtype=logits.dtype)+.5)/width
    row_grid = torch.stack(((2*column-1)[None].expand(len(points), -1),
                            grid[0, 0, :, 1, None].expand(-1, width)), -1)[None]
    probability = torch.nn.functional.grid_sample(
        probability_map, row_grid, padding_mode='border', align_corners=False)[0, 0]
    left = torch.nn.functional.pad(probability[:, :-1], (1, 0), value=-1)
    right = torch.nn.functional.pad(probability[:, 1:], (0, 1), value=-1)
    if mode == 'row_distribution':
        threshold = torch.maximum(probability.amax(-1, keepdim=True)*.5,
                                  probability.new_tensor(2/width))
        above_threshold = probability > threshold
    else:
        above_threshold = probability > .5
    peaks = above_threshold & (probability >= left) & (probability > right)
    dx = ((x0+column*(x1-x0))[None]-points[:, 0, None]).abs()
    nearest_peak = dx.masked_fill(~peaks, torch.inf).amin(-1)
    height_error = (sampled[1]-points[:, 2]).abs()
    return dict(raw_points=len(points), height_mae_m=float(height_error.mean()),
                height_median_error_m=float(height_error.median()),
                height_fraction_under_015=float((height_error < .15).float().mean()),
                mean_probability_at_gt=float(sampled[0].mean()),
                mean_probability_same_rows=float(row_average.mean()),
                gt_vs_row_probability_contrast=float((sampled[0]-row_average).mean()),
                probability_normalization=mode,
                peak_probability_threshold=None if mode == 'row_distribution' else .5,
                peak_threshold_policy=('density>max(2/W,half row peak)' if mode == 'row_distribution'
                                       else 'sigmoid>0.5'),
                fraction_gt_points_near_peak_03=float((nearest_peak < .3).float().mean()),
                fraction_gt_points_near_peak_05=float((nearest_peak < .5).float().mean()))


def audit_cubic_gt(lanes):
    """GT-only reference fits; never used to initialize or update predictions."""
    result = []
    for lane in lanes:
        p = lane['points']
        span = np.ptp(p[:, 1])
        if span < 1e-8:
            result.append(dict(json_index=lane['json_index'], status='not a function of y'))
            continue
        b = cubic_basis((p[:, 1] - p[:, 1].min()) / span, 1.)
        fit = b @ np.linalg.lstsq(b, p[:, [0, 2]], rcond=None)[0]
        error = np.linalg.norm(fit - p[:, [0, 2]], axis=-1)
        item = dict(json_index=lane['json_index'], points=len(p), y_span_m=float(span),
                    least_squares_p95_xz_m=float(np.quantile(error, .95)))
        if len(p) < 20:
            a = np.vstack((np.column_stack((b, -np.ones(len(b)))),
                           np.column_stack((-b, -np.ones(len(b))))))
            optimum = linprog([0, 0, 0, 0, 1], A_ub=a, b_ub=np.r_[p[:, 0], -p[:, 0]],
                               bounds=[(None, None)] * 4 + [(0, None)], method='highs')
            if not optimum.success:
                raise RuntimeError(optimum.message)
            item.update(minimum_worst_x_error_any_cubic_m=float(optimum.fun),
                        p95_requires_all_points=math.ceil(.95 * len(p)) == len(p))
        result.append(item)
    return result


def evaluate_controls(prediction, lanes, length=100.):
    x = prediction['shape_x_controls'][0].detach().cpu().double().numpy()
    z = prediction['shape_z_controls'][0].detach().cpu().double().numpy()
    seeds = np.linspace(-25, 25, 101)
    paired, nearest, lateral_only, height_only = [], [], [], []
    trees = []
    for k in range(len(x)):
        controls = np.stack((x[k], np.linspace(0, length, 4), z[k]), -1)
        # The cubic derivative is bounded by 3*max(control-edge length).
        # This bounds each sampled curve interval by 0.025 m in 3D.
        intervals = max(100, math.ceil(3 * np.linalg.norm(np.diff(controls, axis=0), axis=-1).max() / .025))
        y = np.linspace(0, length, intervals + 1)
        points = np.stack((cubic_basis(y, length) @ x[k], y, cubic_basis(y, length) @ z[k]), -1)
        trees.append(cKDTree(points))
    for lane in lanes:
        p = lane['points']
        b = cubic_basis(p[:, 1], length)
        dx = (b @ x.T).T[:, None] + seeds[None, :, None] - p[None, None, :, 0]
        dz = (b @ z.T).T[:, None] - p[None, None, :, 2]
        paired.append(np.sqrt(dx*dx + dz*dz))
        lateral_only.append(np.abs(dx))
        height_only.append(np.broadcast_to(np.abs(dz), dx.shape))
        shifted = np.broadcast_to(p, (101, *p.shape)).copy()
        shifted[..., 0] -= seeds[:, None]
        nearest.append(np.stack([tree.query(shifted.reshape(-1, 3), workers=1)[0].reshape(101, len(p))
                                 for tree in trees]))
    return dict(predicted_x_controls=x.tolist(), predicted_z_controls=z.tolist(),
                paired_xz_p80=summarize_distances(paired, lanes, .8),
                paired_xz_p75_03=summarize_distances(paired, lanes, .75, threshold=.3),
                paired_xz_p80_05=summarize_distances(paired, lanes, .8, threshold=.5),
                paired_xz_p75_05=summarize_distances(paired, lanes, .75, threshold=.5),
                paired_xz_p95=summarize_distances(paired, lanes, .95),
                diagnostic_axis_p80_03=dict(
                    lateral_only=summarize_distances(lateral_only, lanes, .8),
                    height_only=summarize_distances(height_only, lanes, .8)),
                nearest_3d_p95=summarize_distances(nearest, lanes, .95),
                nearest_sampling_max_interval_m=.025)


def load_model(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    protocol = saved['meta']['protocol']
    if (saved['meta'].get('invalid_for_model_comparison') or protocol['target_overlap']
            or protocol['feature_cache_used'] or protocol['target_oversampling']):
        raise ValueError('Checkpoint does not satisfy target-scene holdout protocol')
    cfg = Config.fromstring(saved['meta']['cfg'], '.py')
    # The evaluated checkpoint contains the complete encoder state. Evaluation
    # must not depend on importing its original initialization file again.
    model = Q0Pretrainer(cfg, initialize_encoder=False)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.checkpoint_postprocessing = saved['meta'].get('checkpoint_postprocessing')
    return model.to(device).eval(), cfg, saved['step'], protocol


def point_rasters(points):
    keep = ((points[:, 0] >= -25) & (points[:, 0] <= 25)
            & (points[:, 1] >= 0) & (points[:, 1] <= 100))
    points = points[keep]
    ix = np.clip(((points[:, 0] + 25) * 5).astype(int), 0, 249)
    iy = np.clip((points[:, 1] * 2.5).astype(int), 0, 249)
    index = iy * 250 + ix
    count = np.bincount(index, minlength=62500)
    height = np.full(62500, np.inf)
    np.minimum.at(height, index, points[:, 2])
    height = np.ma.masked_where(count == 0, height).reshape(250, 250)
    valid = np.abs(points[:, 5:8]).sum(-1) > 0
    color_count = np.bincount(index[valid], minlength=62500)
    color = np.stack([np.bincount(index[valid], weights=points[valid, c], minlength=62500)
                      / np.maximum(color_count, 1) for c in (5, 6, 7)], -1)
    color = np.clip((color * [58.395, 57.12, 57.375] + [123.675, 116.28, 103.53]) / 255, 0, 1)
    color[color_count == 0] = 1
    return height, color.reshape(250, 250, 3)


def render_comparison(image_path, points, calibration, lanes, predictions, result, output, step):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 8})
    image = np.asarray(Image.open(image_path).convert('RGB'))
    height, rgb = point_rasters(points)
    y = np.linspace(0, 100, 1001)
    curves = []
    for prediction in predictions:
        x = prediction['shape_x_controls'][0].detach().cpu().numpy()
        z = prediction['shape_z_controls'][0].detach().cpu().numpy()
        curves.append(np.stack(((cubic_basis(y) @ x.T).T,
                                 np.broadcast_to(y, (len(x), len(y))),
                                 (cubic_basis(y) @ z.T).T), -1))
    colors = plt.get_cmap('tab10').colors
    extractor = result.get('extractor_label', 'Spatial ordered prior')
    metric = result.get('visualized_metric', 'paired_xz_p80')
    coverage_label = {'paired_xz_p80': '80% points, xz < 0.3 m',
                      'paired_xz_p75_03': '75% points, xz < 0.3 m',
                      'paired_xz_p80_05': '80% points, xz < 0.5 m',
                      'paired_xz_p75_05': '75% points, xz < 0.5 m'}[metric]
    fig, axes = plt.subplots(2, 4, figsize=(17.6, 8), gridspec_kw=dict(hspace=.35, wspace=.32))
    fig.subplots_adjust(top=.90, bottom=.10, left=.06, right=.98)
    fig.suptitle(f'BEV-to-Q0 extractor comparison | {image_path.stem} | step {step}', fontsize=12)
    axes[0, 0].imshow(image)
    axes[0, 0].set_aspect('equal', adjustable='datalim')
    axes[0, 0].set_axis_off()
    axes[0, 0].set_title('New Q0 camera projection + raw GT', fontsize=10)
    axes[0, 1].imshow(height, extent=(-25, 25, 100, 0), vmin=-3, vmax=3,
                       cmap='coolwarm', aspect='auto')
    axes[0, 1].set_title('Observed LiDAR minimum z: -3 to +3 m', fontsize=10)
    axes[1, 0].imshow(rgb, extent=(-25, 25, 100, 0), aspect='auto')
    axes[1, 0].set_title('Painted point colors; unobserved: white', fontsize=10)
    for ax in (axes[0, 1], axes[1, 0]):
        ax.set(xlim=(-25, 25), ylim=(0, 100), xlabel='x (m)', ylabel='y (m)')
    for k, curve in enumerate(curves[1]):
        xx = np.where(np.abs(curve[:, 0]) <= 50, curve[:, 0], np.nan)
        axes[1, 1].plot(xx, curve[:, 1], color=colors[k], lw=1.2)
        if 'road_track_x' in predictions[1]:
            prediction = predictions[1]
            support = prediction['road_track_support'][0, k].detach().cpu().numpy()
            track_x = (prediction['road_track_x'][0, k]
                       - prediction['road_track_x_origin'][0, k]).detach().cpu().numpy()
            track_y = prediction['road_track_y'].detach().cpu().numpy()
            keep = (support > .1) & (np.abs(track_x) <= 50)
            axes[1, 1].scatter(track_x[keep], track_y[keep], color=colors[k], s=3)
        elif 'lane_row_x_origin' in predictions[1]:
            prediction = predictions[1]
            track_x = (prediction['lane_row_x'][0, k]
                       - prediction['lane_row_x_origin'][0, k]).detach().cpu().numpy()
            weights = prediction['lane_row_weights'][0, k].detach().cpu().numpy()
            track_y = (np.arange(len(track_x))+.5)*100/len(track_x)
            keep = (weights > 0) & (np.abs(track_x) <= 50)
            axes[1, 1].scatter(track_x[keep], track_y[keep], color=colors[k], s=4)
    axes[1, 1].set(xlim=(-50, 50), ylim=(0, 100), xlabel='relative x (m)', ylabel='y (m)')
    template_title = ('Five templates + predicted row readouts'
                      if ('road_track_x' in predictions[1] or 'lane_row_x_origin' in predictions[1])
                      else 'Five relative templates (before shifts)')
    axes[1, 1].set_title(template_title, fontsize=10)
    z_limits = []
    by_json = {l['json_index']: l for l in lanes}
    for model_index, name in enumerate(('baseline', 'spatial')):
        col = model_index + 2
        bank = curves[model_index]
        summary = result[name][metric]
        for k, curve in enumerate(bank):
            for seed in np.linspace(-25, 25, 101):
                xx = curve[:, 0] + seed
                xx = np.where(np.abs(xx) <= 25, xx, np.nan)
                axes[0, col].plot(xx, curve[:, 1], color='.4', lw=.25, alpha=.008)
        for index, match in enumerate(summary['lanes']):
            p = by_json[match['json_index']]['points']
            curve = bank[match['template']].copy()
            curve[:, 0] += -25 + .5 * match['seed']
            supported = (curve[:, 1] >= p[0, 1]) & (curve[:, 1] <= p[-1, 1])
            curve = curve[supported]
            color = colors[index % 10]
            axes[0, col].plot(p[:, 0], p[:, 1], '--', color='.15', lw=1.2)
            clipped_x = np.where(np.abs(curve[:, 0]) <= 25, curve[:, 0], np.nan)
            axes[0, col].plot(clipped_x, curve[:, 1], color=color, lw=1.6)
            axes[1, col].plot(p[:, 1], p[:, 2], '--', color='.15', lw=1.2)
            axes[1, col].plot(curve[:, 1], curve[:, 2], color=color, lw=1.6)
            z_limits.extend(p[:, 2].tolist() + curve[:, 2].tolist())
            if model_index == 1:
                projection = calibration['ground_to_raw_img']
                axes[0, 0].plot(*visual.camera_points(p, projection, image).T, '--', color='white', lw=1)
                axes[0, 0].plot(*visual.camera_points(curve, projection, image).T, color=color, lw=1.6)
        title = 'Global pooled prior' if model_index == 0 else extractor
        axes[0, col].set_title(f'{title}: {summary["covered"]}/{summary["gt"]} GT', fontsize=10)
        axes[0, col].set(xlim=(-25, 25), ylim=(0, 100), xlabel='x (m)', ylabel='y (m)')
        axes[1, col].set_title(f'{title}: height vs raw GT', fontsize=10)
        axes[1, col].set(xlim=(0, 100), xlabel='y (m)', ylabel='z (m)')
        for row in range(2):
            axes[row, col].grid(alpha=.15)
    lo, hi = min(z_limits), max(z_limits)
    margin = max(.1, .1 * (hi - lo))
    for col in (2, 3):
        axes[1, col].set_ylim(lo - margin, hi + margin)
    axes[0, 0].set(xlim=(0, image.shape[1]), ylim=(image.shape[0], 0))
    evaluation_note = ('Official validation split; no validation data used for training. '
                       if result.get('evaluation_split') == 'validation'
                       else 'Target segments excluded from training. ')
    fig.text(.5, .025, evaluation_note + 'Dashed: raw GT. Faint: all 505 Q0 candidates. '
             'Solid: assigned Q0 per GT, selected for evaluation only. Coverage: ' + coverage_label + '.',
             ha='center', fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    sys.path.insert(0, str(visual.AUDIT_SCRIPTS))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    alignment = require_matplotlib_panel_alignment(fig, strict=True)
    with tempfile.TemporaryDirectory(prefix='q0-figure-qa-') as temporary:
        pdf = Path(temporary) / 'comparison.pdf'
        # Audit/preview files are temporary; each sample directory gets SVG only.
        fig.savefig(pdf, format='pdf')
        fig.savefig(Path(temporary) / 'comparison.png', dpi=300)
        for script in ('audit_pdf_text.py', 'audit_figure_collisions.py'):
            audit = subprocess.run([sys.executable, str(visual.AUDIT_SCRIPTS / script), str(pdf)],
                                   capture_output=True, text=True)
            if audit.returncode:
                raise RuntimeError(audit.stdout + audit.stderr)
        fig.savefig(output, format='svg')
    plt.close(fig)
    return fig, alignment


@torch.no_grad()
def evaluate_ply_example(path, models, cfg, device, render_dir=None, step=None):
    from plyfile import PlyData
    scene, prefix = path.parent.parent.name, path.parent.name
    label_root = Path(cfg.dataset_cfg.data_label_dir)
    labels = list((label_root / 'validation' / scene).glob(prefix + '*.json'))
    if len(labels) != 1:
        raise ValueError('PLY example must resolve to exactly one official frame')
    image_path = Path(cfg.dataset_cfg.data_image_dir) / 'validation' / scene / (labels[0].stem + '.jpg')
    batch = infer._build_sample(cfg, image_path)
    vertices = PlyData.read(path)['vertex'].data
    xyz = np.column_stack([vertices[k] for k in ('x', 'y', 'z')]).astype(np.float32)
    rgb = np.column_stack([vertices[k] for k in ('red', 'green', 'blue')]).astype(np.float32)
    norm = batch['img_metas'][0]['current']['img_norm_cfg']
    rgb = (rgb - np.asarray(norm['mean'])) / np.asarray(norm['std'])
    current_queue = batch['img'].shape[1] - 1
    values = np.column_stack((xyz, vertices['intensity'], vertices['elongation'], rgb,
                              np.full(len(xyz), current_queue))).astype(np.float32)
    painted = torch.from_numpy(values).to(device)
    current = batch['lidar'][0][batch['lidar'][0][:, -1] == current_queue]
    displayed = models[0].pts_bev_encoder.points_in_lane_ground([painted], batch['img_metas'])[0]
    roi_xyz = displayed[:, :3].cpu().numpy()
    roi = ((roi_xyz >= np.array([-25, 0, -5])) & (roi_xyz <= np.array([25, 100, 5]))).all(-1)
    distances = cKDTree(current[:, :3].numpy()).query(xyz[roi])[0]
    if not len(distances) or distances.max() > .01:
        raise ValueError('PLY points do not align with current-frame LiDAR within 1 cm')
    predictions = {}
    for mode in ('normal_temporal', 'provided_ply', 'provided_ply_far_only',
                 'provided_ply_flat_z', 'provided_ply_zero_rgb', 'provided_ply_shuffle_rgb'):
        records = {}
        for name, model in zip(('baseline', 'spatial'), models):
            if mode == 'normal_temporal':
                features = model.pts_bev_encoder([p.to(device) for p in batch['lidar']],
                    images=batch['img'].to(device), img_metas=batch['img_metas'])
            else:
                # Keep the original ROI fixed before any intervention, so
                # flattening height cannot introduce originally excluded points.
                points = painted[torch.as_tensor(roi, device=device)].clone()
                if mode == 'provided_ply_far_only':
                    lane_points = model.pts_bev_encoder.points_in_lane_ground([points], batch['img_metas'])[0]
                    points = points[lane_points[:, 1] >= 40]
                elif mode == 'provided_ply_flat_z':
                    points[:, 2] = 0
                elif mode == 'provided_ply_zero_rgb':
                    points[:, 5:8] = 0
                elif mode == 'provided_ply_shuffle_rgb':
                    valid_rgb = points[:, 5:8].abs().sum(-1) > 0
                    colors = points[valid_rgb, 5:8].clone()
                    order = torch.randperm(len(colors), generator=torch.Generator().manual_seed(7))
                    points[valid_rgb, 5:8] = colors[order.to(device)]
                if not len(points):
                    raise ValueError('Provided PLY contains no far points')
                with patch.object(model.pts_bev_encoder, 'paint_points', return_value=[points]):
                    features = model.pts_bev_encoder([points], img_metas=batch['img_metas'])
            records[name] = model.anchor_prior(features)
        # GT is loaded only after the input-only inference above.
        calibration = batch['img_metas'][0]['current']['openlane_calib']
        lanes = raw_lanes(batch['raw_label'][0], calibration)
        far_lanes = [dict(l, points=l['points'][l['points'][:, 1] >= 40]) for l in lanes]
        far_lanes = [l for l in far_lanes if len(l['points']) >= 2]
        predictions[mode] = {name: dict(all_visible_gt=evaluate_controls(p, lanes),
                                      far_visible_gt=evaluate_controls(p, far_lanes) if far_lanes else None)
                             for name, p in records.items()}
        if render_dir is not None and mode == 'provided_ply':
            result = {name: values['all_visible_gt'] for name, values in predictions[mode].items()}
            if type(models[1].anchor_prior).__name__ == 'ContinuousBEVShapePrior':
                result['extractor_label'] = 'Continuous path prior'
            elif type(models[1].anchor_prior).__name__ == 'GuidedSpatialBEVShapePrior':
                result['extractor_label'] = ('Geometric lane-field prior' if models[1].anchor_prior.geometric_readout
                                             else 'Lane-field guided spatial prior')
            output = render_dir / scene / image_path.stem / 'q0_feature_comparison.svg'
            render_comparison(image_path, displayed.cpu().numpy(), calibration, lanes,
                              (records['baseline'], records['spatial']), result, output, step)
        print('PLY EVAL', mode, 'complete',
              [predictions[mode][name]['all_visible_gt']['paired_xz_p80']['covered']
               for name in ('baseline', 'spatial')], flush=True)
    return dict(ply=str(path), frame=image_path.stem, points=len(painted),
                far_points=int((displayed[:, 1] >= 40).sum()),
                xyz_alignment_max_m=float(distances.max()), predictions=predictions)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--spatial', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--validation-scenes', type=Path,
                        help='Evaluate only the official validation section of this scene list')
    parser.add_argument('--expected-gt', type=int,
                        help='Independently audited raw GT count, required for a validation goal gate')
    parser.add_argument('--require-coverage-goal', action='store_true',
                        help='Require selected criteria to exceed baseline and cover strictly over 90% of all audited GT')
    parser.add_argument('--goal-metric', choices=('paired_xz_p80', 'paired_xz_p75_03',
                                                'paired_xz_p80_05', 'paired_xz_p75_05'),
                        help='Select one acceptance criterion; all four remain reported (default: require all four)')
    parser.add_argument('--allow-lane-field-loss-difference', action='store_true',
                        help='Explicitly compare architecture plus lane-field supervision; retain both protocols')
    parser.add_argument('--ply', type=Path, help='Also inspect the supplied colored PLY and its far-only points')
    args = parser.parse_args()
    if args.validation_scenes and args.require_coverage_goal and not args.expected_gt:
        parser.error('--validation-scenes with --require-coverage-goal requires --expected-gt')
    torch.cuda.set_device(args.device)
    torch.set_num_threads(4)
    register_all_modules()
    old, cfg, old_step, old_protocol = load_model(args.baseline, args.device)
    new, new_cfg, new_step, new_protocol = load_model(args.spatial, args.device)
    validate_comparison_input_frames(cfg, new_cfg)
    extractor_type = new_protocol['model_type']
    if (old_protocol['model_type'] != 'LiDARShapeTokenPrior'
            or extractor_type not in ('SpatialBEVShapePrior', 'ContinuousBEVShapePrior',
                                      'GuidedSpatialBEVShapePrior')):
        raise ValueError('Baseline/spatial checkpoint types are incorrect')
    if old_step != new_step:
        raise ValueError(f'Compare equal training budgets: {old_step} != {new_step}')
    validate_comparison_protocols(old_protocol, new_protocol, args.allow_lane_field_loss_difference)
    selection = read_validation_scenarios(args.validation_scenes) if args.validation_scenes else None
    split = 'validation' if selection else 'training'
    if selection:
        for checkpoint, protocol in ((args.baseline, old_protocol), (args.spatial, new_protocol)):
            validate_validation_training_manifest(checkpoint, protocol, selection)
    dataset_cfg = infer._inference_dataset_cfg(cfg)
    dataset_cfg.update(split=split, test_mode=True)
    dataset_cfg['dataset_cfg'].update(excluded_segments=[], load_interval=1)
    dataset_cfg['dataset_cfg']['train_pipeline'] = dataset_cfg['dataset_cfg']['val_pipeline']
    dataset = DATASETS.build(dataset_cfg)
    index = {(p.parent.name, p.stem) if selection else p.stem: i
             for i, p in enumerate(dataset.label_files)}
    if selection:
        groups = {}
        for record in selection:
            if (record['segment'], record['frame']) not in index or not Path(record['image']).is_file():
                raise ValueError(f'Missing validation image/label: {record["image"]}')
            groups.setdefault(record['group'], []).append(record)
    else:
        groups = {g: [dict(frame=f) for f in frames] for g, frames in cfg.q0_target_frames.items()}
    results = []
    args.work_dir.mkdir(parents=True, exist_ok=True)
    for group, records in groups.items():
        for record in records:
            frame = record['frame']
            sample_index = index[(record['segment'], frame) if selection else frame]
            batch = openlane_collate([dataset[sample_index]])
            inputs = dict(images=batch['img'].to(args.device), img_metas=batch['img_metas'])
            lidar = [p.to(args.device) for p in batch['lidar']]
            # No lanes, category, frame ID or annotation fields enter either model.
            painted = []
            painter = old.pts_bev_encoder.paint_points
            def capture(*call_args, **call_kwargs):
                value = painter(*call_args, **call_kwargs)
                if args.render:
                    display = old.pts_bev_encoder.points_in_lane_ground(value, call_kwargs.get('img_metas'))
                    painted.append(display[0].detach().cpu().numpy())
                return value
            with patch.object(old.pts_bev_encoder, 'paint_points', side_effect=capture):
                old_pred = old.anchor_prior(old.pts_bev_encoder(lidar, **inputs))
            new_pred = new.anchor_prior(new.pts_bev_encoder(lidar, **inputs))
            label = batch['raw_label'][0]
            calibration = dataset._build_calib(label)
            lanes = raw_lanes(label, calibration)
            if not lanes:
                raise ValueError(f'Target {frame} has no raw GT; it cannot count as passed')
            result = dict(frame=frame, group=group, evaluation_split=split, raw_gt=len(lanes),
                          extractor_label=({'ContinuousBEVShapePrior': 'Continuous path prior',
                                            'GuidedSpatialBEVShapePrior': 'Lane-field guided spatial prior'}
                                           .get(extractor_type, 'Spatial ordered prior')),
                          packed_gt=int((batch['gt_lanes_visibility'][0].sum(-1) >= 2).sum()),
                          gt_geometry_audit=audit_cubic_gt(lanes),
                          baseline=evaluate_controls(old_pred, lanes),
                          spatial=evaluate_controls(new_pred, lanes))
            if getattr(new.anchor_prior, 'geometric_readout', False):
                result['extractor_label'] = 'Geometric lane-field prior'
            if getattr(new.anchor_prior, 'curvature_prior_strength', 0):
                result['extractor_label'] = 'Global-curvature guided prior'
                result['global_curve_evidence'] = dict(
                    second_control_differences_m=new_pred['lane_curvature_prior'][0].detach().cpu().tolist(),
                    precisions=new_pred['lane_curvature_precision'][0].detach().cpu().tolist(),
                    fitted_lateral_origins_m=new_pred['lane_row_x_origin'][0].detach().cpu().tolist())
            field_metrics = evaluate_lane_field(new_pred, lanes, new.proposal_expander.pc_range)
            if field_metrics is not None:
                result['lane_field_evidence'] = field_metrics
            results.append(result)
            if args.render:
                result['visualized_metric'] = args.goal_metric or 'paired_xz_p80'
                image_path = Path(dataset.image_root) / label['file_path']
                figure_path = args.work_dir / 'svg' / image_path.parent.name / frame / 'q0_feature_comparison.svg'
                _, result['figure_alignment'] = render_comparison(
                    image_path, painted[0], calibration, lanes,
                    (old_pred, new_pred), result, figure_path, old_step)
            print('EVAL', frame, 'raw GT', len(lanes), 'baseline/spatial p80',
                  result['baseline']['paired_xz_p80']['covered'],
                  result['spatial']['paired_xz_p80']['covered'], 'p75<0.5m',
                  result['baseline']['paired_xz_p75_05']['covered'],
                  result['spatial']['paired_xz_p75_05']['covered'], flush=True)
    output = dict(training_steps=old_step, protocol=old_protocol, samples=results,
                  evaluation_split=split, group_coverage=grouped_coverage(results),
                  comparison_protocols=dict(baseline=old_protocol, spatial=new_protocol),
                  lane_field_supervision_ablation=args.allow_lane_field_loss_difference,
                  extractor_type=extractor_type,
                  checkpoints=dict(baseline=str(args.baseline), spatial=str(args.spatial)),
                  criterion='raw visible JSON points inside model ROI; distinct Q0 candidate per GT',
                  scope='Q0 geometry only, not final detection Recall/F1')
    if old.checkpoint_postprocessing or new.checkpoint_postprocessing:
        output['checkpoint_postprocessing'] = dict(baseline=old.checkpoint_postprocessing,
                                                   spatial=new.checkpoint_postprocessing)
    if selection:
        output['validation_scene_list'] = dict(
            path=str(args.validation_scenes),
            sha256=hashlib.sha256(args.validation_scenes.read_bytes()).hexdigest(),
            frames=len(selection), expected_raw_gt=args.expected_gt)
    if getattr(new.anchor_prior, 'geometric_readout', False):
        output['curve_readout'] = 'Predicted row x/z -> cubic least squares; relative x and absolute z; 5x101 queries'
    if getattr(new.anchor_prior, 'curvature_prior_strength', 0):
        output['curve_readout'] = 'Predicted rows plus BEV-predicted curvature and confidence -> weighted cubic fit; 5x101 queries'
        output['global_curvature_prior'] = dict(
            base_strength=new.anchor_prior.curvature_prior_strength,
            control_scale_m=new.anchor_prior.curvature_prior_scale,
            supervision='Existing raw Q0 geometry and lane-field losses; no additional curvature GT loss')
    if args.ply:
        output['ply_example'] = evaluate_ply_example(
            args.ply, (old, new), cfg, args.device,
            render_dir=args.work_dir / 'svg' if args.render else None, step=old_step)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.require_coverage_goal:
        goal = coverage_goal_report(output, [r['frame'] for records in groups.values() for r in records],
                                    args.expected_gt if selection else 74,
                                    required_metrics=[args.goal_metric] if args.goal_metric else None)
        output['coverage_goal'] = goal
    (args.work_dir / f'comparison_step_{old_step}.json').write_text(json.dumps(output, indent=2))
    if args.require_coverage_goal:
        print('COVERAGE_GOAL', json.dumps(goal), flush=True)
        if not goal['passed']:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
