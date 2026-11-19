"""Trace OpenLane curve proposals and decoder refinement without changing training.

The default samples/checkpoint are the three requested validation frames and the
20260924 experiment's best epoch 43. See QUERY_STAGES.md for interpretation.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
import sys
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.parent))
from LaneLoom.vis.output import infer_openlane_sample as infer
from LaneLoom.models.head.curve_nms import curve_nms, mean_distance_curve_nms

plt = infer.plt
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                     'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'font.size': 8})
DEFAULT_CHECKPOINT = REPO_ROOT.parent / (
    'LaneLoom-removed-experiments/openlane-temporal-legacy/work_dirs/'
    'laneloom_openlane_image_lidar_temporal_0923_z_confidence_full_50e/'
    'exp_date_20260924_174958/checkpoints/best_OpenLaneOfficial_F1_epoch_43.pth')
DEFAULT_IMAGES = [REPO_ROOT / 'data/openlane/images/validation' / scene / frame
                  for scene, frame in (
    ('segment-17962792089966876718_2210_933_2230_933_with_camera_labels',
     '150914058009157400.jpg'),
    ('segment-17135518413411879545_1480_000_1500_000_with_camera_labels',
     '155294852943939800.jpg'),
    ('segment-14931160836268555821_5778_870_5798_870_with_camera_labels',
     '150733834142979700.jpg'))]
AUDIT_SCRIPTS = Path('/root/.codex/skills/nature-figure/scripts')


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items() if k != 'bev'}
    if isinstance(value, (tuple, list)):
        return [cpu(v) for v in value]
    return value


class StageCapture:
    """Local forward hooks; masks hide a modality from decoder cross attention."""

    def __init__(self, head, mode='joint'):
        self.head, self.mode = head, mode
        self.data, self.handles = {}, []

    def __enter__(self):
        def store(name):
            def hook(module, args, output):
                self.data[name] = cpu(output)
                if name == 'prior':
                    features = module._feature(args[0])
                    self.data['bev_mean_abs'] = features.detach().abs().mean(1).cpu()
            return hook

        for name, module in (
                ('prior', self.head.anchor_prior),
                ('expanded', self.head.proposal_expander),
                ('decoder', self.head.lane_decoder)):
            self.handles.append(module.register_forward_hook(store(name)))

        def sampler_hook(module, args, output):
            output = dict(output)
            mask = output['key_padding_mask'].clone()
            lidar_count = output['lidar_valid_mask'].shape[-1]
            if self.mode == 'image_only':
                mask[..., -lidar_count:] = True
            elif self.mode == 'bev_only':
                mask[..., :-lidar_count] = True
            output['key_padding_mask'] = mask
            self.data.setdefault('sampling', []).append(cpu(dict(
                reference_points=args[0], image_valid_mask=output['image_valid_mask'],
                lidar_valid_mask=output['lidar_valid_mask'], key_padding_mask=mask)))
            return output

        self.handles.append(self.head.query_sampler.register_forward_hook(sampler_hook))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()


def postprocess(prediction, head):
    """Apply the checkpoint's threshold and NMS independently at each layer."""
    scores = prediction['pred_cls'].sigmoid()
    visibility = prediction['pred_vis'].sigmoid()
    cfg = head.curve_nms_cfg or {}
    vt = float(cfg.get('visibility_threshold', .5))
    valid = (scores >= head.score_threshold) & (visibility >= vt).any(-1)
    if any(k in cfg for k in ('iou_threshold', 'containment_threshold')):
        valid &= curve_nms(
            prediction['pred_x'], prediction['pred_z'], visibility, scores,
            tube_radius=float(cfg.get('tube_radius', .5)),
            z_weight=float(cfg.get('z_weight', 1.)),
            epsilon=float(cfg.get('epsilon', 1e-6)),
            iou_threshold=float(cfg.get('iou_threshold', .6)),
            containment_threshold=float(cfg.get('containment_threshold', .9)),
            candidate_mask=valid)
    if cfg.get('mean_distance_threshold') is not None:
        valid &= mean_distance_curve_nms(
            prediction['pred_x'], prediction['pred_z'], visibility, scores,
            distance_threshold=float(cfg['mean_distance_threshold']),
            visibility_threshold=vt, candidate_mask=valid)
    return dict(x=prediction['pred_x'], z=prediction['pred_z'],
                visibility=visibility, scores=scores, valid_queries=valid,
                category=prediction['pred_category'].argmax(-1))


def stages_from_capture(data, head):
    decoder = data['decoder']
    layers = decoder['aux_predictions'] + [decoder]
    count = data['expanded']['curve_x_controls'].shape[1]
    ids = torch.arange(count)
    selected = decoder.get('selected_query_indices', ids[None])[0]
    basis = head.lane_decoder.bezier_basis.detach().cpu()
    initial = dict(name='Q0: BEV proposals', query_ids=ids,
                   x=torch.einsum('ac,bqc->bqa', basis,
                                  data['expanded']['curve_x_controls']),
                   z=torch.einsum('ac,bqc->bqa', basis,
                                  data['expanded']['curve_z_controls']))
    result = [initial]
    for i, layer in enumerate(layers):
        query_ids = ids if i == 0 else selected
        result.append(dict(name=f'Layer {i + 1}: joint refinement',
                           query_ids=query_ids,
                           curve_x_controls=layer['curve_x_controls'],
                           curve_z_controls=layer['curve_z_controls'],
                           **postprocess(layer, head)))
    return result


def geometry(stage, y):
    x, z = stage['x'][0].numpy(), stage['z'][0].numpy()
    return np.stack((x, np.broadcast_to(y, x.shape), z), axis=-1)


def camera_points(curve, projection, image):
    uv = infer._project(curve, projection)
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] <= image.shape[1])
              & (uv[:, 1] >= 0) & (uv[:, 1] <= image.shape[0]))
    uv[~inside] = np.nan
    return uv


def curves_for_stage(stage, y, tracked_ids=None, tracked_visibility=None, vt=.5):
    xyz = geometry(stage, y)
    ids = stage['query_ids'].numpy()
    if tracked_ids is None:
        indices = np.flatnonzero(stage['valid_queries'][0].numpy())
        masks = stage['visibility'][0].numpy() >= vt
    else:
        index_by_id = {int(q): i for i, q in enumerate(ids)}
        indices = [index_by_id[int(q)] for q in tracked_ids]
        masks = np.zeros(xyz.shape[:2], dtype=bool)
        masks[indices] = tracked_visibility
    result = []
    for index in indices:
        points = xyz[index].copy()
        points[~masks[index]] = np.nan  # preserve gaps; never join invisible anchors
        if np.isfinite(points).all(-1).sum() >= 2:
            result.append((int(ids[index]), points))
    return result


def save_figure(fig, path):
    """Deliver only SVG; audit intermediates live in an automatically removed directory."""
    fig.canvas.draw()
    with tempfile.TemporaryDirectory(prefix='laneloom-figure-qa-') as temporary:
        qa_root = Path(temporary)
        if AUDIT_SCRIPTS.is_dir():
            sys.path.insert(0, str(AUDIT_SCRIPTS))
            from audit_panel_alignment import require_matplotlib_panel_alignment
            require_matplotlib_panel_alignment(fig, strict=True)
            pdf = qa_root / 'figure.pdf'
            fig.savefig(pdf)
            for script in ('audit_pdf_text.py', 'audit_figure_collisions.py'):
                result = subprocess.run(
                    [sys.executable, str(AUDIT_SCRIPTS / script), str(pdf)],
                    capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
        fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def paired_errors(stages, y, gt, tracked_ids, visibility):
    """Fixed final-prediction/GT assignment and support for descriptive errors.

    Includes every final detection in assignment, with no distance-based cherry
    picking. Unmatched detections/GT are reported; these are not official metrics.
    """
    from scipy.optimize import linear_sum_assignment
    final = geometry(stages[-1], y)
    lookup = {int(q): i for i, q in enumerate(stages[-1]['query_ids'])}
    support, costs = {}, np.full((len(tracked_ids), len(gt)), 1e6)
    for i, q in enumerate(tracked_ids):
        for j, lane in enumerate(gt):
            order = np.argsort(lane[:, 1])
            lane = lane[order]
            mask = visibility[i] & (y >= lane[0, 1]) & (y <= lane[-1, 1])
            if mask.sum() < 2:
                continue
            target = np.column_stack([np.interp(y[mask], lane[:, 1], lane[:, k])
                                      for k in (0, 2)])
            support[i, j] = (mask, target)
            costs[i, j] = np.linalg.norm(final[lookup[int(q)], mask][:, [0, 2]]
                                        - target, axis=-1).mean()
    rows, cols = linear_sum_assignment(costs)
    pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if (int(i), int(j)) in support]
    result = dict(assignment='Hungarian on final mean xz distance; no distance cutoff',
                  final_detections=len(tracked_ids), gt_lanes=len(gt),
                  unmatched_detections=len(tracked_ids) - len(pairs),
                  unmatched_gt=len(gt) - len(pairs), pairs=[])
    for i, j in pairs:
        q = int(tracked_ids[i])
        mask, target = support[i, j]
        record = dict(query_id=q, gt_index=j, anchor_count=int(mask.sum()), stages=[])
        for stage in stages:
            index = stage['query_ids'].tolist().index(q)
            difference = geometry(stage, y)[index, mask][:, [0, 2]] - target
            record['stages'].append(dict(name=stage['name'],
                x_mae_m=float(np.abs(difference[:, 0]).mean()),
                z_mae_m=float(np.abs(difference[:, 1]).mean()),
                xz_distance_m=float(np.linalg.norm(difference, axis=-1).mean())))
        result['pairs'].append(record)
    return result


def select_tracked_queries(stage, vt=.5):
    """Keep detections; if none exist, show scored candidates with an explicit label."""
    valid = stage['valid_queries'][0]
    indices = torch.where(valid)[0]
    fallback = not indices.numel()
    visibility = stage['visibility'][0] >= vt
    if fallback:
        supported = torch.where(visibility.sum(-1) >= 2)[0]
        pool = supported if supported.numel() else torch.arange(valid.numel())
        order = stage['scores'][0, pool].argsort(descending=True)[:5]
        indices = pool[order]
    masks = visibility[indices].numpy().copy()
    if fallback:
        masks[masks.sum(-1) < 2] = True
    return indices.numpy(), masks, fallback


def render_query_process(data, stages, interventions, y, image, gt, projection,
                         ids, visible, vt, path, fallback=False, reference_ply=None):
    """One SVG shows real template extraction, tracked refinement and interventions."""
    errors = paired_errors(stages, y, gt, ids, visible)
    means = [np.mean([p['stages'][i]['xz_distance_m'] for p in errors['pairs']])
             if errors['pairs'] else np.nan for i in range(len(stages))]
    fig, axes = plt.subplots(4, 4, figsize=(15, 13),
                             gridspec_kw={'height_ratios': [2.4, 4, 2, 3]})
    fig.subplots_adjust(left=.065, right=.985, bottom=.085, top=.93,
                        wspace=.33, hspace=.55)
    palette = {int(q): plt.get_cmap('tab20')(i % 20) for i, q in enumerate(ids)}
    templates = data['prior']['shape_points'][0].numpy()
    if reference_ply is None:
        axes[0, 0].imshow(data['bev_mean_abs'][0], origin='lower',
                          extent=(-25, 25, 0, 100), cmap='Greys', aspect='auto')
        title = 'BEV feature: mean absolute activation'
    else:
        from plyfile import PlyData
        vertices = PlyData.read(reference_ply)['vertex'].data
        rgb = np.column_stack([vertices[k] for k in ('red', 'green', 'blue')]) / 255.
        axes[0, 0].scatter(vertices['x'], vertices['y'], c=rgb, s=.3,
                           linewidths=0, rasterized=True)
        for lane in gt:
            axes[0, 0].plot(lane[:, 0], lane[:, 1], '--', color='.25', lw=.8)
        for q, curve in curves_for_stage(stages[0], y, ids, visible, vt):
            axes[0, 0].plot(curve[:, 0], curve[:, 1], color=palette[q], lw=1.2)
        title = 'Your PLY + tracked Q0 queries\nPLY: single FOV; model: 3 frames'
    axes[0, 0].set(title=title, xlim=(-25, 25), ylim=(0, 100), xlabel='x (m)', ylabel='y (m)')
    for i, curve in enumerate(templates):
        color = plt.get_cmap('tab10')(i)
        axes[1, 0].plot(curve[:, 0], curve[:, 1], color=color, lw=1.7,
                        label=f'Template {i}')
        axes[2, 0].plot(curve[:, 1], curve[:, 2], color=color, lw=1.7)
    template_x = templates[..., 0]
    axes[1, 0].set(title=f'{len(templates)} predicted relative templates',
                   xlim=(float(template_x.min()) - 1, float(template_x.max()) + 1),
                   ylim=(0, 100), xlabel='relative x (m)', ylabel='y (m)')
    axes[2, 0].set(title='Predicted template height',
                   xlim=(0, 100), xlabel='y (m)', ylabel='z (m)')
    height_curves = [lane[:, 2] for lane in gt]
    for column, stage in enumerate(stages, 1):
        curves = curves_for_stage(stage, y, ids, visible, vt)
        camera, bev, height = axes[:3, column]
        camera.imshow(image)
        camera.set_aspect('equal', adjustable='datalim')
        camera.set_axis_off()
        name = ('Q0: tracked queries (subset of 505)' if column == 1
                else f'Iteration {column - 1}: FV + BEV refinement')
        camera.set_title(name, fontsize=9)
        if column == 1:
            # Q0 coverage must show the entire Cartesian product, not just
            # final survivors. Faint background colors identify template families.
            template_ids = data['expanded']['selected_template_indices'][0]
            for q, curve in enumerate(geometry(stage, y)):
                points = curve.copy()
                points[(points[:, 0] < -25) | (points[:, 0] > 25)] = np.nan
                bev.plot(points[:, 0], points[:, 1],
                         color=plt.get_cmap('tab10')(int(template_ids[q])),
                         lw=.45, alpha=.09, zorder=1)
        for lane in gt:
            camera.plot(*camera_points(lane, projection, image).T, '--', color='white', lw=1.2)
            bev.plot(lane[:, 0], lane[:, 1], '--', color='.25', lw=1.3)
            height.plot(lane[:, 1], lane[:, 2], '--', color='.25', lw=1.3)
        for q, curve in curves:
            camera.plot(*camera_points(curve, projection, image).T, color=palette[q], lw=1.8)
            bev.plot(curve[:, 0], curve[:, 1], color=palette[q], lw=1.7)
            height.plot(curve[:, 1], curve[:, 2], color=palette[q], lw=1.7)
            height_curves.append(curve[:, 2])
        camera.set(xlim=(0, image.shape[1]), ylim=(image.shape[0], 0))
        metric = (f'All 505 Q0 + tracked subset\nPaired error: {means[0]:.3f} m' if column == 1
                  else f'Mean paired xz error: {means[column - 1]:.3f} m'
                  if np.isfinite(means[column - 1]) else 'No supported GT pairs')
        bev.set(title=metric, xlim=(-25, 25), ylim=(0, 100), xlabel='x (m)', ylabel='y (m)')
        height.set(title='Query height vs GT', xlim=(0, 100), xlabel='y (m)', ylabel='z (m)')
    finite = np.concatenate(height_curves) if height_curves else np.array([0.])
    finite = finite[np.isfinite(finite)]
    limits = (float(finite.min()) - .15, float(finite.max()) + .15) if finite.size else (-1, 1)
    for axis in axes[2, 1:]:
        axis.set_ylim(*limits)
    axes[3, 0].plot(range(len(stages)), means, 'o-', color='#0072b2', lw=1.7)
    axes[3, 0].set(title='Tracked query error (fixed GT pairs)',
                   xticks=range(len(stages)), xticklabels=['Q0', 'Iter 1', 'Iter 2'],
                   ylabel='Mean xz distance (m)')
    for column, (mode, stage) in enumerate(interventions, 1):
        axis = axes[3, column]
        for lane in gt:
            axis.plot(lane[:, 0], lane[:, 1], '--', color='.25', lw=1.3)
        indices, masks, candidate_fallback = select_tracked_queries(stage, vt)
        candidate_ids = stage['query_ids'].numpy()[indices]
        curves = curves_for_stage(stage, y, candidate_ids, masks, vt)
        for q, curve in curves:
            axis.plot(curve[:, 0], curve[:, 1], color='#0072b2', lw=1.6)
        title = {'joint': 'Joint FV + BEV evidence', 'image_only': 'FV evidence only',
                 'bev_only': 'BEV evidence only'}[mode]
        count_label = f'0 detections; {len(curves)} candidates' if candidate_fallback else f'{len(curves)} lanes'
        axis.set(title=f'{title}\n{count_label}', xlim=(-25, 25), ylim=(0, 100),
                 xlabel='x (m)', ylabel='y (m)')
    for row in (1, 2, 3):
        for axis in axes[row]:
            axis.grid(alpha=.2)
    fig.suptitle('BEV templates and iterative curve-query refinement | Epoch 43', fontsize=13)
    handles = [infer.Line2D([0], [0], color='.25', ls='--', lw=1.5)]
    labels = ['GT reference']
    if len(ids) <= 12:
        handles += [infer.Line2D([0], [0], color=palette[int(q)], lw=2) for q in ids]
        labels += [f'q{int(q)}' for q in ids]
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .025),
               ncol=min(7, len(labels)), frameon=False, fontsize=8)
    fig.text(.5, .006,
             f'Tracked {"below-threshold candidates" if fallback else "final detections"}: {len(ids)} | GT pairs: {len(errors["pairs"])} | '
             f'Unpaired GT: {errors["unmatched_gt"]} | '
             'Fixed support; raw curves; RGB-painted LiDAR BEV; inference interventions',
             ha='center', fontsize=8)
    save_figure(fig, path)
    print('Paired mean xz errors (Q0, Iter 1, Iter 2):', means, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, nargs='+', default=DEFAULT_IMAGES)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-root', type=Path, default=REPO_ROOT / 'vis/output/query_stages')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--reference-ply', type=Path,
                        help='Show one matching PLY behind Q0; requires exactly one --images path.')
    args = parser.parse_args()
    if args.reference_ply is not None and len(args.images) != 1:
        parser.error('--reference-ply requires exactly one --images path.')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; real curve-query inference requires a working GPU.')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = Config.fromfile(args.config) if args.config else Config.fromstring(saved['meta']['cfg'], '.py')
    del saved
    model = infer._build_model(cfg, args.checkpoint, device)
    if model.lane_head.lane_decoder.num_layers != 2:
        raise ValueError('This comparison layout requires the two-layer epoch 43 decoder.')
    for image_path in args.images:
        image_path = image_path.resolve()
        batch = infer._build_sample(cfg, image_path)
        image = np.asarray(Image.open(image_path).convert('RGB'))
        gt = infer._ground_truth_lanes(batch)
        projection = batch['img_metas'][0]['current']['openlane_calib']['ground_to_raw_img']
        y = np.asarray(cfg.model.lane_head.lane_decoder.anchor_y_steps)
        runs = {}
        for mode in ('joint', 'image_only', 'bev_only'):
            with StageCapture(model.lane_head, mode) as capture:
                prediction = infer._run_inference(model, batch, device)
            stages = stages_from_capture(capture.data, model.lane_head)
            for key in ('x', 'z', 'visibility', 'scores', 'valid_queries'):
                torch.testing.assert_close(stages[-1][key], prediction[key])
            runs[mode] = (capture.data, stages, prediction)
        data, stages, prediction = runs['joint']
        vt = float((model.lane_head.curve_nms_cfg or {}).get('visibility_threshold', .5))
        final_indices, visible, fallback = select_tracked_queries(stages[-1], vt)
        ids = stages[-1]['query_ids'].numpy()[final_indices]
        output = infer._sample_output_dir(image_path, args.output_root.resolve())
        output.mkdir(parents=True, exist_ok=True)
        path = output / 'query_refinement.svg'
        render_query_process(data, stages,
                             [(mode, runs[mode][1][-1]) for mode in ('image_only', 'bev_only', 'joint')],
                             y, image, gt, projection, ids, visible, vt, path, fallback,
                             args.reference_ply)
        print('Saved:', path, flush=True)


if __name__ == '__main__':
    main()
