"""GT-only feasibility probes; these curves are never model predictions."""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from mmengine.config import Config
from scipy.optimize import linear_sum_assignment, lsq_linear

from LaneLoom.datasets.openlane.dataset import OpenLaneDataset
from LaneLoom.datasets.openlane.full_shape_targets import build_full_shape_targets
from LaneLoom.tools.evaluate_q0_prior import cubic_basis, raw_lanes
from LaneLoom.tools.q0_scenarios import read_validation_scenarios


def fit_relative_cubic(points, ratio=.8, iterations=8, free_height_origin=False, initial_subset=None,
                       lateral_control_bound=None, height_control_bound=None):
    """Construct a feasible origin-relative cubic, trying the actual 101 seeds.

    Trimmed least squares is a constructive probe, not a proof of the optimum.
    No parameters or seeds produced here may enter inference or training.
    """
    basis = cubic_basis(points[:, 1])[:, 1:]
    seeds = np.linspace(-25., 25., 101)
    target = np.broadcast_to(points[:, (0, 2)], (101, len(points), 2)).copy()
    target[:, :, 0] -= seeds[:, None]
    z_basis = cubic_basis(points[:, 1]) if free_height_origin else basis
    initial = np.arange(len(points)) if initial_subset is None else np.asarray(initial_subset)
    x_controls = np.einsum('cn,sn->sc', np.linalg.pinv(basis[initial]), target[:, initial, 0])
    z_controls = np.einsum('cn,sn->sc', np.linalg.pinv(z_basis[initial]), target[:, initial, 1])
    def solve(matrix, values, bound):
        controls = np.linalg.lstsq(matrix, values, rcond=None)[0]
        if bound is not None and np.any(np.abs(controls) > bound):
            controls = lsq_linear(matrix, values, bounds=(-bound, bound), method='bvls').x
        return controls

    for controls, matrix, coordinate, bound in ((x_controls, basis, 0, lateral_control_bound),
                                               (z_controls, z_basis, 1, height_control_bound)):
        if bound is not None:
            for index in np.flatnonzero(np.max(np.abs(controls), axis=-1) > bound):
                controls[index] = solve(matrix[initial], target[index, initial, coordinate], bound)
    required = math.ceil(ratio * len(points))
    best = None
    for iteration in range(iterations + 1):
        dx = np.einsum('nc,sc->sn', basis, x_controls) - target[..., 0]
        dz = np.einsum('nc,sc->sn', z_basis, z_controls) - target[..., 1]
        distances = np.hypot(dx, dz)
        errors = np.partition(distances, required - 1, axis=-1)[:, required - 1]
        selected = int(errors.argmin())
        if best is None or errors[selected] < best['p80_xz_m']:
            best = dict(seed=float(seeds[selected]), p80_xz_m=float(errors[selected]),
                        x_controls=np.r_[0., x_controls[selected]].tolist(),
                        z_controls=(z_controls[selected] if free_height_origin
                                    else np.r_[0., z_controls[selected]]).tolist())
        if iteration == iterations:
            break
        indices = np.argsort(distances, axis=-1)[:, :required]
        for seed_index, keep in enumerate(indices):
            x_controls[seed_index] = solve(basis[keep], target[seed_index, keep, 0], lateral_control_bound)
            z_controls[seed_index] = solve(z_basis[keep], target[seed_index, keep, 1], height_control_bound)
    return best


def dictionary_coverage(candidates, lanes, ratio=.8, threshold=.3):
    """Best feasible subset of up to five fitted templates, all 101 translations."""
    count = len(candidates)
    seeds = np.linspace(-25., 25., 101)
    costs = []
    for lane in lanes:
        points = lane['points']
        b = cubic_basis(points[:, 1])
        x = b @ np.asarray([c['x_controls'] for c in candidates]).T
        z = b @ np.asarray([c['z_controls'] for c in candidates]).T
        error = np.hypot(x.T[:, None] + seeds[None, :, None] - points[None, None, :, 0],
                         z.T[:, None] - points[None, None, :, 2])
        index = math.ceil(ratio * len(points)) - 1
        costs.append(np.partition(error, index, axis=-1)[..., index])
    costs = np.stack(costs)
    best = None
    for selected in itertools.combinations(range(count), min(5, count)):
        cost = costs[:, selected].reshape(len(lanes), -1)
        rows, columns = linear_sum_assignment((cost >= threshold) * 1e6 + np.minimum(cost, 1e3))
        covered = int((cost[rows, columns] < threshold).sum())
        score = (-covered, float(np.minimum(cost[rows, columns], 1e3).sum()))
        if best is None or score < best[0]:
            matches = [dict(json_index=lanes[row]['json_index'],
                            source_template=int(selected[column // 101]),
                            seed=float(seeds[column % 101]), p80_xz_m=float(cost[row, column]),
                            covered=bool(cost[row, column] < threshold))
                       for row, column in zip(rows, columns)]
            best = (score, dict(covered=covered, total=len(lanes),
                               selected_templates=list(selected), lanes=matches))
    return best[1]


def audit_training_targets(cfg, samples):
    """Replay exactly the configured annotation-only target construction."""
    arguments = cfg.train_dataloader.dataset.to_dict()
    arguments.pop('type')
    dataset = OpenLaneDataset(**arguments)
    basis = cubic_basis(dataset.anchor_y_steps)
    totals = dict(packed_lanes=0, full_lanes=0, perfect_template_coverage=0)
    for sample in samples:
        file = next((dataset.label_root / 'training').glob(f'*/{sample["frame"]}.json'))
        packed = dataset._parse_openlane_gt(json.loads(file.read_text()))
        target = build_full_shape_targets(
            packed['gt_lanes_3d'].numpy(), packed['gt_lanes_visibility'].numpy(),
            dataset.anchor_y_steps, max_templates=dataset.max_shape_templates,
            min_lane_points=dataset.full_shape_min_points, min_lane_span=dataset.full_shape_min_span,
            guarded_short_merge=dataset.full_shape_guarded_short_merge,
            z_extrapolation_grace_m=dataset.full_shape_z_extrapolation_grace_m,
            z_extrapolation_decay_m=dataset.full_shape_z_extrapolation_decay_m,
            visible_fit_tolerance_m=dataset.full_shape_visible_fit_tolerance_m)
        candidates, residuals = [], []
        for curve in target['gt_shape_templates'][target['gt_shape_template_valid']]:
            controls = np.linalg.lstsq(basis, curve[:, (0, 2)], rcond=None)[0]
            residuals.append(float(np.linalg.norm(basis @ controls - curve[:, (0, 2)], axis=-1).max()))
            candidates.append(dict(x_controls=controls[:, 0].tolist(), z_controls=controls[:, 1].tolist()))
        lanes = [dict(json_index=l['json_index'], points=np.asarray(l['points'])) for l in sample['raw_lanes']]
        coverage = dictionary_coverage(candidates, lanes) if candidates else dict(covered=0, total=len(lanes), lanes=[])
        result = dict(packed_lanes=int((packed['gt_lanes_visibility'].sum(-1) >= 2).sum()),
                      full_lanes=int(target['gt_lanes_full_valid'].sum()), template_count=len(candidates),
                      median_mode_max_cubic_residual_m=residuals, perfect_template_coverage=coverage)
        sample['training_target_probe'] = result
        for key in ('packed_lanes', 'full_lanes'):
            totals[key] += result[key]
        totals['perfect_template_coverage'] += coverage['covered']
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('work_dirs/q0_bev_holdout/fidelity/baseline/config.py'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-targets', action='store_true')
    parser.add_argument('--validation-scenes', type=Path,
                        help='Audit the official validation section of this list, not training targets')
    parser.add_argument('--free-height-origin', action='store_true',
                        help='Match the geometric head with an absolute height intercept')
    parser.add_argument('--lateral-control-bound', type=float,
                        help='Diagnostic stability cap in meters; does not change model/coverage criteria')
    parser.add_argument('--height-control-bound', type=float,
                        help='Diagnostic stability cap for height controls in meters')
    parser.add_argument('--restarts', type=int, default=0,
                        help='Additional deterministic trimmed-fit starts for unsuccessful lane fits')
    args = parser.parse_args()
    if args.validation_scenes and args.training_targets:
        parser.error('The legacy training-target replay does not apply to validation scenes')
    if any(value is not None and value <= 0 for value in
           (args.lateral_control_bound, args.height_control_bound)):
        parser.error('Control stability bounds must be positive')
    fit_options = dict(free_height_origin=args.free_height_origin,
                       lateral_control_bound=args.lateral_control_bound,
                       height_control_bound=args.height_control_bound)
    cfg = Config.fromfile(args.config)
    dataset = OpenLaneDataset.__new__(OpenLaneDataset)
    split = 'validation' if args.validation_scenes else 'training'
    label_root = Path(cfg.train_dataloader.dataset.dataset_cfg.data_label_dir) / split
    if args.validation_scenes:
        groups = {}
        for sample in read_validation_scenarios(args.validation_scenes):
            groups.setdefault(sample['group'], []).append(sample)
    else:
        groups = {g: [dict(frame=f) for f in frames] for g, frames in cfg.q0_target_frames.items()}
    records = []
    random = np.random.default_rng(7)
    for group, samples in groups.items():
        for sample in samples:
            frame = sample['frame']
            files = ([label_root / sample['segment'] / f'{frame}.json']
                     if args.validation_scenes else list(label_root.glob(f'*/{frame}.json')))
            if len(files) != 1:
                raise ValueError(f'{frame}: expected exactly one GT JSON, got {len(files)}')
            label = json.loads(files[0].read_text())
            lanes = raw_lanes(label, dataset._build_calib(label))
            fitted = []
            for lane in lanes:
                best = fit_relative_cubic(lane['points'], **fit_options)
                if best['p80_xz_m'] >= .3:
                    for _ in range(args.restarts):
                        subset = random.choice(len(lane['points']), min(4, len(lane['points'])), replace=False)
                        candidate = fit_relative_cubic(lane['points'], iterations=4, initial_subset=subset,
                                                       **fit_options)
                        if candidate['p80_xz_m'] < best['p80_xz_m']:
                            best = candidate
                fitted.append(dict(json_index=lane['json_index'], **best))
            dictionary = dictionary_coverage(fitted, lanes)
            record = dict(frame=frame, group=group, raw_gt=len(lanes),
                          independent_fits=fitted, five_template_dictionary=dictionary,
                          raw_lanes=[dict(json_index=l['json_index'], points=l['points'].tolist()) for l in lanes])
            records.append(record)
            print(frame, 'feasible individual', sum(c['p80_xz_m'] < .3 for c in fitted),
                  'feasible dictionary', dictionary['covered'], '/', len(lanes), flush=True)
    result = dict(scope='GT-only constructive feasibility, not predictions or a certified upper bound; no trained weights',
                  evaluation_split=split, free_height_origin=args.free_height_origin,
                  diagnostic_control_bounds=dict(lateral=args.lateral_control_bound,
                                                 height=args.height_control_bound),
                  fit_restarts=args.restarts,
                  criterion='80% original visible points, same-y xz strictly < 0.3 m; distinct queries',
                  gt=sum(r['raw_gt'] for r in records),
                  independently_feasible=sum(c['p80_xz_m'] < .3 for r in records for c in r['independent_fits']),
                  five_template_feasible=sum(r['five_template_dictionary']['covered'] for r in records), samples=records)
    if args.training_targets:
        result['training_target_probe'] = audit_training_targets(cfg, records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print('TOTAL', {key: result[key] for key in ('gt', 'independently_feasible', 'five_template_feasible')}, flush=True)


if __name__ == '__main__':
    main()
