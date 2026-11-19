"""E4: initial-query (Q0) coverage of ground-truth lanes on OpenLane validation.

A ground-truth lane counts as covered when at least one of the initial
curve queries lies within ``--distance`` meters of it, in the lateral-height
(xz) plane, at no fewer than ``--ratio`` of its visible output anchors. The
defaults (0.3 m, 80%) are the Q0 criterion used by the training code. A
looser 0.5 m criterion is reported alongside.

Two query sets are scored: all initial queries produced by the curve query
generator (Q0, 505 curves) and the initial curves of the queries retained
for the second decoder layer (64 curves). A training-free straight-line
reference is scored as well: straight longitudinal lines at the same lateral
seeds with zero height, which is exactly the initial query geometry of the
E1 straight-template ablation. Coverage is also grouped by lane
curvature, lateral position and visible length. Ground truth is the
current-frame annotation at the 50 output anchors.

Usage (from the repository root, PYTHONPATH set to its parent):
    python ablation_extras/tools/e4_q0_coverage.py CONFIG CHECKPOINT --tag full

Outputs: ablation_extras/return_back/e4_q0coverage_<tag>/
"""

import argparse
import csv
import json
import math

import torch

from LaneLoom.ablation_extras.tools._runtime import (
    RETURN_ROOT, Progress, build_model_and_loader, gpu_name, now)

CURVATURE_BINS = ((0.0, 0.5, 'straight (<0.5 m)'),
                  (0.5, 2.0, 'moderate (0.5-2 m)'),
                  (2.0, math.inf, 'strong (>=2 m)'))
LATERAL_BINS = ((0.0, 2.5, 'ego boundaries (|x|<2.5 m)'),
                (2.5, 6.5, 'adjacent (2.5-6.5 m)'),
                (6.5, math.inf, 'outer (>=6.5 m)'))
LENGTH_BINS = ((0.0, 20.0, 'short (<20 m)'),
               (20.0, 50.0, 'medium (20-50 m)'),
               (50.0, math.inf, 'long (>=50 m)'))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--distance', type=float, default=0.3)
    parser.add_argument('--loose-distance', type=float, default=0.5)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--seed-min', type=float, default=-25.0)
    parser.add_argument('--seed-max', type=float, default=25.0)
    parser.add_argument('--num-seeds', type=int, default=101)
    parser.add_argument('--max-batches', type=int, default=0,
                        help='Stop after this many batches (0 = full split).')
    return parser.parse_args()


def lane_descriptors(lanes, visibility):
    """Curvature proxy, lateral position and visible length per lane."""
    descriptors = []
    for lane, visible in zip(lanes, visibility):
        y = lane[visible, 1]
        x = lane[visible, 0]
        design = torch.stack((y, torch.ones_like(y)), dim=-1)
        solution = torch.linalg.lstsq(design, x.unsqueeze(-1)).solution
        residual = (design @ solution).squeeze(-1) - x
        descriptors.append(dict(
            chord_deviation_m=float(residual.abs().max()),
            near_lateral_m=float(x[torch.argmin(y)].abs()),
            visible_length_m=float(y.max() - y.min())))
    return descriptors


def coverage_for_sample(q0_points, gt_lanes, gt_visibility, distance,
                        loose_distance, ratio):
    """Per-lane best Q0 match for one frame.

    q0_points: [Q, A, 3]; gt_lanes: [G, A, 3]; gt_visibility: [G, A].
    """
    visible = gt_visibility > 0.5
    keep = visible.sum(dim=-1) >= 2
    if not keep.any():
        return []
    lanes = gt_lanes[keep].float()
    visible = visible[keep]
    count = visible.sum(dim=-1).float()
    delta = q0_points[:, None, :, :].float() - lanes[None]
    point_distance = torch.linalg.vector_norm(delta[..., (0, 2)], dim=-1)
    results = []
    tight = ((point_distance < distance) & visible[None]).sum(-1) / count
    loose = ((point_distance < loose_distance) & visible[None]).sum(-1) / count
    masked = point_distance.masked_fill(~visible[None], math.inf)
    ordered = masked.sort(dim=-1).values
    index = (torch.ceil(count * ratio).long() - 1).clamp_min(0)
    percentile = ordered.gather(
        -1, index.view(1, -1, 1).expand(ordered.size(0), -1, 1)).squeeze(-1)
    best_percentile = percentile.min(dim=0).values
    descriptors = lane_descriptors(lanes.cpu(), visible.cpu())
    for lane_index, descriptor in enumerate(descriptors):
        results.append(dict(
            covered=bool((tight[:, lane_index] >= ratio).any()),
            covered_loose=bool((loose[:, lane_index] >= ratio).any()),
            best_percentile_m=float(best_percentile[lane_index]),
            **descriptor))
    return results


def straight_lines(anchor_y, seed_min, seed_max, num_seeds):
    """Straight longitudinal lines at the lateral seeds: [N, A, 3]."""
    seeds = torch.linspace(seed_min, seed_max, num_seeds,
                           device=anchor_y.device, dtype=anchor_y.dtype)
    lines = torch.zeros(num_seeds, anchor_y.numel(), 3,
                        device=anchor_y.device, dtype=anchor_y.dtype)
    lines[..., 0] = seeds[:, None]
    lines[..., 1] = anchor_y[None]
    return lines


def bin_name(value, bins):
    for low, high, name in bins:
        if low <= value < high:
            return name
    return bins[-1][2]


def aggregate(rows, key=None, bins=None):
    groups = {}
    for row in rows:
        name = 'all' if key is None else bin_name(row[key], bins)
        groups.setdefault(name, []).append(row)
    order = ['all'] if key is None else [b[2] for b in bins]
    summary = []
    for name in order:
        members = groups.get(name, [])
        if not members:
            continue
        distances = sorted(r['best_percentile_m'] for r in members)
        summary.append(dict(
            group=name,
            lanes=len(members),
            coverage_pct=round(
                100.0 * sum(r['covered'] for r in members) / len(members), 2),
            coverage_loose_pct=round(
                100.0 * sum(r['covered_loose'] for r in members)
                / len(members), 2),
            median_best_p80_m=round(distances[len(distances) // 2], 3),
            retained_coverage_pct=round(
                100.0 * sum(r.get('retained_covered', False)
                            for r in members) / len(members), 2),
            straight_coverage_pct=round(
                100.0 * sum(r.get('straight_covered', False)
                            for r in members) / len(members), 2),
            straight_coverage_loose_pct=round(
                100.0 * sum(r.get('straight_covered_loose', False)
                            for r in members) / len(members), 2)))
    return summary


def initial_curves(model, data):
    """All Q0 curves and the Q0 curves of the retained queries, one pass."""
    head = model.lane_head
    batch = model.data_preprocessor(data, False)
    img_feats, pts_feats, img_metas, _, _ = model._encode_batch(batch)
    prior = head._anchor_prior(img_feats, pts_feats, img_metas)
    if prior is None or 'anchor_points' not in prior:
        raise SystemExit('The model has no curve query generator output.')
    current_bev = head.forward_bev(img_feats, pts_feats, img_metas)
    occupancy = head._lidar_occupancy(pts_feats, current_bev)
    predictions = head._predict(
        current_bev, anchor_prior=prior,
        query_feature_sampler=head._query_feature_sampler(
            img_feats, current_bev, img_metas, occupancy))
    return batch, prior['anchor_points'], predictions['anchor_points']


def main():
    args = parse_args()
    print(f'[{now()}] E4 start: {args.tag} on {gpu_name()}', flush=True)
    model, loader = build_model_and_loader(
        args.config, args.checkpoint, f'e4_q0coverage_{args.tag}',
        batch_size=args.batch_size)
    total = len(loader) if not args.max_batches else min(
        len(loader), args.max_batches)
    progress = Progress(total, 'E4 batches', every=100)
    rows = []
    with torch.no_grad():
        for index, data in enumerate(loader):
            if index >= total:
                break
            batch, q0_all, q0_kept = initial_curves(model, data)
            for sample in range(q0_all.size(0)):
                device = q0_all.device
                lanes = batch['gt_lanes_3d'][sample].to(device)
                visible = batch['gt_lanes_visibility'][sample].to(device)
                all_rows = coverage_for_sample(
                    q0_all[sample], lanes, visible,
                    args.distance, args.loose_distance, args.ratio)
                kept_rows = coverage_for_sample(
                    q0_kept[sample], lanes, visible,
                    args.distance, args.loose_distance, args.ratio)
                straight = straight_lines(
                    q0_all[sample][0, :, 1].float(), args.seed_min,
                    args.seed_max, args.num_seeds)
                straight_rows = coverage_for_sample(
                    straight, lanes, visible,
                    args.distance, args.loose_distance, args.ratio)
                for row, line in zip(all_rows, straight_rows):
                    row['straight_covered'] = line['covered']
                    row['straight_covered_loose'] = line['covered_loose']
                    row['straight_best_percentile_m'] = line[
                        'best_percentile_m']
                for row, kept in zip(all_rows, kept_rows):
                    row['retained_covered'] = kept['covered']
                    row['retained_covered_loose'] = kept['covered_loose']
                    row['retained_best_percentile_m'] = kept[
                        'best_percentile_m']
                rows.extend(all_rows)
            progress.update(index + 1)

    out_dir = RETURN_ROOT / f'e4_q0coverage_{args.tag}'
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = dict(
        overall=aggregate(rows),
        by_curvature=aggregate(rows, 'chord_deviation_m', CURVATURE_BINS),
        by_lateral_position=aggregate(rows, 'near_lateral_m', LATERAL_BINS),
        by_visible_length=aggregate(rows, 'visible_length_m', LENGTH_BINS))
    payload = dict(tag=args.tag, config=args.config,
                   checkpoint=args.checkpoint, gpu=gpu_name(),
                   criterion=dict(distance_m=args.distance,
                                  loose_distance_m=args.loose_distance,
                                  visible_ratio=args.ratio),
                   batches=total, **tables)
    (out_dir / 'coverage.json').write_text(json.dumps(payload, indent=2) + '\n')
    with open(out_dir / 'lanes.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| Group | Lanes | Q0 coverage @0.3 m (%) | Q0 coverage @0.5 m (%) '
             '| Straight coverage @0.3 m (%) | Straight coverage @0.5 m (%) '
             '| Median best P80 (m) | Retained-64 coverage @0.3 m (%) |',
             '|---|---|---|---|---|---|---|---|']
    for section, entries in tables.items():
        lines.append(f'| **{section}** | | | | | | | |')
        for entry in entries:
            lines.append(
                f"| {entry['group']} | {entry['lanes']} | "
                f"{entry['coverage_pct']} | {entry['coverage_loose_pct']} | "
                f"{entry['straight_coverage_pct']} | "
                f"{entry['straight_coverage_loose_pct']} | "
                f"{entry['median_best_p80_m']} | "
                f"{entry['retained_coverage_pct']} |")
    table = '\n'.join(lines) + '\n'
    (out_dir / 'table.md').write_text(table)
    print(table)
    print(f'[{now()}] E4 SUCCESS. Pack and send this folder to Claude: '
          f'{out_dir}', flush=True)


if __name__ == '__main__':
    main()
