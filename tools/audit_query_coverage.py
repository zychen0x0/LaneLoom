"""Audit every Q0 seed/template and locate per-GT coverage loss.

Only a temporary replay cache is written; sample output directories stay SVG-only.
Run --assert-q0-coverage to fail when any annotated lane lacks a Q0 candidate
within 0.3 m at 80% of visible anchors. This is a diagnostic, not official recall.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from LaneLoom.vis.output import visualize_query_stages as vis
from LaneLoom.datasets.openlane.full_shape_targets import build_full_shape_targets

CACHE = Path('/tmp/laneloom-epoch43-query-coverage.pt')


def replay(images, checkpoint, cache=CACHE):
    if cache.exists():
        saved = torch.load(cache, map_location='cpu', weights_only=False)
        if (saved['checkpoint'] == str(checkpoint.resolve())
                and saved['images'] == [str(p.resolve()) for p in images]):
            return saved['records']
    cfg = Config.fromstring(torch.load(checkpoint, map_location='cpu',
                                      weights_only=False)['meta']['cfg'], '.py')
    model = vis.infer._build_model(cfg, checkpoint, torch.device('cuda:0'))
    records = []
    for image in images:
        batch = vis.infer._build_sample(cfg, image)
        with vis.StageCapture(model.lane_head) as capture:
            prediction = vis.infer._run_inference(model, batch, torch.device('cuda:0'))
        stages = vis.stages_from_capture(capture.data, model.lane_head)
        record = dict(image=str(image), data=capture.data, prediction=prediction,
                      stages=stages, batch=vis.cpu({k: v for k, v in batch.items() if k != 'img'}))
        records.append(record)
    torch.save(dict(checkpoint=str(checkpoint.resolve()), images=[str(p.resolve()) for p in images],
                    records=records), cache)
    return records


def distances(stage, gt, visibility):
    xz = torch.stack((stage['x'][0], stage['z'][0]), dim=-1).double()
    target = gt[..., [0, 2]].double()
    distances = torch.linalg.vector_norm(xz[:, None] - target[None], dim=-1)
    ratios = ((distances < .3) & visibility[None]).sum(-1) / visibility.sum(-1).clamp_min(1)
    sorted_distance = distances.masked_fill(~visibility[None], torch.inf).sort(-1).values
    index = (torch.ceil(.8 * visibility.sum(-1)).long() - 1).clamp_min(0)
    p80 = sorted_distance.gather(-1, index[None, :, None].expand(distances.shape[0], -1, 1)).squeeze(-1)
    return distances, ratios, p80


def audit(record, verbose=True):
    batch, data, stages = record['batch'], record['data'], record['stages']
    valid = (batch['gt_lanes_visibility'][0] > .5).sum(-1) >= 2
    gt = batch['gt_lanes_3d'][0, valid]
    visibility = batch['gt_lanes_visibility'][0, valid] > .5
    slots = torch.where(valid)[0]
    seed_ids = data['expanded']['selected_seed_indices'][0]
    template_ids = data['expanded']['selected_template_indices'][0]
    assert len(template_ids) == 505
    assert torch.bincount(template_ids).tolist() == [101] * 5
    assert torch.equal(seed_ids, torch.arange(101).repeat(5))
    # Verify the complete Cartesian product, with seed translations shared across all controls.
    cx = data['expanded']['curve_x_controls'][0].reshape(5, 101, 4)
    cz = data['expanded']['curve_z_controls'][0].reshape(5, 101, 4)
    seeds = torch.linspace(-25, 25, 101)
    torch.testing.assert_close(cx, data['prior']['shape_x_controls'][0, :, None] + seeds[None, :, None])
    torch.testing.assert_close(cz, data['prior']['shape_z_controls'][0, :, None].expand_as(cz))
    selected = stages[-1]['query_ids']
    kept_stage1 = {k: v[:, selected] if torch.is_tensor(v) and k not in ('query_ids',) else v
                   for k, v in stages[1].items()}
    kept_stage1['query_ids'] = selected
    budget = len(selected)
    pools = [('Q0-all505', stages[0]), ('L1-all505', stages[1]),
             (f'L1-kept{budget}', kept_stage1), (f'L2-all{budget}', stages[2])]
    results = {name: distances(stage, gt, visibility) for name, stage in pools}
    if verbose:
        print('\nFRAME', Path(record['image']).stem, 'GT', len(gt), flush=True)
        print('Q0 template counts', torch.bincount(template_ids, minlength=5).tolist(),
              f'kept{budget}', torch.bincount(template_ids[selected], minlength=5).tolist(),
              'detected', torch.bincount(template_ids[selected][record['prediction']['valid_queries'][0]], minlength=5).tolist())
        print('Stage geometry coverage (0.3m / 80% GT-visible anchors):',
              {name: int((result[1].max(0).values >= .8).sum()) for name, result in results.items()})
    table = []
    for j, slot in enumerate(slots):
        rows = []
        for name, stage in pools:
            dist, ratio, p80 = results[name]
            best = p80[:, j].argmin()
            qid = int(stage['query_ids'][best])
            rows.append(dict(stage=name, qid=qid, template=int(template_ids[qid]),
                             p80=float(p80[best, j]), best_ratio=float(ratio[:, j].max()),
                             query_ratio=float(ratio[best, j]),
                             score=float(stage['scores'][0, best]) if 'scores' in stage else None))
        q0_dist, _, _ = results['Q0-all505']
        best_q0 = rows[0]['qid']
        support = visibility[j]
        p = gt[j, support]
        final_visibility = stages[-1]['visibility'][0] >= .5
        final_dist = results[f'L2-all{budget}'][0][:, j]
        final_ratio = ((final_dist < .3) & final_visibility & support).sum(-1) / support.sum()
        score_pass = stages[-1]['scores'][0] >= .5
        actual_keep = record['prediction']['valid_queries'][0]
        selected_q0 = int(selected.tolist().index(best_q0)) if best_q0 in selected.tolist() else None
        row = dict(gt_slot=int(slot), y=(float(p[0, 1]), float(p[-1, 1])),
                   x=(float(p[0, 0]), float(p[-1, 0])), stages=rows,
                   q0_best_survives=selected_q0 is not None,
                   final_geom_vis=float(final_ratio.max()),
                   score_mask_best=float(final_ratio.masked_fill(~score_pass, 0).max()),
                   nms_mask_best=float(final_ratio.masked_fill(~actual_keep, 0).max()))
        table.append(row)
        if verbose:
            print('GT', int(slot), 'x/y', row['x'], row['y'],
                  ' | '.join(f"{r['stage']} q{r['qid']}/t{r['template']} p80={r['p80']:.3f} pool_cov={r['best_ratio']:.2f} score={r['score']}" for r in rows),
                  'vis/score/nms coverage', [round(row[k], 2) for k in ('final_geom_vis', 'score_mask_best', 'nms_mask_best')])
    return table, all(row['stages'][0]['best_ratio'] >= .8 for row in table)


def probe_targets(record):
    b = record['batch']
    gt = b['gt_lanes_3d'][0].numpy()
    visibility = b['gt_lanes_visibility'][0].numpy() > .5
    y = record['prediction']['anchor_y_steps'].numpy()
    target = build_full_shape_targets(
        gt, visibility, y, max_templates=5, min_lane_points=4, min_lane_span=15.,
        guarded_short_merge=True, z_extrapolation_grace_m=20., z_extrapolation_decay_m=20.)
    shapes = record['data']['prior']['shape_points'][0].numpy()
    target_shapes = target['gt_shape_templates'][target['gt_shape_template_valid']]
    print('TARGET PROBE', Path(record['image']).stem, 'valid full templates', len(target_shapes))
    result = []
    for j in np.flatnonzero(visibility.sum(-1) >= 2):
        m = visibility[j]
        points = gt[j, m]
        ratios = []
        for templates, shifts in ((shapes, np.linspace(-25, 25, 1001)),
                                  (target_shapes, np.linspace(-25, 25, 101))):
            dx = templates[:, None, m, 0] + shifts[None, :, None] - points[None, None, :, 0]
            dz = templates[:, None, m, 2] - points[None, None, :, 2]
            ratios.append(float((np.hypot(dx, dz) < .3).mean(-1).max()))
        full_error = np.hypot(target['gt_lanes_3d_full'][j, m, 0] - points[:, 0],
                              target['gt_lanes_3d_full'][j, m, 2] - points[:, 2])
        own_p80 = (float(np.sort(full_error)[int(np.ceil(.8 * len(points))) - 1])
                   if target['gt_lanes_full_valid'][j] else None)
        row = dict(gt_slot=int(j), dense_seed_best_ratio=ratios[0],
                   perfect_target_template_best_ratio=ratios[1], own_target_p80=own_p80)
        result.append(row)
        print(row)
    return target, result


def probe_budgets(records, checkpoint):
    cfg = Config.fromstring(torch.load(checkpoint, map_location='cpu', weights_only=False)['meta']['cfg'], '.py')
    device = torch.device('cuda:0')
    model = vis.infer._build_model(cfg, checkpoint, device)
    decoder = model.lane_head.lane_decoder
    for record in records:
        batch = vis.infer._build_sample(cfg, Path(record['image']))
        inputs = dict(img=batch['img'].to(device), lidar=[v.to(device) for v in batch['lidar']],
                      img_metas=batch['img_metas'])
        with torch.inference_mode():
            img_feats, pts_feats, metas, _, _ = model._encode_batch(inputs)
            for budget in (128, 505):
                decoder.layer_query_budget = budget
                with vis.StageCapture(model.lane_head) as capture:
                    prediction = vis.cpu(model.lane_head.predict(img_feats, pts_feats, metas))
                replayed = dict(record, data=capture.data, prediction=prediction,
                                stages=vis.stages_from_capture(capture.data, model.lane_head))
                print('BUDGET INTERVENTION', budget, Path(record['image']).stem)
                audit(replayed)
        decoder.layer_query_budget = 64


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, nargs='+', default=vis.DEFAULT_IMAGES[1:])
    parser.add_argument('--checkpoint', type=Path, default=vis.DEFAULT_CHECKPOINT)
    parser.add_argument('--assert-q0-coverage', action='store_true')
    parser.add_argument('--probe-targets', action='store_true')
    parser.add_argument('--probe-budgets', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    records = replay(args.images, args.checkpoint)
    covered = [audit(r)[1] for r in records]
    if args.probe_targets:
        for record in records:
            probe_targets(record)
    if args.probe_budgets:
        probe_budgets(records, args.checkpoint)
    if args.assert_q0_coverage and not all(covered):
        raise SystemExit('FAIL: not every GT has a covering Q0 candidate in the full 505-query pool.')


if __name__ == '__main__':
    main()
