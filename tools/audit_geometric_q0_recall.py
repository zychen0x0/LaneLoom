"""Replay the documented validation frames through Q0, decoder and filtering.

Raw-point coverage uses distinct queries and the same criteria as the Q0
pretraining report. Official metrics always use current-frame raw annotations.
No annotations are passed to model inference. Results are diagnostic subsets,
not estimates of the full validation split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from LaneLoom.datasets.openlane.dataset import openlane_collate
from LaneLoom.tools.evaluate_q0_prior import cubic_basis, raw_lanes
from LaneLoom.tools.q0_scenarios import read_validation_scenarios
from LaneLoom.tools.report_epoch17_recall import official, metrics, nms
from LaneLoom.vis.output import visualize_query_stages as vis
from mmdet3d.registry import DATASETS


def raw_coverage(stage, lanes, y, ids, visible=False, threshold=.3, ratio=.8):
    """Evaluate cubic geometry at raw annotation y; interpolate visibility."""
    cx = stage['curve_x_controls'][0].numpy()[ids]
    cz = stage['curve_z_controls'][0].numpy()[ids]
    costs, fractions = [], []
    for lane in lanes:
        p = lane['points']
        basis = cubic_basis(p[:, 1])
        d = np.hypot(cx @ basis.T - p[:, 0], cz @ basis.T - p[:, 2])
        if visible:
            visibility = stage['visibility'][0].numpy()[ids]
            support = np.array([np.interp(p[:, 1], y, v, left=0, right=0)
                                for v in visibility]).reshape(len(ids), len(p))
            d = np.where(support >= .5, d, 1e4)
        k = int(np.ceil(ratio * len(p))) - 1
        costs.append(np.partition(d, k, axis=-1)[:, k])
        fractions.append((d < threshold).mean(-1))
    output = [dict(json_index=l['json_index'], covered=False, query_id=None)
              for l in lanes]
    if len(ids) and len(lanes):
        cost = np.array(costs)
        rr, cc = linear_sum_assignment((cost >= threshold)*1e6 + np.minimum(cost, 1e3))
        for r, c in zip(rr, cc):
            output[r].update(covered=bool(cost[r, c] < threshold),
                             query_id=int(stage['query_ids'][ids[c]]),
                             percentile_m=float(cost[r, c]), fraction=float(fractions[r][c]))
    return dict(gt=len(lanes), covered=sum(r['covered'] for r in output), lanes=output)


def analyze(record, head):
    data, batch = record['data'], record['batch']
    stages = vis.stages_from_capture(data, head)
    stages[0].update(curve_x_controls=data['expanded']['curve_x_controls'],
                     curve_z_controls=data['expanded']['curve_z_controls'])
    raw = batch['raw_label'][0]
    lanes = raw_lanes(raw, batch['img_metas'][0]['current']['openlane_calib'])
    y = record['prediction']['anchor_y_steps'].numpy()
    first, final = stages[1], stages[-1]
    selected = final['query_ids'].numpy()
    all_first = np.arange(first['x'].shape[1])
    all_final = np.arange(final['x'].shape[1])
    x, z, v, s = [final[k][0].numpy() for k in ('x', 'z', 'visibility', 'scores')]
    actual = np.flatnonzero(record['prediction']['valid_queries'][0].numpy()
                            & ((v >= .5).sum(-1) > 1))
    assert np.array_equal(actual, nms(x, z, v, s)), 'NMS replay differs from predict'
    pools = {
        'q0': (stages[0], np.arange(stages[0]['x'].shape[1]), False),
        'layer1': (first, all_first, False),
        'layer1_kept': (first, selected, False),
        'layer2_geometry': (final, all_final, False),
        'layer2_visibility': (final, all_final, True),
        'score': (final, nms(x, z, v, s, distance=None), True),
        'final': (final, actual, True),
    }
    result = dict(scene=record['scene'], raw_gt=len(lanes), stages={})
    for name, (stage, ids, use_vis) in pools.items():
        sx, sz = [stage[k][0].numpy() for k in ('x', 'z')]
        pv = stage['visibility'][0].numpy() >= .5 if use_vis else np.ones_like(sx, bool)
        result['stages'][name] = dict(
            queries=len(ids),
            strict=raw_coverage(stage, lanes, y, ids, use_vis),
            relaxed=raw_coverage(stage, lanes, y, ids, use_vis, .5, .75),
            official=official(sx, sz, pv, ids, y, raw))
    result['score_sweep'] = {str(t): official(x, z, v >= .5, nms(x,z,v,s,t), y,raw)
                             for t in (.1,.2,.3,.4,.5,.6,.7)}
    result['visibility_sweep'] = {str(t): official(x,z,v >= t,nms(x,z,v,s,visibility=t),y,raw)
                                  for t in (.1,.2,.3,.4,.5)}
    result['nms_sweep'] = {str(t): official(x,z,v >= .5,nms(x,z,v,s,distance=t),y,raw)
                          for t in (None,.1,.2,.35,.5)}
    return result


def summarize(rows):
    def total_official(items):
        counts = {k: sum(i[k] for i in items) for k in (
            'gt_count','pred_count','recall_matches','precision_matches','match_count',
            'oracle_one_to_one_matches')}
        return dict(counts, **metrics(counts))
    output = dict(frames=len(rows), raw_gt=sum(r['raw_gt'] for r in rows), stages={})
    for stage in rows[0]['stages']:
        items = [r['stages'][stage] for r in rows]
        output['stages'][stage] = dict(
            strict=sum(i['strict']['covered'] for i in items),
            relaxed=sum(i['relaxed']['covered'] for i in items),
            official=total_official([i['official'] for i in items]))
    for sweep in ('score_sweep','visibility_sweep','nms_sweep'):
        output[sweep] = {t: total_official([r[sweep][t] for r in rows]) for t in rows[0][sweep]}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--budget', type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    scenes = read_validation_scenarios(ROOT / 'docs/场景.md')
    if args.limit:
        scenes = scenes[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = Config.fromstring(saved['meta']['cfg'], '.py')
    del saved
    dataset_cfg = vis.infer._inference_dataset_cfg(cfg)
    dataset_cfg['test_mode'] = True
    dataset_cfg['dataset_cfg']['load_interval'] = 1
    dataset = DATASETS.build(dataset_cfg)
    lookup = {str(Path(p).resolve()): i for i,p in enumerate(dataset.label_files)}
    device = torch.device('cuda:0')
    model = vis.infer._build_model(cfg, args.checkpoint, device)
    if args.budget:
        model.lane_head.lane_decoder.layer_query_budget = args.budget
    rows = []
    for scene in scenes:
        cache = args.output / (scene['frame'] + '.pt')
        if cache.exists():
            record = torch.load(cache, weights_only=False)
            assert record['checkpoint'] == str(args.checkpoint.resolve())
            assert record['budget'] == args.budget
        else:
            label = Path(cfg.dataset_cfg.data_label_dir) / 'validation' / scene['segment'] / (scene['frame']+'.json')
            batch = openlane_collate([dataset[lookup[str(label.resolve())]]])
            data, handles = {}, []
            for name, module in [('prior', model.lane_head.anchor_prior),
                                 ('expanded', model.lane_head.proposal_expander),
                                 ('decoder', model.lane_head.lane_decoder)]:
                def hook(module, inputs, output, name=name):
                    data[name] = vis.cpu({k:v for k,v in output.items()
                                          if k not in ('query_features','bev')})
                handles.append(module.register_forward_hook(hook))
            try:
                prediction = vis.infer._run_inference(model, batch, device)
            finally:
                for handle in handles:
                    handle.remove()
            record = dict(scene=scene, checkpoint=str(args.checkpoint.resolve()),
                          budget=args.budget, data=data, prediction=prediction,
                          batch=vis.cpu({k:v for k,v in batch.items() if k not in ('img','lidar')}))
            torch.save(record, cache)
        row = analyze(record, model.lane_head)
        rows.append(row)
        f = row['stages']['final']['official']
        print(scene['group'], scene['frame'], 'raw',row['raw_gt'],
              'strict', {k:v['strict']['covered'] for k,v in row['stages'].items()},
              'official R/P', round(f['recall'],3),round(f['precision'],3),flush=True)
    report = dict(checkpoint=str(args.checkpoint.resolve()), budget=args.budget,
                  criteria=dict(strict='raw points xz <0.3m at 80%; distinct queries',
                                relaxed='raw points xz <0.5m at 75%; distinct queries',
                                visibility='linear interpolation of probabilities, >=0.5; outside anchors invisible',
                                official='repository OpenLane evaluator; current-frame raw_label'),
                  summary=summarize(rows),
                  groups={g:summarize([r for r in rows if r['scene']['group']==g])
                          for g in sorted(set(s['group'] for s in scenes))}, samples=rows)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print('SUMMARY',json.dumps(report['summary']),flush=True)


if __name__ == '__main__':
    main()
