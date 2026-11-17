"""Quantify cached checkpoint stages using both strict and official criteria.

Exports only SVG figures into gt_shape_extension_prototype; numerical evidence
is saved separately in epoch17_recall_diagnostic. Oracle coverage is explicitly
separate from the official evaluator's minimum-cost, one-to-one matching.
"""
from __future__ import annotations

import json
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from LaneLoom.evaluation.lane_evaluation import OpenLaneLaneEvaluator
from LaneLoom.evaluation.lane_evaluation.min_cost_flow import solve_min_cost_flow
from LaneLoom.models.head.curve_nms import mean_distance_curve_nms
from LaneLoom.models.valid_range_tube_iou import valid_range_tube_iou

OUT = ROOT / 'work_dirs/epoch17_recall_diagnostic'
FIG = ROOT / 'work_dirs/gt_shape_extension_prototype/scenes/checkpoint_epoch17'
EV = OpenLaneLaneEvaluator()
plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans',
                     'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def arr(v):
    return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)


def metrics(s):
    r = s['recall_matches'] / max(s['gt_count'], 1)
    p = s['precision_matches'] / max(s['pred_count'], 1)
    return dict(recall=r, precision=p, f1=2*r*p/max(r+p, 1e-12))


def strict(x, z, gt, vis, ids):
    rows = []
    for g in np.flatnonzero(vis.sum(-1) >= 2):
        m = vis[g]
        d = np.hypot(x[ids][:, m] - gt[g, m, 0], z[ids][:, m] - gt[g, m, 2])
        if not len(ids):
            rows.append(dict(gt=int(g), covered=False, query=None))
            continue
        ratio = (d < .3).mean(-1)
        j = np.lexsort((d.mean(-1), -ratio))[0]
        rows.append(dict(gt=int(g), query=int(ids[j]), covered=bool(ratio[j] >= .8),
                         ratio=float(ratio[j]), mean_xz_m=float(d[j].mean()),
                         mean_abs_x_m=float(np.abs(x[ids[j], m]-gt[g, m, 0]).mean()),
                         mean_abs_z_m=float(np.abs(z[ids[j], m]-gt[g, m, 2]).mean())))
    return dict(covered=sum(r['covered'] for r in rows), gt_count=len(rows), per_gt=rows)


def official(x, z, pv, ids, y, raw):
    lanes = [np.column_stack((x[q, pv[q]], y[pv[q]], z[q, pv[q]])) for q in ids]
    s = EV.evaluate(lanes, [1]*len(lanes), raw)
    # Synthetic stages have no meaningful category predictions.
    s.pop('category_matches')
    s.update(metrics(s))
    gl, gv, _ = EV.ground_truth_from_raw_label(raw)
    gl = [g[np.asarray(v) > 0] for g, v in zip(gl, gv)]
    gl, gids = EV._prepare_lanes(gl, list(range(len(gl))))
    gl, gv, gids = EV._resample_lanes(gl, gids)
    pl, pids = EV._prepare_lanes(lanes, list(map(int, ids)))
    pl, pvis, pids = EV._resample_lanes(pl, pids)
    rows = []
    if len(gl) and len(pl):
        pairs, adj, cost = EV._pair_matrices(gl, gv, pl, pvis)
        ratios = np.array([[pairs[g,p]['num_matches']/gv[g].sum()
                            for p in range(len(pl))] for g in range(len(gl))])
        valid = (ratios >= .75) & (cost < EV.dist_th*len(EV.y_samples))
        rr, cc = linear_sum_assignment(-valid.astype(float))
        s['oracle_one_to_one_matches'] = int(valid[rr,cc].sum())
        assignment = {g:p for g,p,c in solve_min_cost_flow(adj,cost) if c < 150}
        for g, gid in enumerate(gids):
            p = int(np.lexsort((cost[g], -ratios[g]))[0])
            matched = assignment.get(g)
            dx = np.abs(pl[p][:,0]-gl[g][:,0]); dz=np.abs(pl[p][:,1]-gl[g][:,1])
            overlap = (gv[g] > .5) & (pvis[p] > .5)
            rows.append(dict(gt=gid, best_query=pids[p], best_recall_ratio=float(ratios[g,p]),
                             oracle_covered=bool(valid[g].any()),
                             gt_y=[float(EV.y_samples[gv[g]>.5][0]),float(EV.y_samples[gv[g]>.5][-1])],
                             pred_y=[float(EV.y_samples[pvis[p]>.5][0]),float(EV.y_samples[pvis[p]>.5][-1])],
                             mean_abs_x_overlap=float(dx[overlap].mean()) if overlap.any() else None,
                             mean_abs_z_overlap=float(dz[overlap].mean()) if overlap.any() else None,
                             assigned_query=None if matched is None else pids[matched],
                             assigned_recall_ratio=0. if matched is None else float(ratios[g,matched]),
                             official_recalled=bool(matched is not None and ratios[g,matched]>=.75)))
    else:
        s['oracle_one_to_one_matches'] = 0
        rows = [dict(gt=g, best_query=None, oracle_covered=False, official_recalled=False) for g in gids]
    s['per_gt'] = rows
    return s


def nms(x, z, v, scores, threshold=.5, distance=.35, visibility=.5):
    candidate = (scores>=threshold) & ((v>=visibility).sum(-1)>1)
    if distance is None:
        return np.flatnonzero(candidate)
    keep=mean_distance_curve_nms(*[torch.as_tensor(a).unsqueeze(0) for a in (x,z,v,scores)],
                                distance_threshold=distance, visibility_threshold=visibility,
                                candidate_mask=torch.as_tensor(candidate).unsqueeze(0))[0].numpy()
    return np.flatnonzero(candidate & keep)


def analyze(c):
    batch=c['batch']; y=arr(c['output']['anchor_y_steps']); gt=arr(batch['gt_lanes_3d'][0])
    vis=arr(batch['gt_lanes_visibility'][0])>.5; raw=batch['raw_label'][0]
    d=c['stages']['decoder']; l=d['aux_predictions'][0]
    q=arr(c['stages']['q0']['anchor_points'][0]); selected=arr(d['selected_query_indices'][0])
    x,z,v,s=[arr(c['output'][k][0]) for k in ('x','z','visibility','scores')]
    lx,lz,lv=[arr(l[k][0]) for k in ('pred_x','pred_z','pred_vis')];lv=1/(1+np.exp(-lv))
    kept=nms(x,z,v,s)
    actual=np.flatnonzero(arr(c['output']['valid_queries'][0]) & ((v>=.5).sum(-1)>1))
    assert np.array_equal(kept,actual), 'Diagnostic NMS must agree with real predict output'
    stages={
        'q0_505':(q[...,0],q[...,2],np.ones_like(q[...,0],bool),np.arange(len(q))),
        'layer0_505_full':(lx,lz,np.ones_like(lx,bool),np.arange(len(lx))),
        'layer0_selected64_full':(lx,lz,np.ones_like(lx,bool),selected),
        'decoder64_full':(x,z,np.ones_like(x,bool),np.arange(len(x))),
        'decoder64_visibility':(x,z,v>=.5,np.arange(len(x))),
        'score_pass_full':(x,z,np.ones_like(x,bool),np.flatnonzero(s>=.5)),
        'score_pass_visibility':(x,z,v>=.5,nms(x,z,v,s,distance=None)),
        'final':(x,z,v>=.5,kept),
    }
    result=dict(frame=c['scene']['frame'],scene_type=c['scene']['scene_type'],
                backend=c['backend'], raw_gt_count=len(raw['lane_lines']),
                packed_gt_count=int((vis.sum(-1)>=2).sum()),
                full_target_lane_count=int(c['targets']['gt_lanes_full_valid'].sum()),
                template_count=int(c['targets']['gt_shape_template_valid'].sum()), stages={})
    for name,(xx,zz,vv,ids) in stages.items():
        result['stages'][name]=dict(queries=len(ids), strict=strict(xx,zz,gt,vis,ids),
                                   official=official(xx,zz,vv,ids,y,raw))
    result['threshold_sweep']={str(t):official(x,z,v>=.5,nms(x,z,v,s,t),y,raw)
                               for t in (.05,.1,.2,.3,.4,.5,.6,.7)}
    result['visibility_sweep']={str(t):official(x,z,v>=t,nms(x,z,v,s,visibility=t),y,raw)
                                for t in (.1,.2,.3,.4,.5)}
    result['nms_sweep']={str(t):official(x,z,v>=.5,nms(x,z,v,s,distance=t),y,raw)
                         for t in (None,.1,.2,.35,.5)}
    templates=arr(c['stages']['prior']['shape_points'][0]); targets=c['targets']
    target_ids=np.flatnonzero(targets['gt_shape_template_valid']); truth=targets['gt_shape_templates'][target_ids]
    delta=templates[:,None]-truth[None]; dist=np.linalg.norm(delta[...,[0,2]],axis=-1)
    rr,cc=linear_sum_assignment(dist.mean(-1)); result['template_matches']=[]
    for r,col in zip(rr,cc):
        result['template_matches'].append(dict(pred=int(r),gt=int(target_ids[col]),
            mean_xz_m=float(dist[r,col].mean()),mae_x_m=float(np.abs(delta[r,col,:,0]).mean()),
            mae_z_m=float(np.abs(delta[r,col,:,2]).mean()),
            far_x_m=float(np.abs(delta[r,col,y>40,0]).mean()),
            far_z_m=float(np.abs(delta[r,col,y>40,2]).mean())))
    result['decoder_per_gt']=[]
    for row in result['stages']['decoder64_full']['strict']['per_gt']:
        g=row['gt']; m=vis[g]; distances=np.hypot(x[:,m]-gt[g,m,0],z[:,m]-gt[g,m,2])
        good=(distances<.3).mean(-1)>=.8
        good_ids=np.flatnonzero(good); best=int(row['query'])
        iou=valid_range_tube_iou(torch.from_numpy(x),torch.from_numpy(z),torch.from_numpy(v),
            torch.from_numpy(gt[g,:,0]),torch.from_numpy(gt[g,:,2]),torch.from_numpy(vis[g]),tube_radius=.5)
        result['decoder_per_gt'].append(dict(**row, best_geometry_score=float(s[best]),
            best_geometry_tube_iou=float(iou[best]), matching_query_count=int(good.sum()),
            max_matching_score=float(s[good].max()) if good.any() else None,
            max_matching_score_query=int(good_ids[np.argmax(s[good])]) if good.any() else None))
    result['kept_queries']=[dict(query=int(k),source_query=int(selected[k]),score=float(s[k]),
                                visible_y=y[v[k]>=.5].tolist()) for k in kept]
    packed_ids=arr(batch['gt_lane_track_ids'][0])
    result['supervision_audit']=[]
    raw_lanes,raw_visibility,_=EV.ground_truth_from_raw_label(raw)
    for final in result['stages']['final']['official']['per_gt']:
        g=final['gt'];track=raw['lane_lines'][g].get('track_id',-1)
        candidates=np.flatnonzero((packed_ids==track) & (vis.sum(-1)>=2)) if track!=-1 else []
        packed=int(candidates[0]) if len(candidates)==1 else None
        lane=raw_lanes[g][np.asarray(raw_visibility[g])>0]
        anchors=(y>=lane[:,1].min()) & (y<=lane[:,1].max())
        result['supervision_audit'].append(dict(raw_gt=g,track_id=track,packed_gt=packed,
            raw_y=[float(lane[:,1].min()),float(lane[:,1].max())],
            anchors_inside_raw_support=int(anchors.sum()),
            full_target=bool(packed is not None and targets['gt_lanes_full_valid'][packed]),
            final_recalled=final['official_recalled']))
    packed=np.flatnonzero(vis.sum(-1)>=2)
    result['perfect_packed_gt_replay']=official(gt[...,0],gt[...,2],vis,packed,y,raw)
    result['failure_attribution']=[]
    trace_names=['decoder64_full','decoder64_visibility','score_pass_visibility','final']
    trace={name:{row['gt']:row for row in result['stages'][name]['official']['per_gt']}
           for name in trace_names}
    for final in result['stages']['final']['official']['per_gt']:
        g=final['gt']
        if final['official_recalled']:reason='recalled'
        elif not trace['decoder64_full'][g]['oracle_covered']:reason='geometry_or_range'
        elif not trace['decoder64_visibility'][g]['oracle_covered']:reason='visibility'
        elif not trace['score_pass_visibility'][g]['oracle_covered']:reason='score'
        elif not trace['final'][g]['oracle_covered']:reason='nms'
        else:reason='one_to_one_matching'
        result['failure_attribution'].append(dict(raw_gt=g,reason=reason))
    # Confirm the replay agrees with the actual metric adapter, including head NMS.
    from LaneLoom.evaluation.openlane_official_metric import OpenLaneOfficialMetric
    metric=OpenLaneOfficialMetric(score_threshold=.5,visibility_threshold=.5)
    check=metric._evaluate_sample(dict(pred_x=x,pred_z=z,pred_visibility=v,pred_scores=s,
        pred_category=arr(c['output']['category'][0]),anchor_y_steps=y,raw_label=raw,
        pred_nms_keep_queries=arr(c['output']['nms_keep_queries'][0])))
    for key in ('recall_matches','precision_matches','gt_count','pred_count','match_count'):
        assert check[key]==result['stages']['final']['official'][key], key
    return result, stages


def draw_curve(ax, xyz, color, **kwargs):
    ax[0].plot(xyz[:,0],xyz[:,1],color=color,**kwargs)
    ax[1].plot(xyz[:,1],xyz[:,2],color=color,**kwargs)


def setup_pair(ax):
    ax[0].set(xlabel='Lateral x (m)',ylabel='Forward y (m)',xlim=(-25,25),ylim=(0,103))
    ax[1].set(xlabel='Forward y (m)',ylabel='Height z (m)',xlim=(0,103))
    for a in ax:a.grid(alpha=.18)


def plot(c,r,stages):
    frame=r['frame']; y=arr(c['output']['anchor_y_steps']); batch=c['batch']
    gt=arr(batch['gt_lanes_3d'][0]); vis=arr(batch['gt_lanes_visibility'][0])>.5
    gtids=np.flatnonzero(vis.sum(-1)>=2); colors=plt.get_cmap('tab10')
    fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
    for col,(name,title) in enumerate([('q0_505','Initial 505 queries'),('decoder64_full','Decoded 64 queries'),('final','Final predictions')]):
        ax=axes[:,col]; x,z,v,ids=stages[name];st=r['stages'][name]
        if col==0:
            for k in ids:
                draw_curve(ax,np.column_stack((x[k],y,z[k])),color='#b6c0c8',alpha=.18,lw=.35)
        for g in gtids:
            draw_curve(ax,gt[g,vis[g]],color=colors(g%10),lw=2.0,label=f'GT {g}')
        if col<2:
            for row in st['strict']['per_gt']:
                k=row['query']
                if k is not None:
                    draw_curve(ax,np.column_stack((x[k],y,z[k])),color=colors(row['gt']%10),ls='--',lw=1.2)
        else:
            for k in ids:
                draw_curve(ax,np.column_stack((x[k,v[k]],y[v[k]],z[k,v[k]])),color='#20252c',ls='--',lw=1.6)
        setup_pair(ax)
        ax[0].set_title(f'{title}\nstrict geometry: {st["strict"]["covered"]}/{st["strict"]["gt_count"]}; kept {len(ids)}')
        if col==0:ax[0].legend(loc='upper left',fontsize=7,ncol=2)
    fig.suptitle(f'{r["scene_type"]} / {frame} | best checkpoint epoch 17\n'
                 'Colored solid: training GT. Colored dashed: best query per GT. Black dashed: actual output.\n'
                 'Strict geometry: >=80% of GT-visible anchors within 0.3 m (ignores predicted visibility).',fontsize=11)
    z_limits=[a.get_ylim() for a in axes[1]]
    for a in axes[1]:a.set_ylim(min(v[0] for v in z_limits),max(v[1] for v in z_limits))
    fig.savefig(FIG/f'{r["scene_type"]}_{frame}_stages.svg')
    if frame=='150696587520602900':fig.savefig('/tmp/laneloom_stage_preview.png',dpi=120)
    plt.close(fig)
    if frame!='150696587520602900':return
    fig,ax=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    templates=arr(c['stages']['prior']['shape_points'][0]);truth=c['targets']['gt_shape_templates']
    for k,t in enumerate(templates):draw_curve(ax,t,colors(k),lw=1.7,label=f'Pred T{k}')
    for m in r['template_matches']:
        draw_curve(ax,truth[m['gt']],colors(m['pred']),ls='--',lw=2,
                   label=f'GT mode {m["gt"]} / error {m["mean_xz_m"]:.2f} m')
    setup_pair(ax)
    allx=np.concatenate([templates[...,0].ravel(),truth[c['targets']['gt_shape_template_valid'],:,0].ravel()])
    ax[0].set_xlim(float(allx.min())-1,float(allx.max())+1);ax[0].legend(fontsize=8)
    fig.suptitle(f'{frame} | LiDAR relative templates\nSolid: predicted; dashed: fitted GT target. Error is mean x-z distance over 5-100 m.')
    fig.savefig(FIG/f'branch_{frame}_templates.svg');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    raw_lanes,raw_vis,_=EV.ground_truth_from_raw_label(batch['raw_label'][0])
    eligible={row['gt'] for row in r['stages']['final']['official']['per_gt']}
    for g in sorted(eligible):
        lane=raw_lanes[g][np.asarray(raw_vis[g])>0]
        draw_curve(ax,lane,colors(g%10),lw=2,label=f'Raw GT {g}')
    x,z,v,ids=stages['final']
    for k in ids:
        draw_curve(ax,np.column_stack((x[k,v[k]],y[v[k]],z[k,v[k]])),color='#20252c',ls='--',lw=1.5,
                   label=f'Q{k}, score {float(c["output"]["scores"][0,k]):.3f}')
    setup_pair(ax);ax[0].set_xlim(-15,15);ax[0].legend(loc='upper left',fontsize=8)
    o=r['stages']['final']['official']
    fig.suptitle(f'{frame} | actual output vs official raw GT\n'
                 f'Recall {o["recall_matches"]}/{o["gt_count"]}; Precision {o["precision_matches"]}/{o["pred_count"]}; F1 {100*o["f1"]:.2f}%\n'
                 'Solid: official-eligible raw GT (including short lanes). Dashed: predicted visible intervals.')
    fig.savefig(FIG/f'branch_{frame}_official.svg');fig.savefig('/tmp/laneloom_official_preview.png',dpi=120);plt.close(fig)


def log_data():
    path=ROOT/'work_dirs/laneloom_openlane_image_lidar_temporal_0923_z_confidence_full_50e/exp_date_20260924_174958/logs/20260924_174958.log'
    text=path.read_text()
    pattern=(r'Evaluation laneline F1: ([\d.]+).*?Recall: ([\d.]+).*?Precision: ([\d.]+)'
             r'.*?GT / Pred / TP: (\d+) / (\d+) / (\d+).*?before NMS / suppressed / ratio: (\d+) / (\d+) / ([\d.]+)')
    rows=[]
    for i,m in enumerate(re.finditer(pattern,text,re.S)):
        rows.append(dict(epoch=i+1,**{k:float(v) for k,v in zip(
            ('f1','recall','precision','gt','pred','match_count','before_nms','suppressed','nms_ratio'),m.groups())}))
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    for key,color in [('f1','#1c6190'),('recall','#bd593c'),('precision','#38875b')]:
        ax.plot([r['epoch'] for r in rows],[100*r[key] for r in rows],label=key,color=color)
    ax.axhline(65,color='gray',ls='--',lw=.8,label='65% reference');ax.axvline(17,color='gray',ls=':',lw=.8)
    ax.set(xlabel='Epoch',ylabel='Official metric (%)',title='Validation history | current-frame raw GT');ax.legend();ax.grid(alpha=.2)
    fig.savefig(FIG/'validation_history.svg');plt.close(fig)
    return rows


def aggregate(results):
    def pool(items):
        sums={k:sum(row[k] for row in items) for k in
              ('recall_matches','precision_matches','gt_count','pred_count','match_count','oracle_one_to_one_matches')}
        sums.update(metrics(sums))
        return sums
    summary=dict(frames=len(results),stages={})
    for name in results[0]['stages']:
        summary['stages'][name]=dict(
            strict_covered=sum(r['stages'][name]['strict']['covered'] for r in results),
            strict_gt=sum(r['stages'][name]['strict']['gt_count'] for r in results),
            **pool([r['stages'][name]['official'] for r in results]))
    for name in ('threshold_sweep','visibility_sweep','nms_sweep'):
        summary[name]={t:pool([r[name][t] for r in results]) for t in results[0][name]}
    summary['perfect_packed_gt_replay']=pool([r['perfect_packed_gt_replay'] for r in results])
    audit=[a for r in results for a in r['supervision_audit']]
    missing=[a for a in audit if a['packed_gt'] is None]
    summary['supervision']=dict(official_gt=len(audit),missing_from_training=len(missing),
        missing_with_fewer_than_two_anchors=sum(a['anchors_inside_raw_support']<2 for a in missing),
        missing_recalled=sum(a['final_recalled'] for a in missing),
        supervised_recalled=sum(a['final_recalled'] for a in audit if a['packed_gt'] is not None))
    from collections import Counter
    summary['independent_per_gt_failure_attribution']=dict(Counter(
        a['reason'] for r in results for a in r['failure_attribution']))
    summary['strict_good_but_below_score_threshold']=sum(
        a['covered'] and a['max_matching_score']<.5 for r in results for a in r['decoder_per_gt'])
    summary['target_counts']=dict(packed_gt=sum(r['packed_gt_count'] for r in results),
        full_target_lanes=sum(r['full_target_lane_count'] for r in results),
        full_target_modes=sum(r['template_count'] for r in results))
    template_rows=[a for r in results for a in r['template_matches']]
    summary['template_error_stats']={key:dict(mean=float(np.mean([a[key] for a in template_rows])),
        median=float(np.median([a[key] for a in template_rows])),
        p90=float(np.percentile([a[key] for a in template_rows],90)))
        for key in ('mean_xz_m','mae_x_m','mae_z_m')}
    return summary


def plot_aggregate(summary):
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    names=['q0_505','layer0_505_full','layer0_selected64_full','decoder64_full',
           'decoder64_visibility','score_pass_visibility','final']
    labels=['Q0 505','Layer 0','Selected 64','Decoder 64','+ visibility','+ score 0.5','+ NMS']
    values=[summary['stages'][n]['oracle_one_to_one_matches'] for n in names]
    denom=summary['stages']['final']['gt_count']
    axes[0].barh(labels[::-1],np.array(values[::-1])/denom*100,color='#4487a4')
    for i,n in enumerate(values[::-1]):axes[0].text(n/denom*100+1,i,f'{n}/{denom}',va='center',fontsize=8)
    axes[0].set(xlim=(0,117),xlabel='GT coverage (%)',title='Official-distance oracle coverage\nMaximum one-to-one recall; not actual F1')
    sweep=summary['threshold_sweep'];ts=[float(t) for t in sweep]
    for key,color in [('f1','#1c6190'),('recall','#bd593c'),('precision','#38875b')]:
        axes[1].plot(ts,[100*sweep[t][key] for t in sweep],marker='o',ms=3,color=color,label=key)
    axes[1].axvline(.5,color='gray',ls='--',lw=.8);axes[1].legend()
    axes[1].set(xlabel='Score threshold',ylabel='Official metric (%)',title='Actual metric after visibility + NMS\nSame cached predictions; one variable changed')
    for a in axes:a.grid(axis='x',alpha=.15)
    fig.suptitle(f'{summary["frames"]} uniformly spaced validation frames | checkpoint epoch 17\nDiagnostic subset, not a full-validation rerun',fontsize=12)
    fig.savefig(FIG/'validation128_recall_audit.svg');fig.savefig('/tmp/laneloom_aggregate_preview.png',dpi=120);plt.close(fig)


def endpoint_replay(paths):
    """GT-assisted counterfactual; changes only packed-lane endpoint export."""
    rows=[]
    for path in paths:
        c=torch.load(path,map_location='cpu',weights_only=False);b=c['batch'];raw=b['raw_label'][0]
        gt=arr(b['gt_lanes_3d'][0]);vis=arr(b['gt_lanes_visibility'][0])>.5
        tracks=arr(b['gt_lane_track_ids'][0]);raw_lanes,rv,categories=EV.ground_truth_from_raw_label(raw)
        preds=[];cats=[]
        for q in np.flatnonzero(vis.sum(-1)>=2):
            ids=[i for i,l in enumerate(raw['lane_lines']) if l.get('track_id',-2)==int(tracks[q])]
            assert len(ids)==1, 'Raw annotation track must map uniquely to its packed target'
            i=ids[0];lane=raw_lanes[i][np.asarray(rv[i])>0];orig=gt[q,vis[q]]
            lo=max(5.,float(lane[:,1].min()));hi=min(100.,float(lane[:,1].max()))
            ys=np.unique(np.r_[lo,orig[:,1],hi])
            preds.append(np.column_stack((
                interp1d(orig[:,1],orig[:,0],fill_value='extrapolate')(ys),ys,
                interp1d(orig[:,1],orig[:,2],fill_value='extrapolate')(ys))))
            cats.append(categories[i])
        s=EV.evaluate(preds,cats,raw)
        rows.append(dict(frame=path.stem,**{k:s[k] for k in
            ('gt_count','pred_count','recall_matches','precision_matches')}))
    totals={k:sum(s[k] for s in rows) for k in ('gt_count','pred_count','recall_matches','precision_matches')}
    totals.update(metrics(totals))
    return dict(method='GT-assisted oracle: preserve packed GT geometry; linearly extend endpoints '
                'by at most 5 m to raw annotation support, clipped to 5-100 m. '
                'Missing targets remain missing. This is not an inference fix.',summary=totals,scenes=rows)


def plot_endpoint(result):
    analysis=json.loads((OUT/'analysis.json').read_text())
    source=next(s for s in analysis['scenes'] if s['frame']=='150716535143964300')
    lane=next(s for s in source['perfect_packed_gt_replay']['per_gt'] if s['gt']==1)
    before=analysis['summary']['perfect_packed_gt_replay']
    after=result['summary']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for level,span,color,label in [(2,lane['gt_y'],'#4487a4','Official visible support'),
        (1,lane['pred_y'],'#bc634e','Perfect packed target'),
        (0,lane['gt_y'],'#4d896a','Oracle endpoint restoration')]:
        axes[0].plot(span,[level,level],color=color,lw=8,solid_capstyle='butt')
        axes[0].scatter(span,[level,level],color=color,s=30,zorder=4)
        axes[0].text(span[0],level+.16,f'{span[0]:.0f}-{span[1]:.0f} m',fontsize=9)
    axes[0].set(yticks=[0,1,2],yticklabels=['Restored endpoints','Packed target','Raw GT'],
                xlabel='Forward y (m)',xlim=(18,38),ylim=(-.4,2.7),
                title='Observed example: 150716535143964300 / GT 1\nOnly 6/14 samples match: recall ratio 42.9% < 75%')
    values=[before['recall']*100,after['recall']*100]
    axes[1].bar(['Packed GT replay','Same curves + raw endpoints'],values,color=['#bc634e','#4d896a'],width=.55)
    for i,v in enumerate(values):axes[1].text(i,v+2,f'{v:.2f}%',ha='center')
    axes[1].set(ylim=(0,100),ylabel='Official Recall (%)',
                title=f'{analysis["summary"]["frames"]} frames, {before["gt_count"]} official GT\nGT-assisted counterfactual; no missing targets added')
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle('Endpoint quantization loses recall even with perfect training-target geometry',fontsize=12)
    fig.savefig(FIG/'validation128_endpoint_bias.svg');fig.savefig('/tmp/laneloom_endpoint_preview.png',dpi=120);plt.close(fig)


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-count',type=int,default=0)
    parser.add_argument('--endpoint-only',action='store_true')
    args=parser.parse_args()
    if args.sample_count:OUT=OUT/f'validation{args.sample_count}'
    torch.set_num_threads(4);FIG.mkdir(parents=True,exist_ok=True)
    results=[]
    paths=sorted(OUT.glob('*.pt'))
    if args.sample_count:
        assert len(paths)==args.sample_count, 'Finish capturing the requested subset before reporting'
    if args.endpoint_only:
        result=endpoint_replay(paths)
        (OUT/'endpoint_oracle.json').write_text(json.dumps(result,indent=2))
        if args.sample_count==128:plot_endpoint(result)
        print(result['summary'])
        return
    for path in paths:
        c=torch.load(path,map_location='cpu',weights_only=False)
        r,stages=analyze(c);results.append(r)
        if not args.sample_count:plot(c,r,stages)
        print(r['frame'],r['scene_type'],'strict',[(k,v['strict']['covered']) for k,v in r['stages'].items()],
              'official',r['stages']['final']['official']['f1'],flush=True)
    summary=aggregate(results)
    (OUT/'analysis.json').write_text(json.dumps(dict(scenes=results,summary=summary,
        log=log_data() if not args.sample_count else []),indent=2))
    if args.sample_count:plot_aggregate(summary)


if __name__=='__main__':main()
