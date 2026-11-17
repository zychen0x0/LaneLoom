"""Verify the replay metrics and build a standalone diagnostic scene viewer."""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from LaneLoom.evaluation.openlane_official_metric import OpenLaneOfficialMetric
from LaneLoom.models.valid_range_tube_iou import valid_range_tube_iou
from LaneLoom.tools.evaluate_q0_prior import raw_lanes
from LaneLoom.tools.report_epoch17_recall import arr


def verify(root, reports):
    old = json.loads((ROOT / 'work_dirs/q0_bev_holdout/aligned/continued_to_step_21000/'
                     'official_validation_verified/comparison_step_21000.json').read_text())
    expected = {r['frame']:r['spatial'] for r in old['samples']}
    for sample in reports['initialization']['samples']:
        for key, prior_key in [('strict','paired_xz_p80'),('relaxed','paired_xz_p75_05')]:
            actual = sample['stages']['q0'][key]
            prior = expected[sample['scene']['frame']][prior_key]
            assert actual['gt'] == prior['gt'] and actual['covered'] == prior['covered']
    metric = OpenLaneOfficialMetric()
    verified = 0
    for name, report in reports.items():
        for sample in report['samples']:
            record = torch.load(root/name/(sample['scene']['frame']+'.pt'), weights_only=False)
            p = record['prediction']
            inputs = dict(pred_x=p['x'][0], pred_z=p['z'][0],
                          pred_visibility=p['visibility'][0], pred_scores=p['scores'][0],
                          pred_category=p['category'][0], anchor_y_steps=p['anchor_y_steps'],
                          pred_nms_keep_queries=p['nms_keep_queries'][0],
                          raw_label=record['batch']['raw_label'][0])
            actual = metric._evaluate_sample(inputs)
            expected_final = sample['stages']['final']['official']
            for key in ('gt_count','pred_count','recall_matches','precision_matches','match_count'):
                assert actual[key] == expected_final[key], (name,sample['scene']['frame'],key)
            verified += 1
    return dict(pretraining_coverage_frames_verified=47,
                final_predictions_verified_against_metric=verified,
                status='passed')


def probe_targets(root, report):
    """Describe actual training targets and scores of official-eligible queries."""
    evaluator = OpenLaneOfficialMetric().evaluator
    totals = dict(current_lanes=0,enhanced_lanes=0,same_track_lanes=0,
                  changed_support_lanes=0,current_visible_anchors=0,
                  enhanced_visible_anchors_same_tracks=0,added_anchors_same_tracks=0)
    rows = []
    for sample in report['samples']:
        r = torch.load(root/'epoch4'/(sample['scene']['frame']+'.pt'),weights_only=False)
        b,p = r['batch'],r['prediction']
        raw = b['raw_label'][0]
        y=arr(p['anchor_y_steps'])
        x,z,v,s = [arr(p[k][0]) for k in ('x','z','visibility','scores')]
        gl,gv,_ = evaluator.ground_truth_from_raw_label(raw)
        gl=[g[np.asarray(m)>0] for g,m in zip(gl,gv)]
        gl,gids=evaluator._prepare_lanes(gl,list(range(len(gl))))
        gl,gv,gids=evaluator._resample_lanes(gl,gids)
        lanes=[np.column_stack((x[q,v[q]>=.5],y[v[q]>=.5],z[q,v[q]>=.5])) for q in range(len(x))]
        pl,pids=evaluator._prepare_lanes(lanes,list(range(len(lanes))))
        pl,pv,pids=evaluator._resample_lanes(pl,pids)
        if len(gl) and len(pl):
            pairs,_,cost=evaluator._pair_matrices(gl,gv,pl,pv)
            for g,gid in enumerate(gids):
                good=[j for j in range(len(pl)) if
                      pairs[g,j]['num_matches']/gv[g].sum()>=.75 and cost[g,j]<150]
                ids=np.asarray(pids)[good]
                if not len(ids):
                    continue
                q=int(ids[np.argmax(s[ids])])
                item=dict(frame=sample['scene']['frame'],group=sample['scene']['group'],
                          json_index=int(gid),query=q,max_eligible_score=float(s[q]))
                track=int(raw['lane_lines'][gid]['track_id'])
                for name,suffix in [('current',''),('enhanced','_enhanced')]:
                    indices=np.flatnonzero(arr(b['gt_lane_track_ids'+suffix][0])==track)
                    if len(indices):
                        i=int(indices[0]);gt=b['gt_lanes_3d'+suffix][0,i]
                        visibility=b['gt_lanes_visibility'+suffix][0,i]
                        iou=valid_range_tube_iou(p['x'][0,q],p['z'][0,q],p['visibility'][0,q],
                                                 gt[:,0],gt[:,2],visibility)
                        item[name+'_tube_iou']=float(iou)
                rows.append(item)
        cv=b['gt_lanes_visibility'][0]>.5; en=b['gt_lanes_visibility_enhanced'][0]>.5
        ci=b['gt_lane_track_ids'][0]; ei=b['gt_lane_track_ids_enhanced'][0]
        totals['current_lanes']+=int((cv.sum(-1)>=2).sum())
        totals['enhanced_lanes']+=int((en.sum(-1)>=2).sum())
        for i in torch.where(cv.sum(-1)>=2)[0]:
            matches=torch.where(ei==ci[i])[0]
            if len(matches):
                j=matches[0];totals['same_track_lanes']+=1
                totals['changed_support_lanes']+=int(not torch.equal(cv[i],en[j]))
                totals['current_visible_anchors']+=int(cv[i].sum())
                totals['enhanced_visible_anchors_same_tracks']+=int(en[j].sum())
                totals['added_anchors_same_tracks']+=int((en[j]&~cv[i]).sum())
    summary=dict(target_counts=totals,eligible_gt=len(rows),
                 score_below_05=sum(r['max_eligible_score']<.5 for r in rows),
                 score_04_to_05=sum(.4<=r['max_eligible_score']<.5 for r in rows))
    for name in ('current','enhanced'):
        values=[r[name+'_tube_iou'] for r in rows if name+'_tube_iou' in r]
        summary[name+'_tube_median']=float(np.median(values))
        summary[name+'_tube_below05']=sum(v<.5 for v in values)
    return dict(summary=summary,lanes=rows)


def make_viewer(root,reports):
    frames={}
    for name,report in reports.items():
        if name=='epoch4_budget128':
            continue
        for sample in report['samples']:
            scene=sample['scene'];frame=scene['frame']
            r=torch.load(root/name/(frame+'.pt'),weights_only=False)
            b,p=r['batch'],r['prediction'];calib=b['img_metas'][0]['current']['openlane_calib']
            lanes=raw_lanes(b['raw_label'][0],calib)
            if frame not in frames:
                im=Image.open(ROOT/scene['image']);im.thumbnail((800,530))
                buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=80)
                original=Image.open(ROOT/scene['image'])
                frames[frame]=dict(scene=scene,gt=[l['points'].tolist() for l in lanes],
                                   image=base64.b64encode(buffer.getvalue()).decode(),
                                   size=list(original.size),projection=np.asarray(calib['ground_to_raw_img']).tolist(),models={})
            d=r['data']['decoder'];prior=r['data']['prior']
            frames[frame]['models'][name]=dict(
                qx=arr(prior['shape_x_controls'][0]).tolist(),
                qz=arr(prior['shape_z_controls'][0]).tolist(),
                fx=arr(d['curve_x_controls'][0]).tolist(),fz=arr(d['curve_z_controls'][0]).tolist(),
                vis=arr(p['visibility'][0]).tolist(),scores=arr(p['scores'][0]).tolist(),
                keep=arr(p['valid_queries'][0]).tolist(),y=arr(p['anchor_y_steps']).tolist(),
                stages={k:dict(strict=v['strict']['covered'],relaxed=v['relaxed']['covered'])
                        for k,v in sample['stages'].items()},
                qmatches=sample['stages']['q0']['strict']['lanes'],
                raw_gt=sample['raw_gt'],official=sample['stages']['final']['official'])
    template=VIEWER.replace('__DATA__',json.dumps(list(frames.values()),separators=(',',':')))
    (root/'scenes.html').write_text(template)


VIEWER = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Q0 与最终输出覆盖诊断</title>
<style>body{font:15px system-ui;margin:24px;background:#f5f7fa;color:#172335}select,button{padding:8px;margin:4px}canvas{background:white;width:100%;border:1px solid #ccd4df;border-radius:6px}.plots{display:grid;grid-template-columns:1.5fr 1fr 1fr;gap:14px}.card{padding:14px;background:white;margin:12px 0;border-radius:8px}table{border-collapse:collapse;width:100%}td,th{padding:7px;text-align:left;border-bottom:1px solid #ddd}.note{color:#526176}h2{font-size:18px}#camera{aspect-ratio:1.5}#q0,#final{aspect-ratio:.65}</style>
<h1>验证场景：Q0 → 解码 → 最终输出</h1><p class="note">47 帧人工挑选场景；覆盖与完整验证集 F1 分开解读。黑色虚线：原始可见 GT；橙色：为诊断匹配的 Q0；蓝色：模型最终输出。</p>
<div class="card"><select id="model"><option value="epoch4">最佳 epoch 4</option><option value="epoch5">最后 epoch 5</option><option value="original_epoch43">旧 detector epoch 43</option><option value="initialization">合成初始化（未微调）</option></select><select id="group"><option value="all">所有场景</option></select><button id="prev">上一帧</button><select id="frame"></select><button id="next">下一帧</button><label><input type="checkbox" id="all">显示全部 505 个 Q0</label></div>
<div id="stats" class="card"></div><div class="plots"><div><h2>相机投影：GT 与最终输出</h2><canvas id="camera" width="900" height="600"></canvas></div><div><h2>Q0 与 GT（x / y，单位 m）</h2><canvas id="q0" width="450" height="700"></canvas></div><div><h2>最终输出与 GT（x / y，单位 m）</h2><canvas id="final" width="450" height="700"></canvas></div></div><p id="path" class="note"></p>
<p class="note">橙色 Q0 按 GT 做一对一诊断匹配，未参与推理或输出筛选。未覆盖的匹配候选显示为红色。严格覆盖：至少 80% 原始可见点的 xz 误差小于 0.3 m；宽松覆盖：至少 75% 点小于 0.5 m。最终覆盖还施加预测可见区间；官方指标另用 evaluator 的 ROI、重采样和最小代价匹配。</p>
<script>const data=__DATA__;const $=id=>document.getElementById(id);const labels={normal:'正常',occlusion:'遮挡',slope:'上下坡',intersection:'交叉路口',curve:'弯道',merge_split:'汇聚分叉',night:'夜间',backlight:'逆光',extreme:'极端',complex:'复杂'};
for(const g of [...new Set(data.map(r=>r.scene.group))])$('group').add(new Option(labels[g],g));
function refill(){const old=$('frame').value;$('frame').innerHTML='';for(const r of data.filter(r=>$('group').value==='all'||r.scene.group===$('group').value))$('frame').add(new Option(labels[r.scene.group]+' / '+r.scene.frame,r.scene.frame));if([...$('frame').options].some(o=>o.value===old))$('frame').value=old;draw()}
function cubic(c,t){return c[0]*(1-t)**3+3*c[1]*(1-t)**2*t+3*c[2]*(1-t)*t*t+c[3]*t**3}
function line(ctx,pts,color,width=2,dash=[]){ctx.beginPath();let active=false;for(const p of pts){if(!p||!Number.isFinite(p[0])||!Number.isFinite(p[1])){active=false;continue}if(active)ctx.lineTo(...p);else ctx.moveTo(...p);active=true}ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([])}
function bez(cx,cz,offset=0,ys=null,mask=null){return (ys||Array.from({length:101},(_,i)=>i)).map((y,i)=>mask&&!mask[i]?null:[cubic(cx,y/100)+offset,y,cubic(cz,y/100)])}
function base(id){const c=$(id),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.font='14px system-ui';const xy=p=>p?[40+(p[0]+25)/50*(c.width-60),c.height-35-p[1]/100*(c.height-65)]:null;ctx.strokeStyle='#e2e7ee';ctx.fillStyle='#526176';for(let y=0;y<=100;y+=20){const yy=xy([0,y])[1];line(ctx,[[40,yy],[c.width-20,yy]],'#e2e7ee',1);ctx.fillText(y,8,yy+4)}for(let x=-20;x<=20;x+=10){const xx=xy([x,0])[0];line(ctx,[[xx,30],[xx,c.height-35]],'#e2e7ee',1);ctx.fillText(x,xx-9,c.height-10)}return [ctx,xy]}
let generation=0;function draw(){const gen=++generation;const r=data.find(r=>r.scene.frame===$('frame').value);if(!r)return;const m=r.models[$('model').value],f=m.official;const pct=x=>(100*x).toFixed(2)+'%';$('stats').innerHTML=`<b>${labels[r.scene.group]} / ${r.scene.frame}</b>　原始 ROI GT：${m.raw_gt}　官方 ROI GT：${f.gt_count}　官方 R/P/F1：${pct(f.recall)} / ${pct(f.precision)} / ${pct(f.f1)}<table><tr><th>阶段</th><th>严格覆盖</th><th>宽松覆盖</th></tr>${Object.entries(m.stages).map(([k,v])=>`<tr><td>${k}</td><td>${v.strict}/${m.raw_gt}</td><td>${v.relaxed}/${m.raw_gt}</td></tr>`).join('')}</table>`;$('path').textContent=r.scene.image;
const [qc,qxy]=base('q0'),[fc,fxy]=base('final');qc.save();qc.beginPath();qc.rect(40,30,390,635);qc.clip();fc.save();fc.beginPath();fc.rect(40,30,390,635);fc.clip();
if($('all').checked)for(let t=0;t<5;t++)for(let seed=0;seed<=100;seed++)line(qc,bez(m.qx[t],m.qz[t],seed*.5-25).map(qxy),'#a5afbd22',1);
for(const match of m.qmatches){if(match.query_id===null)continue;const t=Math.floor(match.query_id/101),seed=match.query_id%101;line(qc,bez(m.qx[t],m.qz[t],seed*.5-25).map(qxy),match.covered?'#dc8a1b':'#da4444',2)}
const predicted=[];for(let q=0;q<m.keep.length;q++)if(m.keep[q]){const p=bez(m.fx[q],m.fz[q],0,m.y,m.vis[q].map(v=>v>=.5));predicted.push(p);line(fc,p.map(fxy),'#1976d2',2.5)}
for(const gt of r.gt){line(qc,gt.map(qxy),'#182533',2,[5,4]);line(fc,gt.map(fxy),'#182533',2,[5,4])}qc.restore();fc.restore();
const ctx=$('camera').getContext('2d'),im=new Image();im.onload=()=>{if(gen!==generation)return;ctx.clearRect(0,0,900,600);ctx.drawImage(im,0,0,900,600);function proj(p){if(!p)return null;const v=[...p,1],a=r.projection.map(row=>row.reduce((s,z,j)=>s+z*v[j],0));if(a[2]<=0)return null;return [a[0]/a[2]/r.size[0]*900,a[1]/a[2]/r.size[1]*600]}for(const gt of r.gt)line(ctx,gt.map(proj),'#ff4141',2.5,[7,4]);for(const p of predicted)line(ctx,p.map(proj),'#20edc0',3);ctx.fillStyle='#000b';ctx.fillRect(8,8,340,30);ctx.fillStyle='white';ctx.font='15px system-ui';ctx.fillText('相机投影：红虚线 GT / 绿色最终输出',16,29)};im.src='data:image/jpeg;base64,'+r.image}
$('group').onchange=refill;$('frame').onchange=draw;$('model').onchange=draw;$('all').onchange=draw;for(const [id,step]of [['prev',-1],['next',1]])$(id).onclick=()=>{$('frame').selectedIndex=($('frame').selectedIndex+step+$('frame').length)%$('frame').length;draw()};refill();</script></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'work_dirs/geometric_q0_recall_audit')
    args=parser.parse_args();torch.set_num_threads(4)
    names=('initialization','original_epoch43','epoch4','epoch5','epoch4_budget128')
    reports={n:json.loads((args.root/n/'report.json').read_text()) for n in names}
    result=verify(args.root,reports)
    (args.root/'verification.json').write_text(json.dumps(result,indent=2))
    probe=probe_targets(args.root,reports['epoch4'])
    (args.root/'target_score_probe.json').write_text(json.dumps(probe,indent=2))
    make_viewer(args.root,reports)
    print(json.dumps(dict(verification=result,probe=probe['summary']),indent=2))


if __name__=='__main__':
    main()
