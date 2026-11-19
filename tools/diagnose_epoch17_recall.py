"""Replay checkpoint predictions and cache stages for a recall/geometry audit.

The cache and reports live outside the user's SVG-only prototype directory.
Uses the real predict path and hooks; no GT enters inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from LaneLoom.vis.output.infer_openlane_sample import _build_model, _build_sample
from LaneLoom.datasets.openlane.full_shape_targets import build_full_shape_targets
from LaneLoom.datasets.openlane.dataset import openlane_collate
from mmdet3d.registry import DATASETS

OUTPUT = ROOT / 'work_dirs/epoch17_recall_diagnostic'


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu(v) for v in value]
    return value


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Path to the epoch-17 checkpoint.')
    parser.add_argument('--all-scenes', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--sample-count', type=int, default=0,
                        help='Uniformly spaced validation indices, independent of hand-picked scenes')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    if args.sample_count:
        OUTPUT = OUTPUT / f'validation{args.sample_count}'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT / 'scene_manifest.json'
    if not manifest.exists() and not args.sample_count:
        with (ROOT / 'work_dirs/gt_shape_extension_prototype/scene_summary.csv').open() as f:
            scenes = list(csv.DictReader(f))
        manifest.write_text(json.dumps(scenes, indent=2))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = Config.fromstring(checkpoint['meta']['cfg'], '.py')
    meta = {k: checkpoint['meta'].get(k) for k in ('epoch', 'iter')}
    del checkpoint
    dataset = None
    if args.sample_count:
        dataset = DATASETS.build(cfg.val_dataloader.dataset)
        indices = np.linspace(0, len(dataset)-1, args.sample_count, dtype=int)
        scenes = [dict(frame=Path(dataset.label_files[i]).stem,
                       scene=Path(dataset.label_files[i]).parent.name,
                       scene_type='validation',dataset_index=int(i)) for i in indices]
        manifest.write_text(json.dumps(dict(dataset_size=len(dataset),scenes=scenes),indent=2))
    else:
        scenes = json.loads(manifest.read_text())
        scenes.sort(key=lambda s: s['frame'] != '150696587520602900')
        if not args.all_scenes:
            scenes = scenes[:1]
    (OUTPUT / 'checkpoint_config.py').write_text(cfg.pretty_text)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('BUILD', device, meta, flush=True)
    model = _build_model(cfg, args.checkpoint, device)
    if device.type == 'cpu':
        from spconv.core import ConvAlgo
        from spconv.pytorch.conv import SparseConvolution
        for module in model.modules():
            if isinstance(module, SparseConvolution):
                module.algo = ConvAlgo.Native
    capture = {}
    handles = []
    for name, module in [('prior', model.lane_head.anchor_prior),
                         ('q0', model.lane_head.proposal_expander),
                         ('decoder', model.lane_head.lane_decoder)]:
        def hook(module, inputs, output, name=name):
            capture[name] = cpu(output)
        handles.append(module.register_forward_hook(hook))
    for scene in scenes:
        dest = OUTPUT / (scene['frame'] + '.pt')
        if dest.exists():
            print('CACHED', scene['frame'], flush=True)
            continue
        start = time.monotonic()
        image = Path(cfg.dataset_cfg.data_image_dir) / 'validation' / scene['scene'] / (scene['frame'] + '.jpg')
        print('LOAD', image, flush=True)
        batch = (openlane_collate([dataset[scene['dataset_index']]]) if dataset is not None
                 else _build_sample(cfg, image))
        print('FORWARD', scene['frame'], 'image', tuple(batch['img'].shape),
              'lidar', tuple(batch['lidar'][0].shape), flush=True)
        with torch.inference_mode():
            output = model(mode='predict', img=batch['img'].to(device),
                           lidar=[v.to(device) for v in batch['lidar']],
                           img_metas=batch['img_metas'])
        dc = cfg.train_dataloader.dataset.dataset_cfg
        targets = build_full_shape_targets(
            batch['gt_lanes_3d'][0].numpy(), batch['gt_lanes_visibility'][0].numpy(),
            model.lane_head.anchor_y_steps.cpu().numpy(),
            max_templates=dc.max_shape_templates,
            min_lane_points=dc.full_shape_min_points, min_lane_span=dc.full_shape_min_span,
            guarded_short_merge=dc.get('full_shape_guarded_short_merge', False),
            z_extrapolation_grace_m=dc.get('full_shape_z_extrapolation_grace_m', 20.),
            z_extrapolation_decay_m=dc.get('full_shape_z_extrapolation_decay_m'))
        if args.sample_count:
            capture['decoder'].pop('bev', None)
            capture['q0'].pop('query_features', None)
        record = dict(scene=scene, checkpoint=str(args.checkpoint), meta=meta,
                      backend=str(device), stages=capture.copy(), output=cpu(output),
                      targets=targets, batch={k: cpu(v) for k, v in batch.items()
                                             if k != 'img' and not (args.sample_count and k=='lidar')})
        torch.save(record, dest)
        capture.clear()
        print('SAVED', dest, 'seconds', round(time.monotonic() - start, 2), flush=True)
    for handle in handles:
        handle.remove()


if __name__ == '__main__':
    main()
