"""Pretrain the LiDAR encoder and Q0 prior on official training data.

This isolates Q0 extraction from the image backbone and downstream decoder.
Architecture comparisons share data, geometry loss, seed lattice and schedule.
Optional lane-field supervision is recorded explicitly as a loss ablation.
Training-scene holdouts require fresh encoders. Official-validation comparisons
may declare a shared training-only encoder; extraction heads still start fresh.
"""
import argparse
import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config
from mmdet3d.registry import DATASETS, MODELS
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits

from LaneLoom.datasets.openlane.dataset import openlane_collate, openlane_raw_visible_gt
from LaneLoom.engine import register_all_modules
from LaneLoom.models.losses.q0_visible_geometry import raw_visible_geometry_loss
from LaneLoom.models.losses.bev_lane_field import bev_lane_field_loss


def load_training_encoder(encoder, path):
    """Import only a shared encoder with an explicit official-training source."""
    path = Path(path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    meta = saved.get('meta', {})
    if meta.get('invalid_for_model_comparison') or 'cfg' not in meta:
        raise ValueError('Encoder source lacks a valid training configuration')
    source = Config.fromstring(meta['cfg'], '.py')
    dataset = source.train_dataloader.dataset
    if (dataset.type != 'OpenLaneDataset' or dataset.get('split') not in ('train', 'training')
            or dataset.dataset_cfg.get('train_split') != 'training'):
        raise ValueError('Encoder source must use only the official training partition')
    prefix = 'pts_bev_encoder.'
    state = saved['state_dict']
    weights = OrderedDict((key[len(prefix):], value) for key, value in state.items()
                          if key.startswith(prefix))
    metadata = getattr(state, '_metadata', None)
    if metadata is not None:
        # MMDetection3D's sparse loader uses version metadata to decide kernel
        # layout conversion. Prefix stripping must preserve that metadata too.
        weights._metadata = OrderedDict(
            (key[len(prefix):] if key.startswith(prefix) else '', value)
            for key, value in metadata.items()
            if key == prefix[:-1] or key.startswith(prefix))
    if not weights:
        raise ValueError('Checkpoint contains no LiDAR encoder')
    encoder.load_state_dict(weights, strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return dict(encoder_initialization='shared official-training checkpoint',
                encoder_source_checkpoint=str(path), encoder_source_sha256=digest.hexdigest(),
                encoder_source_cfg_sha256=hashlib.sha256(meta['cfg'].encode()).hexdigest(),
                encoder_source_epoch=meta.get('epoch'), encoder_source_train_split=dataset.split)


class Q0Pretrainer(nn.Module):
    def __init__(self, config, initialize_encoder=True):
        super().__init__()
        source = config.get('q0_encoder_checkpoint')
        self.freeze_encoder = bool(config.get('q0_freeze_encoder', False))
        if source and config.get('q0_evaluation_split') != 'validation':
            raise ValueError('A training-pretrained encoder cannot be used for training-scene holdout claims')
        if self.freeze_encoder and not source:
            raise ValueError('Frozen encoder requires an explicit shared training checkpoint')
        self.pts_bev_encoder = MODELS.build(config.model.pts_bev_encoder)
        self.encoder_source_info = (load_training_encoder(self.pts_bev_encoder, source)
                                    if source and initialize_encoder else {})
        if self.freeze_encoder:
            self.pts_bev_encoder.requires_grad_(False)
            self.pts_bev_encoder.eval()
        self.anchor_prior = MODELS.build(config.model.lane_head.anchor_prior)
        self.proposal_expander = MODELS.build(config.model.lane_head.proposal_expander)
        # The seed/query embeddings only feed the downstream decoder. Their
        # geometry is unchanged; train only parameters contributing to Q0 loss.
        self.proposal_expander.requires_grad_(False)
        self.raw_visible_supervision = bool(config.get('q0_raw_visible_supervision', False))
        self.lane_field_weight = float(config.get('q0_lane_field_loss_weight', 0.))
        if self.lane_field_weight < 0 or (self.lane_field_weight and not self.raw_visible_supervision):
            raise ValueError('Lane field supervision requires raw-visible geometry and a nonnegative weight')

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            # Frozen parameters alone do not freeze BatchNorm running state.
            self.pts_bev_encoder.eval()
        return self

    def forward(self, batch, device):
        with (torch.no_grad() if self.freeze_encoder else nullcontext()):
            features = self.pts_bev_encoder(
                [p.to(device, non_blocking=True) for p in batch['lidar']],
                images=batch['img'].to(device, non_blocking=True), img_metas=batch['img_metas'])
        prior = self.anchor_prior(features)
        prior.update(self.proposal_expander(prior))
        if self.raw_visible_supervision:
            # Annotations are consumed only after the point-cloud prediction.
            points, visible = openlane_raw_visible_gt(batch['raw_label'], self.proposal_expander.pc_range)
            losses = raw_visible_geometry_loss(
                prior, points.to(device), visible.to(device),
                length=float(self.anchor_prior.anchor_y_steps[-1]))
            if self.lane_field_weight:
                field_losses = bev_lane_field_loss(prior, points.to(device), visible.to(device),
                                                   self.proposal_expander.pc_range)
                losses.update({key: value * self.lane_field_weight if key.startswith('loss_') else value
                               for key, value in field_losses.items()})
            return losses
        gt = {key: value.to(device, non_blocking=True) for key, value in batch.items()
              if key.startswith('gt_') and torch.is_tensor(value)}
        lanes, visibility = gt['gt_lanes_3d'], gt['gt_lanes_visibility']
        losses = self.anchor_prior.loss(
            prior, lanes, visibility, shape_templates=gt.get('gt_shape_templates'),
            shape_template_valid=gt.get('gt_shape_template_valid'),
            shape_template_z_weights=gt.get('gt_shape_template_z_weights'))
        q0_lanes = gt.get('gt_lanes_3d_full', lanes)
        q0_visibility = visibility
        if 'gt_lanes_full_valid' in gt:
            q0_visibility = gt['gt_lanes_full_valid'][..., None].expand(q0_lanes.shape[:-1])
        losses.update(self.proposal_expander.loss(
            prior, q0_lanes, q0_visibility, z_weights=gt.get('gt_lanes_full_z_weights')))
        return losses


def seed_worker(worker_id):
    # Small calibration/curve solves otherwise start large BLAS thread pools
    # in every worker, competing with the other concurrent training jobs.
    threadpool_limits(limits=1, user_api='blas')
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def training_dataset_signature(config):
    encoded = json.dumps(config.train_dataloader.dataset.to_dict(),
                         sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def model_training_signature(config):
    components = dict(encoder=config.model.pts_bev_encoder.to_dict(),
                      prior=config.model.lane_head.anchor_prior.to_dict(),
                      expander=config.model.lane_head.proposal_expander.to_dict(),
                      optimizer=config.optim_wrapper.optimizer.to_dict())
    if config.get('q0_raw_visible_supervision', False):
        components['q0_raw_visible_supervision'] = True
    if 'q0_lane_field_loss_weight' in config:
        components['q0_lane_field_loss_weight'] = float(config.q0_lane_field_loss_weight)
    for key in ('q0_encoder_checkpoint', 'q0_freeze_encoder'):
        if key in config:
            components[key] = config[key]
    return hashlib.sha256(json.dumps(components, sort_keys=True, default=str).encode()).hexdigest()


def validate_resume_protocol(saved, protocol):
    old = saved['meta']['protocol']
    for key in ('encoder_source_sha256', 'encoder_source_cfg_sha256', 'encoder_frozen'):
        if old.get(key) != protocol.get(key):
            raise ValueError(f'Resume encoder source or freezing differs: {key}')
    for key in ('training_labels_sha256', 'excluded_segments', 'batch_size',
                'model_type', 'epochs', 'seed'):
        if old[key] != protocol[key]:
            raise ValueError(f'Resume protocol differs: {key}')
    signature = old.get('training_dataset_sha256')
    if signature is None:
        signature = training_dataset_signature(
            Config.fromstring(saved['meta']['cfg'], '.py'))
    if signature != protocol['training_dataset_sha256']:
        raise ValueError('Resume training dataset or supervision differs')
    signature = old.get('q0_model_sha256')
    if signature is None:
        signature = model_training_signature(Config.fromstring(saved['meta']['cfg'], '.py'))
    if signature != protocol['q0_model_sha256']:
        raise ValueError('Resume model, loss settings or optimizer differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--stop-after-epoch', type=int,
                        help='End training at this epoch without changing the existing LR schedule')
    parser.add_argument('--steps', type=int, default=0,
                        help='Stop at this optimizer step for staged evaluation; 0 completes --epochs')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    stop_epoch = args.stop_after_epoch if args.stop_after_epoch is not None else args.epochs
    if not 1 <= stop_epoch <= args.epochs:
        parser.error('--stop-after-epoch must be between 1 and --epochs')
    torch.cuda.set_device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    register_all_modules()
    cfg = Config.fromfile(args.config)
    if cfg.get('load_from'):
        raise ValueError('Whole-model load_from is not accepted; declare an encoder-only source explicitly')
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    excluded = set(cfg.q0_excluded_segments)
    official_validation = cfg.get('q0_evaluation_split') == 'validation'
    if ((not excluded and not official_validation)
            or any(p.parent.name in excluded for p in dataset.label_files)):
        raise ValueError('Target segment exclusion is absent or violated')
    if dataset.split != 'training' or any(
            p.relative_to(dataset.label_root).parts[0] != 'training' for p in dataset.label_files):
        raise ValueError('Q0 training must use only the official training partition')
    if official_validation:
        validation_segments = {p.name for p in (dataset.label_root / 'validation').iterdir() if p.is_dir()}
        if any(p.parent.name in validation_segments for p in dataset.label_files):
            raise ValueError('Official validation segments cannot enter Q0 training')
    labels = '\n'.join(str(p.relative_to(dataset.label_root)) for p in dataset.label_files) + '\n'
    digest = hashlib.sha256(labels.encode()).hexdigest()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    protocol = dict(training_labels=len(dataset), training_labels_sha256=digest,
                    training_dataset_sha256=training_dataset_signature(cfg),
                    q0_model_sha256=model_training_signature(cfg),
                    excluded_segments=sorted(excluded), target_overlap=0,
                    encoder_initialization='fresh random weights',
                    feature_cache_used=False, target_oversampling=False,
                    scope='Q0 pretraining, downstream decoder not trained',
                    batch_size=args.batch_size, epochs=args.epochs, seed=7,
                    model_type=cfg.model.lane_head.anchor_prior.type)
    if official_validation:
        protocol['evaluation_split'] = 'validation'
    if cfg.get('q0_raw_visible_supervision', False):
        protocol['supervision'] = 'raw_visible_JSON_points; one-to-one joint xz geometry; no completed templates'
    if cfg.model.pts_bev_encoder.get('align_to_lane_ground', False):
        protocol['bev_coordinate_frame'] = 'OpenLane lane ground, calibration-aligned after RGB painting'
    if cfg.get('q0_lane_field_loss_weight', 0):
        protocol['supervision'] += '; sparse multi-label lane-position and visible-point height field'
        protocol['lane_field_loss_weight'] = float(cfg.q0_lane_field_loss_weight)
    model = Q0Pretrainer(cfg)
    if model.encoder_source_info:
        protocol.update(model.encoder_source_info)
        protocol['encoder_frozen'] = model.freeze_encoder
        protocol['head_initialization'] = 'fresh random weights'
        if model.freeze_encoder:
            protocol['scope'] = 'Q0 head pretraining with a shared frozen encoder; downstream decoder not trained'
    initialization = hashlib.sha256()
    for key, value in model.pts_bev_encoder.state_dict().items():
        initialization.update(key.encode())
        initialization.update(value.contiguous().numpy().tobytes())
    protocol['encoder_initialization_sha256'] = initialization.hexdigest()
    protocol['scene_cache_size'] = dataset.scene_cache_size
    protocol['loader_workers'] = args.num_workers
    protocol['loader_blas_threads'] = 1
    protocol['prefetch_factor'] = 1
    model.to(args.device).train()
    optimizer_cfg = dict(cfg.optim_wrapper.optimizer)
    optimizer_cfg.pop('type')
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), **optimizer_cfg)
    base_lr = optimizer_cfg['lr']
    start_epoch, start_batch, step = 0, 0, 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        validate_resume_protocol(saved, protocol)
        model.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        start_epoch, start_batch, step = saved['epoch'], saved['batch'], saved['step']
    (args.work_dir / 'training_samples.txt').write_text(labels)
    cfg.dump(args.work_dir / 'config.py')
    (args.work_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(json.dumps(protocol), flush=True)
    started = time.monotonic()
    batches_per_epoch = len(dataset) // args.batch_size
    run_budget = dict(schedule_epochs=args.epochs,
                      training_epochs=stop_epoch,
                      updates_per_epoch=batches_per_epoch,
                      schedule_updates=args.epochs * batches_per_epoch,
                      training_updates=stop_epoch * batches_per_epoch,
                      stop_after_step=args.steps or None,
                      resumed_at_step=step)
    (args.work_dir / 'run_budget.json').write_text(json.dumps(run_budget, indent=2))
    print('RUN_BUDGET', json.dumps(run_budget), flush=True)

    def save(epoch, batch):
        state = dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                     epoch=epoch, batch=batch, step=step,
                     completed_epochs=epoch + batch / batches_per_epoch,
                     meta=dict(protocol=protocol, cfg=cfg.pretty_text, run_budget=run_budget))
        temporary = args.work_dir / 'last.tmp.pth'
        torch.save(state, temporary)
        temporary.replace(args.work_dir / 'last.pth')

    for epoch in range(start_epoch, stop_epoch):
        first = start_batch if epoch == start_epoch else 0
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(7 + epoch)).tolist()
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=order[first * args.batch_size:],
                            num_workers=args.num_workers, pin_memory=True, drop_last=True,
                            collate_fn=openlane_collate, worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(7 + epoch),
                            persistent_workers=args.num_workers > 0,
                            **(dict(prefetch_factor=1) if args.num_workers else {}))
        for batch_index, batch in enumerate(loader, first):
            progress = (epoch + batch_index / batches_per_epoch) / args.epochs
            warmup = min(1., 1 / 3 + 2 / 3 * step / 500)
            for group in optimizer.param_groups:
                group['lr'] = base_lr * warmup * (1e-3 + .999 * .5 * (1 + math.cos(math.pi * progress)))
            optimizer.zero_grad(set_to_none=True)
            losses = model(batch, args.device)
            loss = sum(value for key, value in losses.items() if key.startswith('loss_'))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f'Nonfinite gradient at step {step}')
            optimizer.step()
            step += 1
            if step == 1 or step % 50 == 0 or (args.steps and step >= args.steps):
                numbers = {key: round(float(value.detach()), 5) for key, value in losses.items()}
                print('STEP', step, 'epoch', epoch + 1, 'batch', batch_index + 1,
                      'seconds', round(time.monotonic() - started, 1), numbers, flush=True)
            if step == 1 or step % 250 == 0 or (args.steps and step >= args.steps):
                save(epoch, batch_index + 1)
            if args.steps and step >= args.steps:
                return
        save(epoch + 1, 0)
        temporary = args.work_dir / f'epoch_{epoch + 1}.tmp.pth'
        torch.save(torch.load(args.work_dir / 'last.pth', map_location='cpu', weights_only=False), temporary)
        temporary.replace(args.work_dir / f'epoch_{epoch + 1}.pth')


if __name__ == '__main__':
    main()
