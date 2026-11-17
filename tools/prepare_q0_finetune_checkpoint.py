"""Compose a strict full-detector warm start with a pretrained LiDAR/Q0 pair."""
import argparse
import hashlib
from pathlib import Path

from mmengine.config import Config
from mmdet3d.registry import MODELS
import torch

from LaneLoom.engine import register_all_modules


def compose_state(model, detector_state, q0_state):
    """Copy every parameter, retaining only configuration-derived buffers."""
    target = model.state_dict()
    parameters = set(dict(model.named_parameters()))
    rebuilt = []
    for key, value in target.items():
        if key.startswith('pts_bev_encoder.'):
            source, source_key = q0_state, key
        elif key.startswith('lane_head.anchor_prior.'):
            source, source_key = q0_state, key.replace('lane_head.anchor_prior.', 'anchor_prior.', 1)
        else:
            source, source_key = detector_state, key
        if key not in parameters and not key.startswith(('img_backbone.', 'pts_bev_encoder.')):
            # Sampling anchors/bases/indices/seeds and fixed offset grids are
            # determined by this model's configuration, not learned weights.
            rebuilt.append(key)
            continue
        if source_key not in source or source[source_key].shape != value.shape:
            raise ValueError(f'Warm-start parameter/buffer is absent or incompatible: {key}')
        target[key] = source[source_key].clone()
    # Keep target state metadata: sparse kernels need their version markers.
    return target, rebuilt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--detector-checkpoint', type=Path, required=True)
    parser.add_argument('--q0-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite initialization: {args.output}')
    register_all_modules()
    cfg = Config.fromfile(args.config)
    detector = torch.load(args.detector_checkpoint, map_location='cpu', weights_only=False)
    q0 = torch.load(args.q0_checkpoint, map_location='cpu', weights_only=False)
    source_cfg = Config.fromstring(q0['meta']['cfg'], '.py')
    encoder = cfg.model.pts_bev_encoder.to_dict()
    if encoder != source_cfg.model.pts_bev_encoder.to_dict():
        raise ValueError('LiDAR configuration must match its pretrained Q0 encoder')
    old_prior = source_cfg.model.lane_head.anchor_prior.to_dict()
    new_prior = cfg.model.lane_head.anchor_prior.to_dict()
    # Only the deterministic anchor sampling grid may change on transfer.
    old_prior.pop('anchor_y_steps')
    new_prior.pop('anchor_y_steps')
    # shape_supervision is inactive when supervise_shapes=False.
    old_prior.pop('shape_supervision', None)
    new_prior.pop('shape_supervision', None)
    if new_prior != old_prior:
        raise ValueError('Extraction head differs from the validated pretrained architecture')
    for source in (Config.fromstring(detector['meta']['cfg'], '.py'), source_cfg):
        dataset = source.train_dataloader.dataset
        if dataset.split not in ('train', 'training') or dataset.dataset_cfg.train_split != 'training':
            raise ValueError('Warm-start sources must use the official training partition')
    model = MODELS.build(cfg.model)
    state, rebuilt = compose_state(model, detector['state_dict'], q0['state_dict'])
    model.load_state_dict(state, strict=True)
    sources = {}
    for name, path in (('detector', args.detector_checkpoint), ('lidar_q0', args.q0_checkpoint)):
        sources[name] = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    provenance = dict(sources=sources, detector_epoch=detector['meta'].get('epoch'),
                      q0_training_step=q0['step'], rebuilt_configuration_buffers=rebuilt,
                      transferred_parameters=len(dict(model.named_parameters())),
                      optimizer_restored=False,
                      purpose='Full-model five-epoch fine-tuning; not a matched from-scratch head comparison')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=state, meta=dict(cfg=cfg.pretty_text,
                    initialization=provenance)), args.output)
    print(provenance, flush=True)


if __name__ == '__main__':
    main()
