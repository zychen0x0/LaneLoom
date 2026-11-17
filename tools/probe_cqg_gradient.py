"""Check decoder-only gradients into CQG using the real checkpoint and losses.

Backbone features are held fixed; the lane head runs in training mode. No
optimizer step is performed. Template/Q0 losses are deliberately excluded.
"""
import json
import sys
from pathlib import Path

import torch
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from LaneLoom.vis.output.infer_openlane_sample import _build_model, _build_sample


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    directory = ROOT / 'work_dirs/epoch17_recall_diagnostic'
    cached = torch.load(directory / '150696587520602900.pt', weights_only=False)
    cfg = Config.fromfile(directory / 'checkpoint_config.py')
    model = _build_model(cfg, Path(cached['checkpoint']), torch.device('cpu'))
    from spconv.core import ConvAlgo
    from spconv.pytorch.conv import SparseConvolution
    for module in model.modules():
        if isinstance(module, SparseConvolution):
            module.algo = ConvAlgo.Native
    image = (Path(cfg.dataset_cfg.data_image_dir) / 'validation'
             / cached['scene']['scene'] / (cached['scene']['frame'] + '.jpg'))
    batch = _build_sample(cfg, image)
    with torch.no_grad():
        images, lidar, metas, _, _ = model._encode_batch(batch)
    head = model.lane_head
    head.train()
    prior = head._anchor_prior(images, lidar, metas)
    bev = head.forward_bev(images, lidar, metas)
    def required(predictions):
        return head.loss_module.required_prediction_mask(
            predictions, batch['gt_lanes_3d'], batch['gt_lanes_visibility'])
    decoded = head._predict(
        bev, anchor_prior=prior,
        query_feature_sampler=head._query_feature_sampler(
            images, bev, metas, head._lidar_occupancy(lidar, bev)),
        required_query_selector=required)
    losses = head.loss_module(decoded, **model._lane_targets(batch))
    observed = {
        'template_x_controls': prior['shape_x_controls'],
        'template_z_controls': prior['shape_z_controls'],
        'template_features': prior['shape_query_features'],
        **{'CQG.' + name: param for name, param in head.anchor_prior.named_parameters()},
    }
    groups = {
        'final_decoder_geometry': ['loss_x', 'loss_z', 'loss_tube'],
        'final_decoder_score': ['loss_score'],
        'layer0_auxiliary': [k for k in losses if k.startswith('loss_aux_0_')],
    }
    results = {}
    for group, names in groups.items():
        loss = sum(losses[n] for n in names)
        grads = torch.autograd.grad(loss, list(observed.values()),
                                    retain_graph=True, allow_unused=True)
        norms = {name: None if grad is None else float(grad.norm())
                 for name, grad in zip(observed, grads)}
        assert any(v is not None and v > 0 for k, v in norms.items()
                   if k.startswith('CQG.template_head')), group
        results[group] = dict(loss_terms=names, value=float(loss.detach()), gradient_l2=norms)
    result = dict(checkpoint=cached['checkpoint'], frame=cached['scene']['frame'],
                  method='Fixed real backbone features, train-mode lane head, actual decoder losses; '
                         'no template/Q0 loss, no optimizer step; seed=0, CPU.',
                  groups=results)
    (directory / 'cqg_gradient_probe.json').write_text(json.dumps(result, indent=2))
    for name, row in results.items():
        print(name, {k: v for k, v in row['gradient_l2'].items()
                     if k in ('template_x_controls', 'template_z_controls', 'template_features')
                     or k.startswith('CQG.template_head')}, flush=True)


if __name__ == '__main__':
    main()
