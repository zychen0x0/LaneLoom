#!/usr/bin/env python3
"""Visualize the effective OpenLane training image pipeline for one sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vis.data_aug._common import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    build_sample_dataset,
    current_image_path,
    load_config,
    make_tensor_canvas,
    output_dir_for,
    require_openlane_data,
    seed_everything,
    select_sample,
    write_report,
)


def _load_resized_originals(sample):
    tensors = []
    for frame_index in range(sample['img'].shape[0]):
        frame_meta = sample['img_metas']['queue'][frame_index]
        height, width = frame_meta['img_shape'][:2]
        image = Image.open(frame_meta['img_filename'][0]).convert('RGB')
        image = image.resize((width, height), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy().transpose(2, 0, 1)
        norm_cfg = frame_meta['img_norm_cfg']
        mean = np.asarray(norm_cfg['mean'], dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(norm_cfg['std'], dtype=np.float32).reshape(3, 1, 1)
        tensors.append(torch.from_numpy((array - mean) / std))
    return torch.stack(tensors, dim=0).unsqueeze(1)


def _build_report(cfg, sample):
    grid_cfg = next((
        dict(stage) for stage in cfg.train_pipeline
        if stage['type'] == 'GridMask'
    ), None)
    zero_mask = torch.all(sample['img'] == 0, dim=2)
    flattened_masks = zero_mask.reshape(-1, *zero_mask.shape[-2:])
    masks_are_shared = all(
        torch.equal(flattened_masks[0], mask)
        for mask in flattened_masks[1:])
    augmentation_mode = sample['img_metas']['image_augmentation_mode']
    recipe_names = sample['img_metas']['augmentation_recipe_names']
    recipe_cfgs = {
        recipe_cfg['name']: recipe_cfg
        for recipe_cfg in cfg.image_augmentation_recipes
    }
    selection_frequency = (
        {name: 1.0 / len(recipe_cfgs) for name in recipe_cfgs}
        if augmentation_mode == 'named_recipes'
        and sample['img'].shape[1] == 1 and recipe_cfgs
        else None)
    grid_report = {'enabled': False}
    if grid_cfg is not None:
        ratio = float(grid_cfg['ratio'])
        mode = int(grid_cfg['mode'])
        stripe_union = 1.0 - (1.0 - ratio) ** 2
        grid_report = {
            'enabled': True,
            'ratio': ratio,
            'prob': float(grid_cfg['prob']),
            'mode': mode,
            'triggered_for_this_sample': bool(zero_mask.any()),
            'expected_masked_fraction_when_triggered': (
                stripe_union if mode == 0 else 1.0 - stripe_union),
            'measured_zero_fraction': float(zero_mask.float().mean()),
            'shared_across_all_temporal_frames_and_views': masks_are_shared,
            'masked_network_value': 0.0,
        }

    visualization_note = (
        'Saved previews only undo normalization for display. Grid-masked '
        'network zeros therefore appear as the normalization mean color '
        'RGB [123.675, 116.28, 103.53], while the model receives zeros.'
        if grid_cfg is not None else
        'The preview shows the one named augmentation selected for this '
        'sample. Its recipe and random parameters are shared by all temporal '
        'frames; different named recipes are not stacked.')

    return {
        'dataset': 'openlane',
        'configured_pipeline': [
            stage['type'] for stage in cfg.train_pipeline],
        'effective_image_pipeline': [
            'LoadTemporalOpenLaneImages',
            'ResizeOpenLaneImages',
            *([
                f'NamedRecipe[{name}]' for name in recipe_names
            ] if augmentation_mode == 'named_recipes' else []),
            'NormalizeOpenLaneImages',
            *(['GridMask'] if grid_cfg is not None else []),
        ],
        'image_augmentation_mode': augmentation_mode,
        'selected_augmentation_recipes': recipe_names,
        'selected_recipe_transforms': {
            name: [
                transform['type']
                for transform in recipe_cfgs[name]['transforms']]
            for name in recipe_names if name in recipe_cfgs
        },
        'single_view_selection_frequency': selection_frequency,
        'shared_across_temporal_frames': True,
        'different_named_augmentations_stacked': False,
        'grid_mask': grid_report,
        'network_tensor': {
            'shape': list(sample['img'].shape),
            'dtype': str(sample['img'].dtype),
            'min': float(sample['img'].min()),
            'max': float(sample['img'].max()),
        },
        'visualization_note': visualization_note,
    }


def visualize_network_input(
        image_path=None, output_root=DEFAULT_OUTPUT_ROOT, seed=0, show=False,
        network_output=None, comparison_output=None):
    """Save selected augmentation and resized-vs-network-input canvases."""
    seed_everything(int(seed))
    cfg = load_config()
    require_openlane_data(cfg)
    dataset = build_sample_dataset(cfg, image_path=image_path)
    sample, sample_index = select_sample(dataset, image_path=image_path)
    source_image = current_image_path(sample)
    output_dir = output_dir_for(source_image, output_root)

    network_path = (
        Path(network_output).expanduser().resolve()
        if network_output else output_dir / '01_network_input.jpg')
    comparison_path = (
        Path(comparison_output).expanduser().resolve()
        if comparison_output
        else output_dir / '02_resized_vs_network_input.jpg')
    network_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)

    network_canvas = make_tensor_canvas(sample['img'])
    network_canvas.save(network_path, quality=95)
    if show:
        network_canvas.show()

    recipe_names = sample['img_metas']['augmentation_recipe_names']
    if len(recipe_names) != sample['img'].shape[1]:
        recipe_names = [
            f'{sample["img_metas"]["image_augmentation_mode"]} view{index}'
            for index in range(sample['img'].shape[1])]
    comparison = torch.cat(
        (_load_resized_originals(sample), sample['img']), dim=1)
    comparison_canvas = make_tensor_canvas(
        comparison,
        row_labels=sample['img_metas']['frame_keys'],
        col_labels=[
            'resized original',
            *[f'network input {name}' for name in recipe_names],
        ])
    comparison_canvas.save(comparison_path, quality=95)

    report = _build_report(cfg, sample)
    report.update({
        'source_image': str(source_image),
        'sample_index': sample_index,
        'output_dir': str(output_dir),
        'outputs': {
            'network_input': str(network_path),
            'resized_vs_network_input': str(comparison_path),
        },
    })
    report_path = output_dir / 'network_input_report.json'
    report['outputs']['report'] = str(report_path)
    write_report(report, report_path)
    return report


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Visualize the actual temporal image tensor produced by the '
            'OpenLane training pipeline.'))
    parser.add_argument(
        'image', nargs='?', default=None,
        help='OpenLane image path. If omitted, a training sample is selected.')
    parser.add_argument(
        '--output-root', default=str(DEFAULT_OUTPUT_ROOT),
        help='Output root; files are grouped by openlane/scene/sample.')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed for sample selection and augmentation.')
    parser.add_argument(
        '--show', action='store_true', help='Open the network-input preview.')
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    visualize_network_input(
        args.image, output_root=args.output_root, seed=args.seed,
        show=args.show)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
