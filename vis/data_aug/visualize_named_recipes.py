#!/usr/bin/env python3
"""Compare all configured OpenLane named augmentations on one image."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmdet3d.registry import DATASETS  # noqa: E402

from vis.data_aug._common import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    load_config,
    make_labeled_grid,
    output_dir_for,
    require_openlane_data,
    resolve_image_path,
    seed_everything,
    write_report,
)


def visualize_named_recipes(
        image_path, output_root=DEFAULT_OUTPUT_ROOT, seed=20260726,
        output=None):
    """Save a labeled grid containing every configured augmentation recipe."""
    image_path = resolve_image_path(image_path)
    seed_everything(int(seed))
    cfg = load_config()
    require_openlane_data(cfg)

    dataset_cfg = copy.deepcopy(dict(cfg.train_dataloader['dataset']))
    dataset_cfg['max_samples'] = 1
    dataset_cfg['dataset_cfg'] = copy.deepcopy(
        dict(dataset_cfg['dataset_cfg']))
    dataset_cfg['dataset_cfg'].update(
        num_views=len(cfg.image_augmentation_recipes),
        image_augmentation_mode='named_recipes',
        image_augmentation_recipes=copy.deepcopy(
            list(cfg.image_augmentation_recipes)),
    )
    dataset = DATASETS.build(dataset_cfg)
    recipe_seeds = [
        int(seed) + view_index for view_index in range(dataset.num_views)]
    view_tensors, _ = dataset._load_image_views(
        image_path, augmentation_recipe_seeds=recipe_seeds)

    mean = np.asarray(cfg.img_norm_cfg['mean'], dtype=np.float32)
    std = np.asarray(cfg.img_norm_cfg['std'], dtype=np.float32)
    view_arrays = view_tensors.permute(0, 2, 3, 1).cpu().numpy()
    view_arrays = np.clip(
        view_arrays * std[None, None, None, :]
        + mean[None, None, None, :], 0, 255).astype(np.uint8)
    panels = [
        (name, Image.fromarray(view_arrays[view_index]))
        for view_index, name in enumerate(dataset.augmentation_recipe_names)]
    canvas = make_labeled_grid(panels)

    output_dir = output_dir_for(image_path, output_root)
    output_path = (
        Path(output).expanduser().resolve()
        if output else output_dir / '03_named_recipes.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)

    report = {
        'dataset': 'openlane',
        'source_image': str(image_path),
        'image_augmentation_mode': 'named_recipes',
        'augmentation_recipe_seeds': recipe_seeds,
        'recipes': {
            recipe_cfg['name']: [
                transform_cfg['type']
                for transform_cfg in recipe_cfg['transforms']]
            for recipe_cfg in cfg.image_augmentation_recipes
        },
        'output_dir': str(output_dir),
        'outputs': {'named_recipes': str(output_path)},
    }
    report_path = output_dir / 'named_recipes_report.json'
    report['outputs']['report'] = str(report_path)
    write_report(report, report_path)
    return report


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Apply every configured OpenLane named augmentation to one image '
            'and save a labeled comparison grid.'))
    parser.add_argument('image', help='Path to one OpenLane source image.')
    parser.add_argument(
        '--output-root', default=str(DEFAULT_OUTPUT_ROOT),
        help='Output root; files are grouped by openlane/scene/sample.')
    parser.add_argument(
        '--seed', type=int, default=20260726,
        help='Base seed; each recipe receives seed + recipe index.')
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    visualize_named_recipes(
        args.image, output_root=args.output_root, seed=args.seed)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
