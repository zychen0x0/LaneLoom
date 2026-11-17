#!/usr/bin/env python3
"""Export a magma-colored relative depth map with Depth Anything V2 Small."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL_ID = 'depth-anything/Depth-Anything-V2-Small-hf'
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / 'openlane'
PAPER_DPI = 600


def _colorize_depth(
        depth: np.ndarray, colormap: str = 'magma',
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0) -> tuple[np.ndarray, float, float]:
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError(
            'Percentiles must satisfy 0 <= lower < upper <= 100.')
    finite = np.isfinite(depth)
    if not finite.any():
        raise ValueError('Depth prediction contains no finite values.')
    lower, upper = np.percentile(
        depth[finite], (lower_percentile, upper_percentile))
    normalized = np.clip(
        (depth - lower) / max(float(upper - lower), 1e-8), 0.0, 1.0)
    rgb = (
        matplotlib.colormaps[colormap](normalized)[..., :3] * 255
    ).round().astype(np.uint8)
    return rgb, float(lower), float(upper)


def export_depth(
        image_path: Path, output_path: Path | None = None,
        colormap: str = 'magma',
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
        dpi: int = PAPER_DPI) -> dict:
    if dpi < 1:
        raise ValueError('dpi must be positive.')
    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f'Input image does not exist: {image_path}')
    if output_path is None:
        output_path = (
            DEFAULT_OUTPUT_ROOT
            / f'{image_path.stem}_depth_anything_v2_small_{colormap}.png')
    output_path = output_path.expanduser().resolve()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = Image.open(image_path).convert('RGB')
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    model = model.to(device).eval()
    inputs = processor(images=image, return_tensors='pt').to(device)
    with torch.inference_mode():
        prediction = model(**inputs)
    postprocessed = processor.post_process_depth_estimation(
        prediction, target_sizes=[(image.height, image.width)])
    depth = postprocessed[0]['predicted_depth'].float().cpu().numpy()
    rgb, display_min, display_max = _colorize_depth(
        depth,
        colormap=colormap,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output_path, dpi=(dpi, dpi))
    finite = np.isfinite(depth)
    return {
        'model': MODEL_ID,
        'device': str(device),
        'source': str(image_path),
        'output': str(output_path),
        'size': [image.width, image.height],
        'dpi': dpi,
        'raw_min': float(depth[finite].min()),
        'raw_max': float(depth[finite].max()),
        'display_min': display_min,
        'display_max': display_max,
        'colormap': colormap,
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Export Depth Anything V2 Small relative depth as RGB.')
    parser.add_argument('image', type=Path, help='Source RGB image.')
    parser.add_argument('--output', type=Path, help='Output RGB PNG path.')
    parser.add_argument(
        '--colormap', default='magma',
        choices=tuple(matplotlib.colormaps),
        help='Matplotlib colormap (default: magma).')
    parser.add_argument(
        '--percentiles', type=float, nargs=2, default=(1.0, 99.0),
        metavar=('LOWER', 'UPPER'),
        help='Display normalization percentiles (default: 1 99).')
    parser.add_argument(
        '--dpi', type=int, default=PAPER_DPI,
        help=f'PNG resolution metadata (default: {PAPER_DPI} DPI).')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    report = export_depth(
        args.image,
        output_path=args.output,
        colormap=args.colormap,
        lower_percentile=args.percentiles[0],
        upper_percentile=args.percentiles[1],
        dpi=args.dpi)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
