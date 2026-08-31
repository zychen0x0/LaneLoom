#!/usr/bin/env python3
"""Rasterize OpenLane UV annotations as a binary lane mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_ROOT = REPO_ROOT / 'data/openlane/images'
DEFAULT_LABEL_ROOT = REPO_ROOT / 'data/openlane/lane3d_1000'
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / 'openlane'


def _resolve_label(image_path: Path, label_path: Path | None) -> Path:
    if label_path is not None:
        if not label_path.is_file():
            raise FileNotFoundError(f'OpenLane label does not exist: {label_path}')
        return label_path

    try:
        relative_path = image_path.relative_to(DEFAULT_IMAGE_ROOT)
    except ValueError as error:
        raise ValueError(
            f'Image must be below {DEFAULT_IMAGE_ROOT}, or --label must be '
            'provided explicitly.') from error
    candidate = (DEFAULT_LABEL_ROOT / relative_path).with_suffix('.json')
    if not candidate.is_file():
        raise FileNotFoundError(f'OpenLane label does not exist: {candidate}')
    return candidate


def _parse_lane_uv_points(lane: dict) -> np.ndarray | None:
    uv = np.asarray(lane.get('uv', []), dtype=np.float64)
    if uv.ndim != 2:
        return None
    if uv.shape[0] == 2:
        uv = uv.T
    if uv.shape[1:] != (2,):
        return None
    return uv


def rasterize_uv_mask(
        label: dict, source_size: tuple[int, int], output_size: tuple[int, int],
        line_width: int = 5) -> Image.Image:
    """Return a strict 0/255 mask from raw ``lane_lines[].uv`` coordinates."""
    if line_width < 1:
        raise ValueError('line_width must be at least 1 pixel.')
    source_width, source_height = source_size
    output_width, output_height = output_size
    if min(source_width, source_height, output_width, output_height) < 1:
        raise ValueError('Image dimensions must be positive.')

    scale = np.array(
        [output_width / source_width, output_height / source_height],
        dtype=np.float64)
    mask = Image.new('L', output_size, 0)
    draw = ImageDraw.Draw(mask)
    for lane in label.get('lane_lines', []):
        uv = _parse_lane_uv_points(lane)
        if uv is None:
            continue
        uv = uv * scale
        finite = np.isfinite(uv).all(axis=1)
        start = 0
        while start < len(uv):
            while start < len(uv) and not finite[start]:
                start += 1
            end = start
            while end < len(uv) and finite[end]:
                end += 1
            if end - start >= 2:
                points = [tuple(point) for point in uv[start:end]]
                draw.line(points, fill=255, width=line_width, joint='curve')
            elif end - start == 1:
                x, y = uv[start]
                radius = line_width / 2
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius), fill=255)
            start = end
    return mask


def export_uv_gt_mask(
        image_path: Path, label_path: Path | None = None,
        output_path: Path | None = None, output_size: tuple[int, int] = (960, 640),
        line_width: int = 5) -> Path:
    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f'OpenLane image does not exist: {image_path}')
    label_path = _resolve_label(
        image_path,
        None if label_path is None else label_path.expanduser().resolve())
    with Image.open(image_path) as image:
        source_size = image.size
    with open(label_path, 'r', encoding='utf-8') as handle:
        label = json.load(handle)

    if output_path is None:
        output_path = DEFAULT_OUTPUT_ROOT / f'{image_path.stem}_uv_gt_mask.png'
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask = rasterize_uv_mask(
        label, source_size, output_size, line_width=line_width)
    mask.save(output_path)
    return output_path


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Export an OpenLane raw-UV GT binary lane mask.')
    parser.add_argument('image', type=Path, help='OpenLane source image.')
    parser.add_argument(
        '--label', type=Path,
        help='Matching JSON label. Inferred for images under data/openlane/images.')
    parser.add_argument(
        '--output', type=Path,
        help='Output PNG (default: vis/input/openlane/<frame>_uv_gt_mask.png).')
    parser.add_argument(
        '--size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
        default=(960, 640), help='Output mask size (default: 960 640).')
    parser.add_argument(
        '--line-width', type=int, default=5,
        help='Lane rasterization width in output pixels (default: 5).')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    output_path = export_uv_gt_mask(
        args.image,
        label_path=args.label,
        output_path=args.output,
        output_size=tuple(args.size),
        line_width=args.line_width)
    print(output_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
