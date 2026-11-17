#!/usr/bin/env python3
"""Export a planar RGB IPM from one OpenLane camera frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
PAPER_DPI = 600
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

def _infer_label_path(image_path: Path) -> Path:
    for image_root in image_path.parents:
        if image_root.name != 'images':
            continue
        relative = image_path.relative_to(image_root).with_suffix('.json')
        candidate = image_root.parent / 'lane3d_1000' / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        'Cannot infer the OpenLane label. Pass it explicitly with --label.')


def _default_output_path(image_path: Path, mode: str) -> Path:
    scene_name = image_path.parent.name
    return (
        REPO_ROOT / 'vis/input/openlane' / scene_name / image_path.stem
        / f'05_ipm_{mode}.png'
    )


def calibrated_homography(label: dict, x_range, y_range, output_size):
    """Return the raw-image to metric-ground-plane homography."""
    from LaneLoom.datasets.openlane.dataset import OpenLaneDataset

    x_min, x_max = x_range
    y_min, y_max = y_range
    width, height = output_size
    if not x_min < x_max or not y_min < y_max:
        raise ValueError('Each metric range must be strictly increasing.')
    if width < 2 or height < 2:
        raise ValueError('Output width and height must both be at least 2.')

    parser = OpenLaneDataset.__new__(OpenLaneDataset)
    ground_to_image = parser._build_calib(label)['ground_to_raw_img']
    ground_plane_to_image = ground_to_image[:, [0, 1, 3]].astype(np.float64)

    x_pixels_per_metre = (width - 1) / (x_max - x_min)
    y_pixels_per_metre = (height - 1) / (y_max - y_min)
    ground_plane_to_bev = np.array([
        [x_pixels_per_metre, 0.0, -x_pixels_per_metre * x_min],
        [0.0, -y_pixels_per_metre, y_pixels_per_metre * y_max],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return ground_plane_to_bev @ np.linalg.inv(ground_plane_to_image)


def manual_homography(src_points, output_size):
    """Return a four-point image-to-rectangle homography."""
    width, height = output_size
    src = np.asarray(src_points, dtype=np.float32).reshape(4, 2)
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Export an OpenLane camera image as a planar RGB IPM.')
    parser.add_argument('image', type=Path)
    parser.add_argument(
        '--mode', choices=('calibrated', 'manual'), default='calibrated')
    parser.add_argument(
        '--label', type=Path,
        help='OpenLane JSON label; inferred from the image path by default.')
    parser.add_argument(
        '--x-range', type=float, nargs=2, default=(-4.5, 4.5),
        metavar=('LEFT_M', 'RIGHT_M'))
    parser.add_argument(
        '--y-range', type=float, nargs=2, default=(10.0, 60.0),
        metavar=('NEAR_M', 'FAR_M'))
    parser.add_argument(
        '--size', type=int, nargs=2, default=(360, 700),
        metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument(
        '--src', type=float, nargs=8,
        metavar=('TL_X', 'TL_Y', 'TR_X', 'TR_Y',
                 'BR_X', 'BR_Y', 'BL_X', 'BL_Y'),
        help='Manual source corners in TL, TR, BR, BL order.')
    parser.add_argument(
        '--dpi', type=int, default=PAPER_DPI,
        help=f'PNG resolution metadata (default: {PAPER_DPI} DPI).')
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.dpi < 1:
        raise ValueError('--dpi must be positive.')
    image_path = args.image.expanduser().resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')

    if args.mode == 'calibrated':
        label_path = (
            args.label.expanduser().resolve()
            if args.label is not None else _infer_label_path(image_path)
        )
        with open(label_path, 'r', encoding='utf-8') as handle:
            label = json.load(handle)
        homography = calibrated_homography(
            label, args.x_range, args.y_range, args.size)
    else:
        if args.src is None:
            raise ValueError('--mode manual requires eight --src coordinates.')
        homography = manual_homography(args.src, args.size)
        print(
            'Warning: manual mode only rectifies the selected quadrilateral; '
            'it is not guaranteed to be a metric BEV.')

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else _default_output_path(image_path, args.mode)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ipm = cv2.warpPerspective(
        image,
        homography,
        tuple(args.size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    rgb_ipm = cv2.cvtColor(ipm, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb_ipm).save(
        output_path, dpi=(args.dpi, args.dpi))

    print(f'IPM: {output_path}')
    print(f'Raster: {args.size[0]} x {args.size[1]} px at {args.dpi} DPI')
    if args.mode == 'calibrated':
        x_res = (args.x_range[1] - args.x_range[0]) / (args.size[0] - 1)
        y_res = (args.y_range[1] - args.y_range[0]) / (args.size[1] - 1)
        print(f'Extent: x={tuple(args.x_range)} m, y={tuple(args.y_range)} m')
        print(f'Pixel scale: x={x_res:.4f} m/px, y={y_res:.4f} m/px')


if __name__ == '__main__':
    main()
