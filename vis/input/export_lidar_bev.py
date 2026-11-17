#!/usr/bin/env python3
"""Export publication-ready, top-down BEV images from PLY point clouds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData


CANVAS_SIZE = (1920, 1280)
BACKGROUND = (255, 255, 255)
POINT_COLOR = (52, 59, 62)


def load_xy(path: Path) -> np.ndarray:
    vertices = PlyData.read(path)["vertex"].data
    xy = np.column_stack((vertices["x"], vertices["y"])).astype(np.float64)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if xy.size == 0:
        raise ValueError(f"No finite xy points found in {path}")
    return xy


def render_bev(
    xy: np.ndarray,
    output_path: Path,
    height_fill: float = 0.90,
    point_diameter: float = 3.6,
    supersampling: int = 2,
) -> tuple[float, tuple[float, float, float, float]]:
    """Render x laterally and y forward/up without rotating the point cloud."""
    width, height = CANVAS_SIZE
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    x_span = max(x_max - x_min, np.finfo(np.float64).eps)
    y_span = max(y_max - y_min, np.finfo(np.float64).eps)

    # Prefer the requested vertical occupancy, but still guarantee that a wide
    # cloud cannot be clipped by the fixed 3:2 canvas.
    scale = min(
        height * height_fill / y_span,
        width * height_fill / x_span,
    )
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    render_size = (width * supersampling, height * supersampling)
    canvas = Image.new("RGB", render_size, BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    px = (width / 2.0 + (xy[:, 0] - center_x) * scale) * supersampling
    # Image coordinates grow downward, so increasing forward y maps upward.
    py = (height / 2.0 - (xy[:, 1] - center_y) * scale) * supersampling
    radius = point_diameter * supersampling / 2.0
    for x_pixel, y_pixel in zip(px, py):
        draw.ellipse(
            (
                x_pixel - radius,
                y_pixel - radius,
                x_pixel + radius,
                y_pixel + radius,
            ),
            fill=POINT_COLOR,
        )

    if supersampling > 1:
        canvas = canvas.resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", dpi=(300, 300), optimize=True)
    return scale, (x_min, x_max, y_min, y_max)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Input PLY files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height-fill", type=float, default=0.90)
    parser.add_argument("--point-diameter", type=float, default=3.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.height_fill <= 1.0:
        raise ValueError("--height-fill must be in (0, 1]")
    for input_path in args.inputs:
        input_path = input_path.resolve()
        frame_id = input_path.parent.name
        output_path = args.output_dir / f"{frame_id}_lidar_bev.png"
        xy = load_xy(input_path)
        scale, bounds = render_bev(
            xy,
            output_path,
            height_fill=args.height_fill,
            point_diameter=args.point_diameter,
        )
        x_min, x_max, y_min, y_max = bounds
        print(
            f"{output_path}: {len(xy)} points, "
            f"x=[{x_min:.2f}, {x_max:.2f}] m, "
            f"y=[{y_min:.2f}, {y_max:.2f}] m, "
            f"scale={scale:.2f} px/m"
        )


if __name__ == "__main__":
    main()
