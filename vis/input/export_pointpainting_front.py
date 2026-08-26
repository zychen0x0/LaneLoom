#!/usr/bin/env python3
"""Export publication-ready PointPainting before/after front-view figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData


CANVAS_SIZE = (6000, 2100)
BACKGROUND = (255, 255, 255)
UNPAINTED_COLOR = (43, 50, 53)


def load_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load finite XYZ points and their PointPainting RGB values."""
    vertices = PlyData.read(path)["vertex"].data
    required = {"x", "y", "z", "red", "green", "blue"}
    available = set(vertices.dtype.names or ())
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Missing PLY vertex properties: {', '.join(missing)}")

    xyz = np.column_stack(
        (vertices["x"], vertices["y"], vertices["z"])
    ).astype(np.float64)
    colors = np.column_stack(
        (vertices["red"], vertices["green"], vertices["blue"])
    ).astype(np.uint8)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    colors = colors[finite]
    if xyz.size == 0:
        raise ValueError(f"No finite XYZ points found in {path}")
    return xyz, colors


def project_vehicle_front(
    xyz: np.ndarray,
    camera_height: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ground-frame XYZ from a virtual camera facing vehicle-forward Y.

    The input coordinate convention remains x=lateral, y=forward and z=up.
    Only points in front of the vehicle are retained.
    """
    depth = xyz[:, 1]
    visible = np.isfinite(xyz).all(axis=1) & (depth > 1e-4)
    xyz = xyz[visible]
    depth = depth[visible]
    projected = np.column_stack(
        (xyz[:, 0] / depth, -(xyz[:, 2] - camera_height) / depth)
    )
    return projected, visible


def enhance_painted_colors(colors: np.ndarray) -> np.ndarray:
    """Apply a restrained global enhancement for print reproduction."""
    rgb = colors.astype(np.float64) / 255.0
    # Slight gamma lift preserves painted lane markings after print/downsampling.
    rgb = np.power(rgb, 0.90)
    luminance = (
        0.2126 * rgb[:, 0]
        + 0.7152 * rgb[:, 1]
        + 0.0722 * rgb[:, 2]
    )[:, None]
    rgb = luminance + 1.65 * (rgb - luminance)
    rgb = 0.5 + 1.15 * (rgb - 0.5)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _pixel_coordinates(
    projected: np.ndarray,
    canvas_size: tuple[int, int],
    fill: float,
    fit_width: bool = False,
    bottom_position: float | None = None,
) -> tuple[np.ndarray, float, tuple[float, float, float, float]]:
    width, height = canvas_size
    lower = projected.min(axis=0)
    upper = projected.max(axis=0)
    span = np.maximum(upper - lower, np.finfo(np.float64).eps)
    if fit_width:
        scale = width * fill / span[0]
    else:
        scale = min(width * fill / span[0], height * fill / span[1])
    center = (lower + upper) / 2.0
    pixels = np.empty_like(projected)
    pixels[:, 0] = width / 2.0 + (projected[:, 0] - center[0]) * scale
    projected_height = span[1] * scale
    if bottom_position is not None:
        pixels[:, 1] = (
            height * bottom_position + (projected[:, 1] - upper[1]) * scale
        )
    elif fit_width and projected_height > height * fill:
        # Preserve the road/near-field bottom and crop sparse high points first.
        bottom = height * (1.0 + fill) / 2.0
        pixels[:, 1] = bottom + (projected[:, 1] - upper[1]) * scale
    else:
        pixels[:, 1] = height / 2.0 + (projected[:, 1] - center[1]) * scale
    bounds = (lower[0], upper[0], lower[1], upper[1])
    return pixels, scale, bounds


def render_front_view(
    projected: np.ndarray,
    depth: np.ndarray,
    colors: np.ndarray,
    *,
    fill: float = 0.80,
    point_diameter: float = 14.4,
    supersampling: int = 2,
    canvas_size: tuple[int, int] = CANVAS_SIZE,
    fit_width: bool = False,
    bottom_position: float | None = None,
) -> tuple[Image.Image, float, tuple[float, float, float, float]]:
    """Render far-to-near points with depth-aware, print-friendly diameters."""
    pixels, scale, bounds = _pixel_coordinates(
        projected,
        canvas_size,
        fill,
        fit_width=fit_width,
        bottom_position=bottom_position,
    )
    render_size = tuple(dimension * supersampling for dimension in canvas_size)
    canvas = Image.new("RGB", render_size, BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    # Drawing far points first makes nearer surfaces visually coherent.
    order = np.argsort(depth)[::-1]
    reference_depth = 30.0
    diameters = point_diameter * np.sqrt(reference_depth / depth)
    diameters = np.clip(diameters, point_diameter * 0.62, point_diameter * 2.0)
    for index in order:
        x_pixel, y_pixel = pixels[index] * supersampling
        radius = diameters[index] * supersampling / 2.0
        color = tuple(int(channel) for channel in colors[index])
        draw.ellipse(
            (
                x_pixel - radius,
                y_pixel - radius,
                x_pixel + radius,
                y_pixel + radius,
            ),
            fill=color,
        )

    if supersampling > 1:
        canvas = canvas.resize(canvas_size, Image.Resampling.LANCZOS)
    return canvas, scale, bounds


def save_png(image: Image.Image, path: Path, dpi: int = 600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", dpi=(dpi, dpi), optimize=True)


def export_comparison(
    input_path: Path,
    output_dir: Path,
    *,
    fill: float = 0.80,
    point_diameter: float = 14.4,
    camera_height: float = 1.5,
    canvas_size: tuple[int, int] = CANVAS_SIZE,
    raw_only: bool = False,
    fit_width: bool = False,
    bottom_position: float | None = None,
) -> list[Path]:
    xyz, painted_colors = load_points(input_path)
    projected, visible = project_vehicle_front(xyz, camera_height=camera_height)
    xyz = xyz[visible]
    raw_painted_colors = painted_colors[visible]
    unpainted_colors = np.broadcast_to(
        np.asarray(UNPAINTED_COLOR, dtype=np.uint8), raw_painted_colors.shape
    )

    before, scale, bounds = render_front_view(
        projected,
        xyz[:, 1],
        unpainted_colors,
        fill=fill,
        point_diameter=point_diameter,
        canvas_size=canvas_size,
        fit_width=fit_width,
        bottom_position=bottom_position,
    )
    after_raw, _, _ = render_front_view(
        projected,
        xyz[:, 1],
        raw_painted_colors,
        fill=fill,
        point_diameter=point_diameter,
        canvas_size=canvas_size,
        fit_width=fit_width,
        bottom_position=bottom_position,
    )

    frame_id = input_path.parent.name
    before_path = output_dir / f"{frame_id}_pointpainting_before.png"
    after_raw_path = output_dir / f"{frame_id}_pointpainting_after_raw_rgb.png"
    save_png(before, before_path)
    save_png(after_raw, after_raw_path)

    output_paths = [before_path, after_raw_path]
    if not raw_only:
        enhanced_painted_colors = enhance_painted_colors(raw_painted_colors)
        after, _, _ = render_front_view(
            projected,
            xyz[:, 1],
            enhanced_painted_colors,
            fill=fill,
            point_diameter=point_diameter,
            canvas_size=canvas_size,
            fit_width=fit_width,
            bottom_position=bottom_position,
        )
        after_path = output_dir / f"{frame_id}_pointpainting_after.png"
        comparison_path = output_dir / f"{frame_id}_pointpainting_comparison.png"
        raw_comparison_path = (
            output_dir / f"{frame_id}_pointpainting_comparison_raw_rgb.png"
        )
        save_png(after, after_path)

        comparison_gap = round(canvas_size[0] * 0.03)
        comparison = Image.new(
            "RGB",
            (canvas_size[0] * 2 + comparison_gap, canvas_size[1]),
            BACKGROUND,
        )
        comparison.paste(before, (0, 0))
        comparison.paste(after, (canvas_size[0] + comparison_gap, 0))
        save_png(comparison, comparison_path)

        raw_comparison = Image.new(
            "RGB",
            (canvas_size[0] * 2 + comparison_gap, canvas_size[1]),
            BACKGROUND,
        )
        raw_comparison.paste(before, (0, 0))
        raw_comparison.paste(
            after_raw, (canvas_size[0] + comparison_gap, 0)
        )
        save_png(raw_comparison, raw_comparison_path)
        output_paths.extend([after_path, comparison_path, raw_comparison_path])

    x_min, x_max, v_min, v_max = bounds
    print(
        f"{input_path}: {len(xyz)} visible points, "
        f"front projection=[{x_min:.4f}, {x_max:.4f}] x "
        f"[{v_min:.4f}, {v_max:.4f}], scale={scale:.1f} px"
    )
    for path in output_paths:
        print(path)
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Painted input PLY files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fill", type=float, default=0.80)
    parser.add_argument("--point-diameter", type=float, default=14.4)
    parser.add_argument("--camera-height", type=float, default=1.5)
    parser.add_argument("--canvas-width", type=int, default=CANVAS_SIZE[0])
    parser.add_argument("--canvas-height", type=int, default=CANVAS_SIZE[1])
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Export only the gray before image and unenhanced RGB after image.",
    )
    parser.add_argument(
        "--fit-width",
        action="store_true",
        help="Keep horizontal occupancy fixed and crop excess high points.",
    )
    parser.add_argument(
        "--bottom-position",
        type=float,
        help="Place the lowest projected point at this fraction of image height.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.fill <= 1.0:
        raise ValueError("--fill must be in (0, 1]")
    if args.point_diameter <= 0.0:
        raise ValueError("--point-diameter must be positive")
    if args.canvas_width <= 0 or args.canvas_height <= 0:
        raise ValueError("Canvas dimensions must be positive")
    if args.bottom_position is not None and not 0.0 < args.bottom_position < 1.0:
        raise ValueError("--bottom-position must be in (0, 1)")
    for input_path in args.inputs:
        export_comparison(
            input_path.expanduser().resolve(),
            args.output_dir.expanduser().resolve(),
            fill=args.fill,
            point_diameter=args.point_diameter,
            camera_height=args.camera_height,
            canvas_size=(args.canvas_width, args.canvas_height),
            raw_only=args.raw_only,
            fit_width=args.fit_width,
            bottom_position=args.bottom_position,
        )


if __name__ == "__main__":
    main()
