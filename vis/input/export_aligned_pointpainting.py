#!/usr/bin/env python3
"""Align painted OpenLane sweeps to the current frame and render one figure."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from export_pointpainting_front import (
    load_points,
    project_vehicle_front,
    render_front_view,
    save_png,
)


WAYMO_VEHICLE_TO_GROUND_AXES = np.asarray(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DEFAULT_POINT_CLOUD_RANGE = (-25.0, 0.0, -5.0, 25.0, 100.0, 5.0)


def align_ground_points_to_current(
    points_ground: np.ndarray,
    source_vehicle_to_global: np.ndarray,
    current_vehicle_to_global: np.ndarray,
) -> np.ndarray:
    """Apply the dataset's source-ground -> global -> current-ground transform."""
    ground_from_vehicle = np.eye(4, dtype=np.float64)
    ground_from_vehicle[:3, :3] = WAYMO_VEHICLE_TO_GROUND_AXES
    source_ground_to_current_ground = (
        ground_from_vehicle
        @ np.linalg.inv(np.asarray(current_vehicle_to_global, dtype=np.float64))
        @ np.asarray(source_vehicle_to_global, dtype=np.float64)
        @ np.linalg.inv(ground_from_vehicle)
    )
    points_h = np.column_stack(
        (np.asarray(points_ground, dtype=np.float64), np.ones(len(points_ground)))
    )
    return (source_ground_to_current_ground @ points_h.T).T[:, :3]


def load_vehicle_poses(
    pose_info_path: Path,
    frame_ids: list[str],
) -> dict[str, np.ndarray]:
    with open(pose_info_path, "rb") as handle:
        frame_infos = pickle.load(handle)
    poses = {}
    requested = set(frame_ids)
    for frame_info in frame_infos:
        timestamp = frame_info.get("metadata", {}).get("timestamp_micros")
        pose = frame_info.get("pose")
        if timestamp is None or pose is None:
            continue
        frame_id = str(timestamp)[:12]
        if frame_id in requested:
            poses[frame_id] = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    missing = sorted(requested - poses.keys())
    if missing:
        raise KeyError(f"Pose info is missing frames: {', '.join(missing)}")
    return poses


def filter_point_cloud_range(
    xyz: np.ndarray,
    point_cloud_range: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    return (
        (xyz[:, 0] >= x_min)
        & (xyz[:, 0] <= x_max)
        & (xyz[:, 1] >= y_min)
        & (xyz[:, 1] <= y_max)
        & (xyz[:, 2] >= z_min)
        & (xyz[:, 2] <= z_max)
    )


def export_aligned_pointpainting(
    input_paths: list[Path],
    pose_info_path: Path,
    current_frame_id: str,
    output_path: Path,
    *,
    point_cloud_range: tuple[
        float, float, float, float, float, float
    ] = DEFAULT_POINT_CLOUD_RANGE,
    canvas_size: tuple[int, int] = (5400, 3600),
    fill: float = 0.92,
    point_diameter: float = 16.6,
    camera_height: float = 1.5,
    bottom_position: float = 0.86,
) -> Path:
    frame_ids = [path.parent.name for path in input_paths]
    if current_frame_id not in frame_ids:
        raise ValueError("Current frame must be one of the input PLY frame IDs")
    poses = load_vehicle_poses(pose_info_path, frame_ids)
    current_pose = poses[current_frame_id]

    loaded_frames = []
    current_near_plane = None
    for input_path, frame_id in zip(input_paths, frame_ids):
        xyz, colors = load_points(input_path)
        loaded_frames.append((frame_id, xyz, colors))
        if frame_id == current_frame_id:
            current_near_plane = float(xyz[:, 1].min())
    if current_near_plane is None:
        raise RuntimeError("Could not derive the current-frame near plane")
    effective_range = list(point_cloud_range)
    effective_range[1] = max(effective_range[1], current_near_plane)
    effective_range = tuple(effective_range)
    print(
        f"Front-view near plane={current_near_plane:.3f} m "
        "(derived from the current painted PLY)"
    )

    aligned_frames = []
    color_frames = []
    for frame_id, xyz, colors in loaded_frames:
        aligned = align_ground_points_to_current(
            xyz, poses[frame_id], current_pose
        )
        in_range = filter_point_cloud_range(aligned, effective_range)
        aligned_frames.append(aligned[in_range])
        color_frames.append(colors[in_range])
        relative_pose = np.linalg.inv(current_pose) @ poses[frame_id]
        distance = float(np.linalg.norm(relative_pose[:2, 3]))
        print(
            f"{frame_id}: {len(xyz)} painted points, {in_range.sum()} retained, "
            f"pose distance to current={distance:.3f} m"
        )

    xyz = np.concatenate(aligned_frames, axis=0)
    colors = np.concatenate(color_frames, axis=0)
    projected, visible = project_vehicle_front(xyz, camera_height=camera_height)
    xyz = xyz[visible]
    colors = colors[visible]
    image, scale, _ = render_front_view(
        projected,
        xyz[:, 1],
        colors,
        fill=fill,
        point_diameter=point_diameter,
        canvas_size=canvas_size,
        fit_width=True,
        bottom_position=bottom_position,
    )
    save_png(image, output_path, dpi=600)
    print(
        f"{output_path}: {len(xyz)} aligned visible points, "
        f"scale={scale:.1f} px, canvas={canvas_size[0]}x{canvas_size[1]}, "
        "dpi=600"
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--pose-info", required=True, type=Path)
    parser.add_argument("--current-frame-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--canvas-width", type=int, default=5400)
    parser.add_argument("--canvas-height", type=int, default=3600)
    parser.add_argument("--fill", type=float, default=0.92)
    parser.add_argument("--point-diameter", type=float, default=16.6)
    parser.add_argument("--camera-height", type=float, default=1.5)
    parser.add_argument("--bottom-position", type=float, default=0.86)
    parser.add_argument(
        "--point-cloud-range",
        type=float,
        nargs=6,
        default=DEFAULT_POINT_CLOUD_RANGE,
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_aligned_pointpainting(
        [path.expanduser().resolve() for path in args.inputs],
        args.pose_info.expanduser().resolve(),
        args.current_frame_id,
        args.output.expanduser().resolve(),
        point_cloud_range=tuple(args.point_cloud_range),
        canvas_size=(args.canvas_width, args.canvas_height),
        fill=args.fill,
        point_diameter=args.point_diameter,
        camera_height=args.camera_height,
        bottom_position=args.bottom_position,
    )


if __name__ == "__main__":
    main()
