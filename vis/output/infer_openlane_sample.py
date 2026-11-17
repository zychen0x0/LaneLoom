"""Infer one OpenLane frame with LaneLoom and save camera, BEV and 3D views."""

from __future__ import annotations

import argparse
import base64
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from matplotlib.lines import Line2D
from PIL import Image


matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
    }
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from LaneLoom.datasets.openlane.dataset import openlane_collate  # noqa: E402
from LaneLoom.engine import register_all_modules  # noqa: E402
from LaneLoom.utils.temporal_sampling import resolve_temporal_sampling  # noqa: E402
from mmdet3d.registry import DATASETS, MODELS  # noqa: E402

register_all_modules()

DEFAULT_IMAGE = (
    REPO_ROOT / "data/openlane/images/validation"
    / "segment-14811410906788672189_373_113_393_113_with_camera_labels"
    / "150724277455060100.jpg"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "work_dirs/laneloom_openlane_image_lidar"
    / "exp_date_20260812_005916/checkpoints"
    / "best_OpenLaneTemporal_F1_epoch_20.pth"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent
PRED_COLOR = "#00ff66"
GT_COLOR = "#ff3b30"
PAPER_PAD_INCHES = 0.2


@dataclass(frozen=True)
class LanePrediction:
    xyz: np.ndarray
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LaneLoom on one OpenLane frame with its matched LiDAR."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--config", type=Path,
        help="Model config; defaults to the config embedded in the checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Output root; files are grouped below "
            "<root>/openlane/<scene>/<frame>."
        ),
    )
    output_group.add_argument(
        "--output-dir",
        type=Path,
        help="Exact output directory (kept for compatibility with older commands).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device, for example cuda:0.",
    )
    parser.add_argument(
        "--score-threshold", type=float,
        help="Override the checkpoint's lane score threshold.",
    )
    return parser.parse_args()


def _sample_output_dir(image_path: Path, output_root: Path) -> Path:
    """Return the standard output directory for one OpenLane sample."""
    return output_root / "openlane" / image_path.parent.name / image_path.stem


def _split_and_label_path(image_path: Path, cfg: Config) -> tuple[str, Path]:
    image_path = image_path.resolve()
    image_root = Path(cfg.dataset_cfg.data_image_dir).resolve()
    try:
        relative = image_path.relative_to(image_root)
    except ValueError as error:
        raise ValueError(f"Image is outside configured image root: {image_path}") from error
    if len(relative.parts) < 3:
        raise ValueError(f"Expected <split>/<scene>/<frame> below image root: {relative}")
    split = relative.parts[0]
    label_relative = relative.with_suffix(".json")
    return split, Path(cfg.dataset_cfg.data_label_dir).resolve() / label_relative


def _inference_dataset_cfg(cfg: Config) -> dict:
    """Translate legacy checkpoint temporal fields without changing sampling."""
    dataset_cfg = cfg.val_dataloader.dataset.to_dict()
    if "temporal_sampling_distance" not in dataset_cfg:
        count = int(dataset_cfg["history_num_frames"])
        distance = float(dataset_cfg["history_distance_m"])
        step = float(dataset_cfg["history_distance_step_m"])
        targets = tuple(distance - i * step for i in range(count))
        sampling = resolve_temporal_sampling(targets if count else False)
        if tuple(dataset_cfg["history_fallback_frame_offsets"]) != sampling.fallback_frame_offsets:
            raise ValueError("Legacy fallback offsets cannot be preserved by distance sampling.")
        if tuple(dataset_cfg["frames"]) != sampling.frame_offsets:
            raise ValueError("Legacy frame queue does not match distance sampling.")
        dataset_cfg["temporal_sampling_distance"] = targets if count else False
    for key in (
        "frames", "history_num_frames", "history_distance_m",
        "history_distance_step_m", "history_fallback_frame_offsets",
        "temporal_gt_augmentation",
    ):
        dataset_cfg.pop(key, None)
    return dataset_cfg


def _build_sample(cfg: Config, image_path: Path) -> dict:
    split, label_path = _split_and_label_path(image_path, cfg)
    if not label_path.is_file():
        raise FileNotFoundError(f"OpenLane label not found: {label_path}")
    dataset_cfg = _inference_dataset_cfg(cfg)
    dataset_cfg["split"] = split
    dataset_cfg["test_mode"] = True
    dataset_cfg["dataset_cfg"]["load_interval"] = 1
    dataset_cfg["dataset_cfg"]["train_pipeline"] = dataset_cfg["dataset_cfg"][
        "val_pipeline"
    ]
    dataset = DATASETS.build(dataset_cfg)
    index = next(
        (i for i, path in enumerate(dataset.label_files)
         if Path(path).resolve() == label_path.resolve()),
        None,
    )
    if index is None:
        raise ValueError(f"Sample not found in dataset: {label_path}")
    return openlane_collate([dataset[index]])


def _build_model(cfg: Config, checkpoint: Path, device: torch.device):
    model_cfg = cfg.model.to_dict()
    encoder = model_cfg.get("pts_bev_encoder")
    if encoder is not None and "queue_length" in encoder:
        distance = _inference_dataset_cfg(cfg)["temporal_sampling_distance"]
        sampling = resolve_temporal_sampling(distance)
        if encoder.pop("queue_length") != sampling.queue_length:
            raise ValueError("Legacy LiDAR queue length does not match dataset sampling.")
        encoder.setdefault("temporal_sampling_distance", distance)
    model = MODELS.build(model_cfg)
    load_checkpoint(model, str(checkpoint), map_location="cpu", strict=True)
    return model.to(device).eval()


def _run_inference(model, batch: dict, device: torch.device) -> dict:
    inputs = {
        "img": batch["img"].to(device),
        "lidar": [points.to(device) for points in batch["lidar"]],
        "img_metas": batch["img_metas"],
    }
    with torch.inference_mode():
        predictions = model(mode="predict", **inputs)
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in predictions.items()
    }


def _visible_lanes(
    predictions: dict, visibility_threshold: float = 0.5
) -> list[LanePrediction]:
    valid = predictions["valid_queries"][0].bool()
    lanes = []
    for query_index in torch.where(valid)[0].tolist():
        visible = predictions["visibility"][0, query_index] >= visibility_threshold
        if int(visible.sum()) < 2:
            continue
        xyz = np.column_stack((
            predictions["x"][0, query_index, visible].numpy(),
            predictions["anchor_y_steps"][visible].numpy(),
            predictions["z"][0, query_index, visible].numpy(),
        ))
        if np.isfinite(xyz).all():
            lanes.append(LanePrediction(xyz, float(predictions["scores"][0, query_index])))
    return sorted(lanes, key=lambda lane: lane.score, reverse=True)


def _ground_truth_lanes(batch: dict) -> list[np.ndarray]:
    return [
        np.asarray(lane)
        for lane in batch["raw_lanes_3d"][0]
        if len(lane) >= 2 and np.isfinite(lane).all()
    ]


def _project(points_xyz: np.ndarray, ground_to_raw_img: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points_xyz, np.ones(len(points_xyz))))
    projected = homogeneous @ np.asarray(ground_to_raw_img).T
    uv = projected[:, :2] / projected[:, 2:3]
    invalid = (~np.isfinite(uv).all(axis=1)) | (projected[:, 2] <= 1e-5)
    uv[invalid] = np.nan
    return uv


def _quadratic_fit_visible_lane(points_xyz: np.ndarray) -> np.ndarray:
    """Fit x(y) and z(y), sampling only inside the lane's visible y range."""
    points_xyz = np.asarray(points_xyz, dtype=float)
    points_xyz = points_xyz[np.isfinite(points_xyz).all(axis=1)]
    if len(points_xyz) < 2:
        return points_xyz
    points_xyz = points_xyz[np.argsort(points_xyz[:, 1])]
    _, unique_indices = np.unique(points_xyz[:, 1], return_index=True)
    points_xyz = points_xyz[np.sort(unique_indices)]
    if len(points_xyz) < 2:
        return points_xyz

    degree = min(2, len(points_xyz) - 1)
    x_of_y = np.poly1d(np.polyfit(points_xyz[:, 1], points_xyz[:, 0], degree))
    z_of_y = np.poly1d(np.polyfit(points_xyz[:, 1], points_xyz[:, 2], degree))
    dense_y = np.linspace(
        points_xyz[0, 1],
        points_xyz[-1, 1],
        5 * len(points_xyz),
    )
    return np.column_stack((x_of_y(dense_y), dense_y, z_of_y(dense_y)))


def _save_camera_lanes(
    image_path: Path,
    lanes: list[np.ndarray],
    ground_to_raw_img: np.ndarray,
    output_path: Path,
    color: str,
) -> None:
    """Save vector lane curves over the original camera image as SVG."""
    with Image.open(image_path) as image:
        width, height = image.size
        image_format = (image.format or image_path.suffix.lstrip(".")).lower()
    mime_type = "image/png" if image_format == "png" else "image/jpeg"
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")

    elements = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        ),
        (
            f'<image width="{width}" height="{height}" '
            f'xlink:href="data:{mime_type};base64,{encoded_image}"/>'
        ),
    ]
    for xyz in lanes:
        uv = _project(xyz, ground_to_raw_img)
        inside = (
            np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )
        valid_indices = np.flatnonzero(inside)
        if len(valid_indices) < 2:
            continue
        # Do not bridge across an invalid or out-of-frame projected point.
        split_at = np.flatnonzero(np.diff(valid_indices) > 1) + 1
        segments = np.split(valid_indices, split_at)
        for indices in segments:
            if len(indices) < 2:
                continue
            points = " ".join(
                f"{uv[index, 0]:.2f},{uv[index, 1]:.2f}" for index in indices
            )
            elements.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                'stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>'
            )
    elements.append("</svg>")
    output_path.write_text(chr(10).join(elements), encoding="utf-8")


def _style_axis(axis) -> None:
    axis.set_facecolor("#f5f5f5")
    axis.grid(True, alpha=0.35)
    axis.tick_params(labelsize=9)


def _save_paper_figure(figure, output_path: Path) -> None:
    """Save one editable SVG suitable for inclusion in a paper."""
    if output_path.suffix.lower() != ".svg":
        raise ValueError(f"Paper figure output must be SVG: {output_path}")
    save_options = {"bbox_inches": "tight", "pad_inches": PAPER_PAD_INCHES}
    figure.savefig(output_path, **save_options)


def _add_lane_legend(axis) -> None:
    handles = [
        Line2D([0], [0], color=GT_COLOR, linewidth=2.5, label="GT"),
        Line2D([0], [0], color=PRED_COLOR, linewidth=2.5, label="Pred"),
    ]
    axis.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#cccccc",
    )


def _save_bev(
    prediction_lanes: list[np.ndarray],
    ground_truth_lanes: list[np.ndarray],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 10))
    for xyz in ground_truth_lanes:
        axis.plot(
            xyz[:, 0],
            xyz[:, 1],
            color=GT_COLOR,
            linewidth=2.5,
            zorder=2,
        )
    for xyz in prediction_lanes:
        axis.plot(
            xyz[:, 0],
            xyz[:, 1],
            color=PRED_COLOR,
            linewidth=2.5,
            zorder=3,
        )
    axis.set_xlim(-25, 25)
    axis.set_ylim(0, 100)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x / m")
    axis.set_ylabel("y / m")
    _style_axis(axis)
    _add_lane_legend(axis)
    figure.tight_layout()
    _save_paper_figure(figure, output_path)
    plt.close(figure)


def _save_3d(
    prediction_lanes: list[np.ndarray],
    ground_truth_lanes: list[np.ndarray],
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    for xyz in ground_truth_lanes:
        axis.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            color=GT_COLOR,
            linewidth=2.5,
            zorder=2,
        )
    for xyz in prediction_lanes:
        axis.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            color=PRED_COLOR,
            linewidth=2.5,
            zorder=3,
        )
    axis.set_xlim(-25, 25)
    axis.set_ylim(0, 100)
    axis.set_zlim(-5, 5)
    axis.set_xlabel("x / m")
    axis.set_ylabel("y / m")
    axis.set_zlabel("z / m", labelpad=-2)
    axis.view_init(elev=22, azim=-58)
    _style_axis(axis)
    _add_lane_legend(axis)
    figure.tight_layout()
    _save_paper_figure(figure, output_path)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()
    checkpoint = args.checkpoint.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if args.config is None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config_text = saved.get("meta", {}).get("cfg")
        if not config_text:
            raise ValueError("Checkpoint has no embedded config; pass --config.")
        cfg = Config.fromstring(config_text, ".py")
        del saved
    else:
        cfg = Config.fromfile(args.config.resolve())

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LaneLoom's sparse LiDAR backbone requires CUDA.")
    if args.score_threshold is not None:
        if not 0 <= args.score_threshold <= 1:
            raise ValueError("score threshold must be between 0 and 1")

    batch = _build_sample(cfg, image_path)
    model = _build_model(cfg, checkpoint, device)
    if args.score_threshold is not None:
        model.lane_head.score_threshold = args.score_threshold
    predictions = _run_inference(model, batch, device)
    lanes = _visible_lanes(predictions)
    prediction_curves = [_quadratic_fit_visible_lane(lane.xyz) for lane in lanes]
    ground_truth_curves = [
        _quadratic_fit_visible_lane(xyz) for xyz in _ground_truth_lanes(batch)
    ]
    ground_to_raw_img = batch["img_metas"][0]["current"]["openlane_calib"]["ground_to_raw_img"]

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _sample_output_dir(image_path, args.output_root.resolve())
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    original_path = output_dir / "01_original.jpg"
    ground_truth_path = output_dir / "02_lane_gt.svg"
    overlay_path = output_dir / "03_lane_overlay.svg"
    bev_path = output_dir / "04_bev.svg"
    view_3d_path = output_dir / "05_3d.svg"

    shutil.copyfile(image_path, original_path)
    _save_camera_lanes(
        image_path,
        ground_truth_curves,
        ground_to_raw_img,
        ground_truth_path,
        GT_COLOR,
    )
    _save_camera_lanes(
        image_path,
        prediction_curves,
        ground_to_raw_img,
        overlay_path,
        PRED_COLOR,
    )
    _save_bev(prediction_curves, ground_truth_curves, bev_path)
    _save_3d(prediction_curves, ground_truth_curves, view_3d_path)

    print(f"checkpoint: {checkpoint}")
    print(f"score threshold: {model.lane_head.score_threshold}")
    print(f"detected lanes: {len(lanes)}")
    for path in (
        original_path,
        ground_truth_path,
        overlay_path,
        bev_path,
        view_3d_path,
    ):
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
