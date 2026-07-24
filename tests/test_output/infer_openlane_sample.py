"""Run LaneLoom on one OpenLane frame and save four prediction-only views.

The temporal images and pose-aligned LiDAR sweeps are loaded through the same
dataset code as evaluation.  Although the requested frame belongs to the
training split, this script replaces its augmentation pipeline with the
deterministic validation pipeline before loading it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import LaneLoom  # noqa: E402,F401 - register project components
from LaneLoom.datasets.openlane.dataset import openlane_collate  # noqa: E402
from mmdet3d.registry import DATASETS, MODELS  # noqa: E402


DEFAULT_IMAGE = (
    REPO_ROOT
    / "data/openlane/images/training"
    / "segment-8603916601243187272_540_000_560_000_with_camera_labels"
    / "155250133427389900.jpg"
)
DEFAULT_CONFIG = REPO_ROOT / "configs/laneloom_openlane_image_lidar.py"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "work_dirs/laneloom_openlane_image_lidar"
    / "exp_date_20260720_220252/checkpoints"
    / "epoch_24.pth"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "vis_output"
LANE_COLORS = (
    "#00ff66",
    "#00d9ff",
    "#ffcc00",
    "#ff4dff",
    "#ff704d",
    "#8cff00",
    "#4d79ff",
    "#ffffff",
)


@dataclass(frozen=True)
class LanePrediction:
    xyz: np.ndarray
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer and visualize one OpenLane sample without GT."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, for example cuda:0 or cpu.",
    )
    return parser.parse_args()


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


def _build_sample(cfg: Config, image_path: Path) -> dict:
    split, label_path = _split_and_label_path(image_path, cfg)
    if not label_path.is_file():
        raise FileNotFoundError(f"OpenLane label not found: {label_path}")

    dataset_cfg = cfg.val_dataloader.dataset.to_dict()
    dataset_cfg["split"] = split
    dataset_cfg["test_mode"] = True
    # A training-split frame must still use deterministic evaluation transforms.
    dataset_cfg["dataset_cfg"]["train_pipeline"] = dataset_cfg["dataset_cfg"][
        "val_pipeline"
    ]
    dataset = DATASETS.build(dataset_cfg)

    target = label_path.resolve()
    try:
        sample_index = next(
            index
            for index, candidate in enumerate(dataset.label_files)
            if Path(candidate).resolve() == target
        )
    except StopIteration as error:
        raise RuntimeError(f"Sample is not present in the configured dataset: {target}") from error
    return openlane_collate([dataset[sample_index]])


def _build_model(cfg: Config, checkpoint: Path, device: torch.device):
    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(checkpoint.resolve()), map_location="cpu")
    model.to(device)
    model.eval()
    if device.type == "cpu":
        _install_cpu_sparse_backbone(model.pts_bev_encoder.backbone)
    return model


class _DenseResult:
    """Minimal SparseConvTensor result interface used by the BEV encoder."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def dense(self) -> torch.Tensor:
        return self.tensor


def _sparse_convolution(
    dense: torch.Tensor,
    mask: torch.Tensor,
    convolution,
    *,
    submanifold: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate one spconv layer at active cells using its checkpoint weights."""
    stride_y, stride_x = (int(value) for value in convolution.stride)
    pad_y, pad_x = (int(value) for value in convolution.padding)
    kernel_y, kernel_x = (int(value) for value in convolution.kernel_size)
    if submanifold:
        output_mask = mask
    else:
        output_mask = F.max_pool2d(
            mask.float(),
            kernel_size=(kernel_y, kernel_x),
            stride=(stride_y, stride_x),
            padding=(pad_y, pad_x),
        ).bool()

    batch_indices, _, output_y, output_x = torch.where(output_mask)
    output_features = dense.new_zeros(
        (len(batch_indices), convolution.out_channels)
    )
    # spconv stores native KRSC weights: [out, kernel_y, kernel_x, in].
    weight = convolution.weight
    input_height, input_width = dense.shape[-2:]
    for kernel_row in range(kernel_y):
        input_y = output_y * stride_y - pad_y + kernel_row
        valid_y = (input_y >= 0) & (input_y < input_height)
        for kernel_column in range(kernel_x):
            input_x = output_x * stride_x - pad_x + kernel_column
            valid = (
                valid_y
                & (input_x >= 0)
                & (input_x < input_width)
            )
            selected = torch.where(valid)[0]
            if selected.numel() == 0:
                continue
            batch = batch_indices[selected]
            y = input_y[selected]
            x = input_x[selected]
            occupied = mask[batch, 0, y, x]
            selected = selected[occupied]
            if selected.numel() == 0:
                continue
            batch = batch_indices[selected]
            y = input_y[selected]
            x = input_x[selected]
            input_features = dense[batch, :, y, x]
            output_features[selected] += input_features @ weight[
                :, kernel_row, kernel_column, :
            ].T
    if convolution.bias is not None:
        output_features += convolution.bias

    output = dense.new_zeros(
        dense.shape[0],
        convolution.out_channels,
        output_mask.shape[-2],
        output_mask.shape[-1],
    )
    output[batch_indices, :, output_y, output_x] = output_features
    return output, output_mask


def _sparse_norm_relu(
    dense: torch.Tensor,
    mask: torch.Tensor,
    norm,
    *,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, _, y, x = torch.where(mask)
    features = dense[batch, :, y, x]
    features = norm(features)
    if residual is not None:
        features = features + residual[batch, :, y, x]
    features = F.relu(features)
    output = torch.zeros_like(dense)
    output[batch, :, y, x] = features
    return output


def _sparse_basic_block(
    dense: torch.Tensor,
    mask: torch.Tensor,
    block,
) -> torch.Tensor:
    residual = dense
    dense, _ = _sparse_convolution(
        dense, mask, block.conv1, submanifold=True
    )
    dense = _sparse_norm_relu(dense, mask, block.norm1)
    dense, _ = _sparse_convolution(
        dense, mask, block.conv2, submanifold=True
    )
    return _sparse_norm_relu(dense, mask, block.norm2, residual=residual)


def _install_cpu_sparse_backbone(backbone) -> None:
    """Replace CUDA-only spconv execution with an equivalent CPU evaluator."""

    def run_stage(dense, mask, stage, *, submanifold):
        dense, mask = _sparse_convolution(
            dense, mask, stage[0], submanifold=submanifold
        )
        dense = _sparse_norm_relu(dense, mask, stage[1])
        dense = _sparse_basic_block(dense, mask, stage[3])
        dense = _sparse_basic_block(dense, mask, stage[4])
        return dense, mask

    def cpu_forward(module, sparse_inputs):
        coordinates = sparse_inputs.indices.long()
        spatial_height, spatial_width = sparse_inputs.spatial_shape
        dense = sparse_inputs.features.new_zeros(
            sparse_inputs.batch_size,
            sparse_inputs.features.shape[1],
            spatial_height,
            spatial_width,
        )
        mask = torch.zeros(
            sparse_inputs.batch_size,
            1,
            spatial_height,
            spatial_width,
            dtype=torch.bool,
            device=dense.device,
        )
        dense[
            coordinates[:, 0],
            :,
            coordinates[:, 1],
            coordinates[:, 2],
        ] = sparse_inputs.features
        mask[
            coordinates[:, 0],
            0,
            coordinates[:, 1],
            coordinates[:, 2],
        ] = True

        dense, mask = run_stage(
            dense, mask, module.stem, submanifold=True
        )
        dense, mask = run_stage(
            dense, mask, module.stage1, submanifold=False
        )
        dense, mask = run_stage(
            dense, mask, module.stage2, submanifold=False
        )

        dense, mask = _sparse_convolution(
            dense, mask, module.mapping[0], submanifold=True
        )
        dense = _sparse_norm_relu(dense, mask, module.mapping[1])
        return _DenseResult(dense)

    backbone.forward = types.MethodType(cpu_forward, backbone)


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
        lanes.append(
            LanePrediction(
                xyz=np.column_stack(
                    (
                        predictions["x"][0, query_index, visible].numpy(),
                        predictions["anchor_y_steps"][visible].numpy(),
                        predictions["z"][0, query_index, visible].numpy(),
                    )
                ),
                score=float(predictions["scores"][0, query_index]),
            )
        )
    return sorted(lanes, key=lambda lane: lane.score, reverse=True)


def _project(points_xyz: np.ndarray, ground_to_raw_img: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points_xyz, np.ones(len(points_xyz))))
    projected = homogeneous @ np.asarray(ground_to_raw_img).T
    uv = projected[:, :2] / projected[:, 2:3]
    invalid = (~np.isfinite(uv).all(axis=1)) | (projected[:, 2] <= 1e-5)
    uv[invalid] = np.nan
    return uv


def _save_overlay(
    image_path: Path,
    lanes: list[LanePrediction],
    ground_to_raw_img: np.ndarray,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for lane_index, lane in enumerate(lanes):
        uv = _project(lane.xyz, ground_to_raw_img)
        inside = (
            np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )
        points = [tuple(point) for point in uv[inside]]
        if len(points) < 2:
            continue
        color = LANE_COLORS[lane_index % len(LANE_COLORS)]
        draw.line(points, fill=color, width=6, joint="curve")
        x, y = points[0]
        draw.text(
            (x + 5, y - 16),
            f"{lane.score:.2f}",
            fill=color,
            stroke_width=2,
            stroke_fill="black",
        )
    image.save(output_path, quality=95)


def _style_axis(axis) -> None:
    axis.set_facecolor("#f5f5f5")
    axis.grid(True, alpha=0.35)
    axis.tick_params(labelsize=9)


def _save_bev(lanes: list[LanePrediction], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 10), dpi=150)
    for lane_index, lane in enumerate(lanes):
        xyz = lane.xyz
        axis.plot(
            xyz[:, 0],
            xyz[:, 1],
            color=LANE_COLORS[lane_index % len(LANE_COLORS)],
            linewidth=2.5,
        )
    axis.set_xlim(-25, 25)
    axis.set_ylim(0, 100)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x / m (left-right)")
    axis.set_ylabel("y / m (forward)")
    axis.set_title(f"LaneLoom prediction — BEV ({len(lanes)} lanes)")
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _save_3d(lanes: list[LanePrediction], output_path: Path) -> None:
    figure = plt.figure(figsize=(9, 7), dpi=150)
    axis = figure.add_subplot(111, projection="3d")
    for lane_index, lane in enumerate(lanes):
        xyz = lane.xyz
        axis.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            color=LANE_COLORS[lane_index % len(LANE_COLORS)],
            linewidth=2.5,
        )
    axis.set_xlim(-25, 25)
    axis.set_ylim(0, 100)
    axis.set_zlim(-5, 5)
    axis.set_xlabel("x / m")
    axis.set_ylabel("y / m")
    axis.set_zlabel("z / m")
    axis.set_title(f"LaneLoom prediction — 3D ({len(lanes)} lanes)")
    axis.view_init(elev=22, azim=-58)
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()
    checkpoint = args.checkpoint.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cfg = Config.fromfile(args.config.resolve())
    device = torch.device(args.device)
    batch = _build_sample(cfg, image_path)
    model = _build_model(cfg, checkpoint, device)
    predictions = _run_inference(model, batch, device)
    lanes = _visible_lanes(predictions)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    original_path = output_dir / "01_original.jpg"
    overlay_path = output_dir / "02_lane_overlay.jpg"
    bev_path = output_dir / "03_bev.png"
    view_3d_path = output_dir / "04_3d.png"

    shutil.copyfile(image_path, original_path)
    current_meta = batch["img_metas"][0]["current"]
    ground_to_raw_img = current_meta["openlane_calib"]["ground_to_raw_img"]
    _save_overlay(image_path, lanes, ground_to_raw_img, overlay_path)
    _save_bev(lanes, bev_path)
    _save_3d(lanes, view_3d_path)

    print(f"checkpoint: {checkpoint}")
    print(f"score threshold: {model.lane_head.score_threshold}")
    print(f"detected lanes: {len(lanes)}")
    for path in (original_path, overlay_path, bev_path, view_3d_path):
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
