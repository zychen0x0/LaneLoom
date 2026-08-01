"""Visualize one real disconnected temporal-enhanced GT lane."""

import argparse
import json
from pathlib import Path

import numpy as np
from mmengine.config import Config
from mmdet3d.registry import DATASETS
from PIL import Image, ImageDraw, ImageFont

import LaneLoom.datasets.openlane.dataset  # noqa: F401


def _network_image(sample):
    image = sample["img"][-1, 0].detach().permute(1, 2, 0).cpu().numpy()
    image *= np.array([58.395, 57.12, 57.375], dtype=np.float32)
    image += np.array([123.675, 116.28, 103.53], dtype=np.float32)
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))


def _project(lane, lane2img):
    homogeneous = np.concatenate(
        [lane, np.ones((lane.shape[0], 1), dtype=np.float32)], axis=1)
    projected = (lane2img @ homogeneous.T).T
    depth = projected[:, 2]
    uv = np.full((lane.shape[0], 2), np.nan, dtype=np.float32)
    positive = depth > 1e-5
    uv[positive] = projected[positive, :2] / depth[positive, None]
    return uv, positive


def _runs(mask):
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    return [run for run in np.split(indices, splits) if run.size]


def _draw_lane_runs(draw, uv, mask, color, width=3, radius=3):
    valid_uv = np.isfinite(uv).all(axis=1)
    for run in _runs(mask & valid_uv):
        points = [tuple(np.round(uv[index]).astype(int)) for index in run]
        if len(points) >= 2:
            draw.line(points, fill=color, width=width)
        for x, y in points:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color)


def _draw_dashed(draw, start, end, color, width=3, pieces=10):
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    for piece in range(0, pieces, 2):
        a = start + (end - start) * (piece / pieces)
        b = start + (end - start) * ((piece + 1) / pieces)
        draw.line(
            (tuple(np.round(a).astype(int)),
             tuple(np.round(b).astype(int))),
            fill=color, width=width)


def _title(panel, text):
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), text, font=font)
    draw.rectangle(
        (8, 8, box[2] + 20, box[3] + 20),
        fill=(0, 0, 0))
    draw.text((14, 14), text, fill=(255, 255, 255), font=font)


def _camera_panels(sample, lane_index):
    base = _network_image(sample)
    lane2img = np.asarray(
        sample["img_metas"]["current"]["lane2img"][0],
        dtype=np.float32)
    current_lanes = sample["gt_lanes_3d"].numpy()
    current_visibility = sample["gt_lanes_visibility"].numpy() > 0.5
    enhanced_lanes = sample["gt_lanes_3d_enhanced"].numpy()
    enhanced_visibility = (
        sample["gt_lanes_visibility_enhanced"].numpy() > 0.5)

    current_panel = base.copy()
    current_draw = ImageDraw.Draw(current_panel)
    for lane, visibility in zip(current_lanes, current_visibility):
        uv, positive = _project(lane, lane2img)
        _draw_lane_runs(
            current_draw, uv, visibility & positive,
            (40, 255, 80), width=2, radius=2)
    _title(current_panel, "Current-frame GT (green)")

    enhanced_panel = base.copy()
    enhanced_draw = ImageDraw.Draw(enhanced_panel)
    for index, (lane, visibility) in enumerate(
            zip(enhanced_lanes, enhanced_visibility)):
        uv, positive = _project(lane, lane2img)
        color = (255, 60, 220) if index == lane_index else (80, 190, 255)
        _draw_lane_runs(
            enhanced_draw, uv, visibility & positive,
            color, width=4 if index == lane_index else 2,
            radius=4 if index == lane_index else 2)

    target_uv, target_positive = _project(
        enhanced_lanes[lane_index], lane2img)
    target_runs = _runs(
        enhanced_visibility[lane_index] & target_positive
        & np.isfinite(target_uv).all(axis=1))
    if len(target_runs) >= 2:
        first_end = target_uv[target_runs[0][-1]]
        second_start = target_uv[target_runs[1][0]]
        _draw_dashed(
            enhanced_draw, first_end, second_start,
            (255, 40, 30), width=4)
        midpoint = np.round((first_end + second_start) / 2).astype(int)
        enhanced_draw.text(
            tuple(midpoint), "visibility gap",
            fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    _title(
        enhanced_panel,
        "Temporal-enhanced GT (cyan); disconnected target (magenta)")
    return current_panel, enhanced_panel


def _bev_panel(sample, lane_index, width, height=460):
    panel = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(panel)
    margin = 55
    x_min, x_max = -20.0, 20.0
    y_min, y_max = 0.0, 70.0

    def xy(point):
        x = margin + (float(point[0]) - x_min) / (x_max - x_min) * (
            width - 2 * margin)
        y = height - margin - (float(point[1]) - y_min) / (
            y_max - y_min) * (height - 2 * margin)
        return int(round(x)), int(round(y))

    draw.line(
        (xy((x_min, y_min)), xy((x_max, y_min))),
        fill=(40, 40, 40), width=2)
    draw.line(
        (xy((0, y_min)), xy((0, y_max))),
        fill=(180, 180, 180), width=1)
    for y in range(10, 71, 10):
        left = xy((x_min, y))
        right = xy((x_max, y))
        draw.line((left, right), fill=(225, 225, 225), width=1)
        draw.text((8, left[1] - 6), f"{y}m", fill=(50, 50, 50))

    lanes = sample["gt_lanes_3d_enhanced"].numpy()
    visibility = (
        sample["gt_lanes_visibility_enhanced"].numpy() > 0.5)
    for index, (lane, mask) in enumerate(zip(lanes, visibility)):
        color = (255, 60, 220) if index == lane_index else (140, 170, 185)
        for run in _runs(mask):
            points = [xy(lane[anchor, :2]) for anchor in run]
            if len(points) >= 2:
                draw.line(
                    points, fill=color,
                    width=5 if index == lane_index else 2)
            for point in points:
                radius = 4 if index == lane_index else 2
                draw.ellipse(
                    (point[0] - radius, point[1] - radius,
                     point[0] + radius, point[1] + radius),
                    fill=color)

    target_runs = _runs(visibility[lane_index])
    if len(target_runs) >= 2:
        start = xy(lanes[lane_index, target_runs[0][-1], :2])
        end = xy(lanes[lane_index, target_runs[1][0], :2])
        _draw_dashed(draw, start, end, (255, 40, 30), width=4)
        midpoint = (
            (start[0] + end[0]) // 2,
            (start[1] + end[1]) // 2)
        draw.text(
            midpoint, "missing anchors",
            fill=(200, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))
    _title(
        panel,
        "BEV: target history-only lane has two valid y-ranges")
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        default="tests/test_data/tmp/temporal_visibility_gap_audit.json")
    parser.add_argument("--hit", type=int, default=0)
    parser.add_argument(
        "--output",
        default="tests/test_data/tmp/temporal_visibility_gap_case.jpg")
    args = parser.parse_args()

    report = json.loads(Path(args.audit).read_text())
    hit = report["hits"][args.hit]
    config = Config.fromfile(
        "configs/laneloom_openlane_image_lidar.py")
    dataset_config = dict(config.train_dataloader.dataset)
    dataset_config["temporal_gt_augmentation"] = True
    dataset = DATASETS.build(dataset_config)
    sample = dataset[hit["sample_index"]]

    current, enhanced = _camera_panels(sample, hit["lane_index"])
    width = current.width
    bev = _bev_panel(sample, hit["lane_index"], width)
    canvas = Image.new(
        "RGB", (width, current.height + enhanced.height + bev.height),
        (255, 255, 255))
    canvas.paste(current, (0, 0))
    canvas.paste(enhanced, (0, current.height))
    canvas.paste(bev, (0, current.height + enhanced.height))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    print(output_path)
    print(json.dumps(hit, ensure_ascii=False))


if __name__ == "__main__":
    main()
