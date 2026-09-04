"""Shared helpers for OpenLane data-augmentation visualizations."""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from mmengine.config import Config  # noqa: E402
from mmdet3d.registry import DATASETS  # noqa: E402

import LaneLoom.datasets  # noqa: E402,F401 - register project datasets.


CONFIG_PATH = (
    REPO_ROOT / 'configs/main/laneloom_openlane_image_lidar_Temporal.py')
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent


def load_config():
    return Config.fromfile(str(CONFIG_PATH))


def require_openlane_data(cfg):
    required_paths = [
        Path(cfg.dataset_cfg.data_label_dir),
        Path(cfg.dataset_cfg.data_image_dir),
        Path(cfg.dataset_cfg.lidar_processed_dir),
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            f'OpenLane/Waymo data is not available: {missing_paths}')


def resolve_image_path(image_path):
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f'Input image does not exist: {image_path}')
    return image_path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_identity(image_path):
    image_path = resolve_image_path(image_path)
    return image_path.parent.name, image_path.stem[:12]


def output_dir_for(image_path, output_root=DEFAULT_OUTPUT_ROOT):
    scene_name, sample_key = sample_identity(image_path)
    output_dir = (
        Path(output_root).expanduser().resolve()
        / 'openlane' / scene_name / sample_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_sample_dataset(cfg, image_path=None, max_samples=128):
    dataset_cfg = copy.deepcopy(dict(cfg.train_dataloader['dataset']))
    dataset_cfg['dataset_cfg'] = copy.deepcopy(
        dict(dataset_cfg['dataset_cfg']))
    dataset_cfg['dataset_cfg']['load_interval'] = 1

    if image_path is None:
        dataset_cfg['max_samples'] = max_samples
        return DATASETS.build(dataset_cfg)

    image_path = resolve_image_path(image_path)
    image_root = Path(cfg.dataset_cfg.data_image_dir).expanduser().resolve()
    try:
        relative = image_path.relative_to(image_root)
    except ValueError as error:
        raise ValueError(
            f'OpenLane image must be under configured root {image_root}.') \
            from error
    if len(relative.parts) < 3:
        raise ValueError(
            'OpenLane image path must end with <split>/<scene>/<frame>.jpg.')

    split = relative.parts[0]
    dataset_cfg['split'] = split
    if split != cfg.dataset_cfg.train_split:
        dataset_cfg['dataset_cfg']['val_pipeline'] = copy.deepcopy(
            list(cfg.train_pipeline))
    return DATASETS.build(dataset_cfg)


def select_sample(dataset, image_path=None):
    if image_path is not None:
        image_path = resolve_image_path(image_path)
        scene_name, frame_key = sample_identity(image_path)
        for index, label_path in enumerate(dataset.label_files):
            if (label_path.parent.name == scene_name
                    and label_path.stem[:12] == frame_key):
                return dataset[index], index
        raise FileNotFoundError(
            f'Cannot find OpenLane label for image sample: {image_path}')

    for index in range(len(dataset)):
        candidate = dataset[index]
        frame_keys = candidate['img_metas']['frame_keys']
        if len(set(frame_keys)) == len(frame_keys):
            return candidate, index
    return dataset[0], 0


def current_image_path(sample):
    return Path(
        sample['img_metas']['current']['img_filename'][0]).resolve()


def make_tensor_canvas(tensor, denorm=True, row_labels=None,
                       col_labels=None):
    """Build one PIL canvas from a [T, V, 3, H, W] image tensor."""
    if tensor.ndim == 6:
        if tensor.shape[0] != 1:
            raise ValueError('Only batch_size=1 can be visualized.')
        tensor = tensor[0]
    if tensor.ndim != 5 or tensor.shape[2] != 3:
        raise ValueError(
            f'Expected [T, V, 3, H, W], received {tuple(tensor.shape)}.')

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    array = tensor.detach().permute(0, 1, 3, 4, 2).cpu().numpy()
    time_steps, view_num, height, width, _ = array.shape
    if denorm:
        array = array * std[None, None, None, None, :]
        array += mean[None, None, None, None, :]
    array = np.clip(array, 0, 255).astype(np.uint8)

    canvas = Image.new(
        'RGB', (width * view_num, height * time_steps), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for time_index in range(time_steps):
        for view_index in range(view_num):
            x0 = view_index * width
            y0 = time_index * height
            canvas.paste(
                Image.fromarray(array[time_index, view_index]), (x0, y0))
            if row_labels is not None or col_labels is not None:
                row_name = (
                    row_labels[time_index] if row_labels else f't{time_index}')
                col_name = (
                    col_labels[view_index]
                    if col_labels else f'view{view_index}')
                draw.text(
                    (x0 + 8, y0 + 8), f'{row_name} {col_name}',
                    fill=(255, 0, 0))
    return canvas


def make_labeled_grid(panels, columns=3):
    rows = (len(panels) + columns - 1) // columns
    panel_width, panel_height = panels[0][1].size
    canvas = Image.new(
        'RGB', (columns * panel_width, rows * panel_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for panel_index, (name, panel) in enumerate(panels):
        if panel.size != (panel_width, panel_height):
            raise ValueError('All panels must use the same image size.')
        x0 = panel_index % columns * panel_width
        y0 = panel_index // columns * panel_height
        canvas.paste(panel, (x0, y0))
        draw.rectangle(
            (x0, y0, x0 + panel_width, y0 + 30), fill=(8, 8, 10))
        draw.text((x0 + 8, y0 + 7), name, fill=(245, 245, 245))
    return canvas


def write_report(report, report_path):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
