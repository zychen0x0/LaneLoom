import ast
from pathlib import Path

import torch
from mmengine.config import Config

from LaneLoom.datasets.openlane.pipelines import GridMask


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PROJECT_REFERENCES = tuple(
    ''.join(parts) for parts in (
        ('lane', 'det3d'),
        ('bev', 'former'),
        ('bev', 'fusion'),
        ('dv-', '3dlane'),
        ('dv_', '3dlane'),
        ('projects.', 'mmdet'),
    ))


def test_source_and_configs_do_not_reference_other_project_code():
    source_files = list(PROJECT_ROOT.rglob('*.py'))
    for source_file in source_files:
        if '__pycache__' in source_file.parts or source_file == Path(__file__).resolve():
            continue
        tree = ast.parse(source_file.read_text(encoding='utf-8'))
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name.lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module.lower())
        forbidden = [
            name for name in FORBIDDEN_PROJECT_REFERENCES
            if any(module == name or module.startswith(name + '.')
                   for module in imported_modules)
        ]
        assert not forbidden, f'{source_file} references {forbidden}'


def test_config_owns_its_default_data_root(monkeypatch):
    monkeypatch.delenv('LANELOOM_DATA_ROOT', raising=False)
    cfg = Config.fromfile(PROJECT_ROOT / 'configs/laneloom_openlane_image_lidar.py')
    mounted_root = Path('/workspace/lanedet3d/data')
    expected_root = mounted_root if mounted_root.is_dir() else PROJECT_ROOT / 'data'
    assert Path(cfg.data_root) == expected_root


def test_grid_mask_is_a_local_module_and_preserves_image_shape():
    augmentation = GridMask(ratio=0.5, prob=1.0).train()
    image = torch.ones(2, 3, 12, 16)
    masked = augmentation(image)

    assert GridMask.__module__ == (
        'LaneLoom.datasets.openlane.pipelines.transforms')
    assert masked.shape == image.shape
    assert torch.any(masked == 0)
