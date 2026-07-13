import inspect
from pathlib import Path

from mmengine.config import Config
from mmengine.model import BaseModel
from mmdet3d.registry import MODELS

import LaneFormer  # noqa: F401 - register the project model.
from LaneFormer.detectors import LaneFormer as LaneFormerModel


def test_laneformer_is_the_single_mmengine_model_entrypoint():
    assert issubclass(LaneFormerModel, BaseModel)
    assert MODELS.get('LaneFormer') is LaneFormerModel

    parameters = inspect.signature(LaneFormerModel.forward).parameters
    assert 'mode' in parameters
    assert parameters.get('return' + '_loss') is None


def test_laneformer_forward_accepts_ddp_keyword_batches():
    signature = inspect.signature(LaneFormerModel.forward)
    signature.bind(
        None,
        mode='loss',
        img='image-batch',
        img_metas='metadata-batch',
        lidar='point-batch')


def test_config_enables_unused_parameter_detection_for_dynamic_bev_branches():
    config_path = Path(__file__).resolve().parents[2] / 'configs/laneformer_openlane_image_lidar.py'
    cfg = Config.fromfile(config_path)
    assert cfg.model_wrapper_cfg.type == 'MMDistributedDataParallel'
    assert cfg.model_wrapper_cfg.find_unused_parameters is True
