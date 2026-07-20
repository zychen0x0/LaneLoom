from mmengine.registry import FUNCTIONS
from mmdet3d.registry import DATASETS as MMDET3D_DATASETS

import LaneLoom.datasets  # noqa: F401 - trigger registry side effects


def test_openlane_dataset_and_collate_are_registered_for_mmengine_scope():
    assert MMDET3D_DATASETS.get('OpenLaneDataset') is not None
    assert FUNCTIONS.get('openlane_collate') is not None
