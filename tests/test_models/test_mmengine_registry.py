import torch
from mmengine.registry import FUNCTIONS
from mmdet3d.registry import DATASETS as MMDET3D_DATASETS

import LaneFormer.datasets  # noqa: F401 - trigger registry side effects
from LaneFormer.models.transformers.spatial_cross_attention import (
    LaneFormerSpatialCrossAttention,
)
from LaneFormer.models.transformers.temporal_self_attention import (
    LaneFormerTemporalSelfAttention,
)


def test_openlane_dataset_and_collate_are_registered_for_mmengine_scope():
    assert MMDET3D_DATASETS.get('OpenLaneDataset') is not None
    assert FUNCTIONS.get('openlane_collate') is not None


def test_spatial_reference_points_accept_explicit_integer_heights():
    attention = LaneFormerSpatialCrossAttention(
        embed_dims=8, bev_h=2, bev_w=2, pc_range=[-25., 0., -5., 25., 100., 5.],
        num_points_in_pillar=5, reference_zs=[-5., -2., 0., 2., 5.])
    references = attention._reference_points(device='cpu', dtype=torch.float32)
    assert references[:, 2, 0].tolist() == [-5., -2., 0., 2., 5.]


def test_bev_reference_points_use_one_meter_cell_centers():
    attention = LaneFormerSpatialCrossAttention(
        embed_dims=8, bev_h=100, bev_w=50,
        pc_range=[-25., 0., -5., 25., 100., 5.],
        num_points_in_pillar=5, reference_zs=[-5., -2., 0., 2., 5.])
    references = attention._reference_points(device='cpu', dtype=torch.float32)

    assert torch.equal(
        references[0, 0, :3], torch.tensor([-24.5, -23.5, -22.5]))
    assert torch.equal(
        references[0, 1, ::50][:3], torch.tensor([0.5, 1.5, 2.5]))


def test_identity_pose_preserves_center_aligned_history_bev():
    attention = LaneFormerTemporalSelfAttention(
        embed_dims=1, bev_h=2, bev_w=2,
        pc_range=[-2., 0., -1., 2., 2., 1.])
    history_bev = torch.tensor([[[[1., 2.], [3., 4.]]]])
    metadata = [dict(current=dict(lidaradj2lidarcurr=torch.eye(4)))]

    warped = attention._warp_to_current(history_bev, metadata)

    assert torch.equal(warped, history_bev)
