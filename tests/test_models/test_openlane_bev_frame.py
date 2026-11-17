"""Same physical points must project identically in image and lane BEV frames."""
import numpy as np
import pytest
import torch
from unittest.mock import patch

from LaneLoom.datasets.openlane.dataset import OpenLaneDataset, WAYMO_VEHICLE_TO_GROUND_AXES
from LaneLoom.models.backbones.dynamic_pillar_bev import DynamicPillarBEVEncoder


def calibration():
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, 3] = [1.54, -.024, 2.1]
    dataset = OpenLaneDataset.__new__(OpenLaneDataset)
    return dataset._build_calib(dict(extrinsic=extrinsic,
        intrinsic=np.array([[120., 0, 100.], [0, 120., 60.], [0, 0, 1.]], np.float32)))


def encoder(aligned=True):
    return DynamicPillarBEVEncoder(point_cloud_range=[-25, 0, -5, 25, 100, 5],
        pillar_size=[.5, 1.], temporal_sampling_distance=False,
        out_channels=16, base_channels=8, align_to_lane_ground=aligned)


def project(matrix, points):
    pixels = matrix @ points.T
    return (pixels[:2]/pixels[2:3]).T


def test_lane_bev_and_rgb_projection_agree_after_calibration_not_only_axis_rotation():
    calib = calibration()
    vehicle = np.array([[20., 0., 0., 1.], [50., -5., 0., 1.]], np.float32)
    raw_ground = vehicle[:, :3] @ WAYMO_VEHICLE_TO_GROUND_AXES.T
    painted = torch.tensor(np.column_stack((raw_ground, np.ones((2, 6)))).astype(np.float32))
    old = painted.clone()
    aligned = encoder().points_in_lane_ground([painted], [dict(current=dict(openlane_calib=calib))])[0]
    expected = (calib['vehicle_to_ground'] @ vehicle.T).T[:, :3]
    torch.testing.assert_close(aligned[:, :3], torch.tensor(expected), atol=1e-5, rtol=0)
    torch.testing.assert_close(aligned[:, 3:], painted[:, 3:])
    torch.testing.assert_close(painted, old)
    good = torch.cat((aligned[:, :3], torch.ones(2, 1)), -1).numpy()
    bad = np.column_stack((raw_ground, np.ones(2)))
    pixels = project(calib['vehicle_to_raw_img'], vehicle)
    assert np.linalg.norm(project(calib['ground_to_raw_img'], bad)-pixels, axis=-1).max() > .5
    np.testing.assert_allclose(project(calib['ground_to_raw_img'], good), pixels, atol=1e-4)


def test_alignment_happens_after_rgb_painting_and_before_pillarization():
    model = encoder()
    calib = calibration()
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3] = calib['vehicle_to_raw_img']
    frame = dict(openlane_calib=calib, lidar2img=[matrix])
    meta = [dict(current=frame)]
    points = [torch.tensor([[0., 20., 0., .2, .3, 0.]])]
    image = torch.arange(120*200, dtype=torch.float32).reshape(1, 1, 1, 1, 120, 200).expand(-1, -1, -1, 3, -1, -1)/24000
    expected_painted = model.paint_points(points, images=image, img_metas=meta)
    expected_aligned = model.points_in_lane_ground(expected_painted, meta)
    captured = []
    def pillarize(value):
        captured.extend(value)
        return dict(batch_size=1, features=torch.zeros(0, 1), coordinates=torch.zeros(0, 3, dtype=torch.int32))
    with patch.object(model, 'pillarize', side_effect=pillarize):
        result = model(points, images=image, img_metas=meta)
    assert torch.isfinite(result['features']).all()
    torch.testing.assert_close(captured[0], expected_aligned[0])
    torch.testing.assert_close(captured[0][:, 3:], expected_painted[0][:, 3:])
    assert expected_painted[0][:, 5:8].abs().sum() > 0


def test_all_accumulated_sweeps_use_current_lane_origin_and_legacy_path_is_unchanged():
    current = calibration()
    history = dict(vehicle_to_ground=np.eye(4, dtype=np.float32)*2)
    points = [torch.tensor([[0., 20., 0., 1., 2., 3., 4., 5., 0.],
                            [0., 20., 0., 1., 2., 3., 4., 5., 1.]])]
    metas = [dict(current=dict(openlane_calib=current), queue={0: dict(openlane_calib=history)})]
    output = encoder().points_in_lane_ground(points, metas)[0]
    torch.testing.assert_close(output[0, :3], output[1, :3])
    torch.testing.assert_close(output[:, -1], points[0][:, -1])
    assert encoder(False).points_in_lane_ground(points, None) is points
    with pytest.raises(ValueError, match='calibration metadata'):
        encoder().points_in_lane_ground(points, [dict(current={})])
