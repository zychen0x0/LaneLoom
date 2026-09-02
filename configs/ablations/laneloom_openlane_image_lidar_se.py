"""Single-frame D5 ablation with the official SECOND voxel stack."""

_base_ = ['./laneloom_openlane_image_lidar_dp.py']

work_dir = './work_dirs/laneloom_openlane_image_lidar_se'

point_cloud_range = [-25.0, 0.0, -5.0, 25.0, 100.0, 5.0]
point_features = [
    'x', 'y', 'z', 'intensity', 'elongation', 'queue_index']
painted_point_channels = len(point_features) + 3
# Match the dynamic-pillar control's approximately 0.1 m XY quantization.
xy_voxel_size = 0.09765625
voxel_size = [xy_voxel_size, xy_voxel_size, 0.2]

model = dict(
    # Replace the integrated dynamic-pillar encoder with the official
    # MMDetection3D hard-voxel SECOND pipeline. Painting still happens first.
    pts_bev_encoder=None,
    pts_point_painter=dict(
        type='RGBPointPainter',
        use_rgb_pointpainting=True,
        point_features=point_features),
    pts_input_dim=painted_point_channels,
    pts_voxel_layer=dict(
        max_num_points=20,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(60000, 80000)),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=painted_point_channels),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=painted_point_channels,
        # [z, y, x]; z has the conventional extra boundary cell.
        sparse_shape=[51, 1024, 512],
        base_channels=16,
        output_channels=128),
    # SparseEncoder collapses its final two z cells into 2 * 128 channels.
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[1, 2, 2]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4]),
    lane_head=dict(
        # SECONDFPN concatenates its three 128-channel outputs.
        anchor_prior=dict(lidar_in_channels=384),
        bev_encoder=dict(
            lidar_bev_adapter=dict(lidar_in_channels=384))))
