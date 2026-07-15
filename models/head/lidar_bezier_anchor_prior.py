"""LiDAR-conditioned cubic-Bézier anchors sampled from image features."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class LaneFormerLiDARBezierAnchorPrior(nn.Module):
    """Predict road shape and turn it into image-conditioned lane queries.

    LiDAR predicts one signed horizontal curvature and one longitudinal grade
    per sample.  They define 50 parallel cubic-Bézier anchors whose start
    positions are the one-metre cell centres of the configured x range.
    """

    def __init__(self, lidar_in_channels, image_in_channels, embed_dims,
                 pc_range, anchor_y_steps, num_anchors=50,
                 hidden_channels=64, max_curvature=0.005, max_grade=0.05,
                 curvature_loss_weight=1.0, grade_loss_weight=1.0,
                 **kwargs):
        super().__init__()
        self.pc_range = tuple(float(value) for value in pc_range)
        self.num_anchors = int(num_anchors)
        self.embed_dims = int(embed_dims)
        self.max_curvature = float(max_curvature)
        self.max_grade = float(max_grade)
        self.curvature_loss_weight = float(curvature_loss_weight)
        self.grade_loss_weight = float(grade_loss_weight)
        if self.num_anchors <= 0:
            raise ValueError('num_anchors must be positive.')
        if self.max_curvature <= 0 or self.max_grade <= 0:
            raise ValueError('max_curvature and max_grade must be positive.')

        anchor_y_steps = torch.as_tensor(anchor_y_steps, dtype=torch.float32)
        if anchor_y_steps.ndim != 1 or anchor_y_steps.numel() < 2:
            raise ValueError('anchor_y_steps must be a one-dimensional sequence.')
        if anchor_y_steps[0] != 0 or anchor_y_steps[-1] <= 0:
            raise ValueError('Bézier anchors must start at y=0 and extend forward.')
        self.register_buffer('anchor_y_steps', anchor_y_steps)
        length = anchor_y_steps[-1]
        t = anchor_y_steps / length
        self.register_buffer(
            'bezier_basis',
            torch.stack(((1 - t) ** 3, 3 * (1 - t) ** 2 * t,
                         3 * (1 - t) * t ** 2, t ** 3), dim=-1))

        x_min, x_max = self.pc_range[0], self.pc_range[3]
        cell_width = (x_max - x_min) / self.num_anchors
        start_x = x_min + (torch.arange(self.num_anchors) + 0.5) * cell_width
        self.register_buffer('start_x', start_x)

        self.geometry_encoder = nn.Sequential(
            nn.Conv2d(lidar_in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(hidden_channels * 16, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 2),
        )
        self.image_projection = nn.Conv2d(image_in_channels, embed_dims, 1)
        nn.init.zeros_(self.geometry_encoder[-1].weight)
        nn.init.zeros_(self.geometry_encoder[-1].bias)

    @staticmethod
    def _meta(img_metas, batch_idx):
        if img_metas is None:
            return {}
        meta = img_metas[batch_idx] if isinstance(img_metas, (list, tuple)) else img_metas
        return meta.get('current', meta) if isinstance(meta, dict) else {}

    @staticmethod
    def _feature(features):
        return features[-1] if isinstance(features, (list, tuple)) else features

    def _predict_geometry(self, pts_feats, img_feats):
        if pts_feats is not None:
            feature = self._feature(pts_feats)
            normalized = torch.tanh(self.geometry_encoder(feature))
            return (normalized[:, 0] * self.max_curvature,
                    normalized[:, 1] * self.max_grade, True)
        image = self._feature(img_feats)
        batch = image.size(0)
        zeros = image.new_zeros(batch)
        return zeros, zeros, False

    def _bezier_controls(self, curvature, grade):
        batch = curvature.size(0)
        start_x = self.start_x.to(curvature).view(1, self.num_anchors, 1)
        start_x = start_x.expand(batch, -1, -1)
        length = self.anchor_y_steps[-1].to(curvature)
        lateral_end = 0.5 * curvature[:, None, None] * length.square()
        x_controls = torch.cat((
            start_x,
            start_x,
            start_x + lateral_end / 3,
            start_x + lateral_end,
        ), dim=-1)
        height_end = grade[:, None, None] * length
        zero = torch.zeros_like(height_end).expand(-1, self.num_anchors, -1)
        height_end = height_end.expand(-1, self.num_anchors, -1)
        z_controls = torch.cat((
            zero,
            height_end / 3,
            height_end * (2 / 3),
            height_end,
        ), dim=-1)
        return x_controls, z_controls

    def _sample_curves(self, x_controls, z_controls):
        basis = self.bezier_basis.to(x_controls)
        x = torch.einsum('ac,bqc->bqa', basis, x_controls)
        z = torch.einsum('ac,bqc->bqa', basis, z_controls)
        y = self.anchor_y_steps.to(x).view(1, 1, -1).expand_as(x)
        return torch.stack((x, y, z), dim=-1)

    def sample_image_features(self, img_feats, img_metas, anchor_points,
                              point_weights=None):
        """Project curves and pool image features along their visible spans."""
        if img_feats is None:
            batch = anchor_points.size(0)
            features = anchor_points.new_zeros(
                batch, self.num_anchors, self.embed_dims)
            valid_ratio = anchor_points.new_zeros(batch, self.num_anchors)
            return features, valid_ratio

        image = self._feature(img_feats)
        batch, cameras, channels, feat_h, feat_w = image.shape
        projected_image = self.image_projection(
            image.reshape(batch * cameras, channels, feat_h, feat_w))
        projected_image = projected_image.reshape(
            batch, cameras, self.embed_dims, feat_h, feat_w)
        ones = torch.ones_like(anchor_points[..., :1])
        homogeneous = torch.cat((anchor_points, ones), dim=-1)
        feature_sum = image.new_zeros(
            batch, self.embed_dims, self.num_anchors,
            self.anchor_y_steps.numel())
        weight_sum = image.new_zeros(
            batch, 1, self.num_anchors, self.anchor_y_steps.numel())

        for batch_idx in range(batch):
            meta = self._meta(img_metas, batch_idx)
            matrices = meta.get('lane2img') or meta.get('lidar2img', [])
            image_shape = meta.get('img_shape', (feat_h, feat_w))
            image_h, image_w = float(image_shape[0]), float(image_shape[1])
            for camera_idx in range(cameras):
                matrix = (matrices[min(camera_idx, len(matrices) - 1)]
                          if matrices else torch.eye(4))
                matrix = torch.as_tensor(
                    matrix, device=image.device, dtype=image.dtype)
                projected = homogeneous[batch_idx] @ matrix.transpose(0, 1)
                depth = projected[..., 2]
                valid_depth = depth > 1e-5
                depth = depth.clamp_min(1e-5)
                grid = torch.stack((
                    projected[..., 0] / depth / max(image_w - 1, 1) * 2 - 1,
                    projected[..., 1] / depth / max(image_h - 1, 1) * 2 - 1,
                ), dim=-1)
                valid = (valid_depth & (grid[..., 0].abs() <= 1)
                         & (grid[..., 1].abs() <= 1))
                sampled = F.grid_sample(
                    projected_image[batch_idx, camera_idx:camera_idx + 1],
                    grid.unsqueeze(0), align_corners=True)
                weight = valid.to(image.dtype).unsqueeze(0).unsqueeze(0)
                if point_weights is not None:
                    weight = weight * point_weights[
                        batch_idx:batch_idx + 1].to(image).unsqueeze(1)
                feature_sum[batch_idx:batch_idx + 1] += sampled * weight
                weight_sum[batch_idx:batch_idx + 1] += weight

        query_features = (
            feature_sum.sum(dim=-1)
            / weight_sum.sum(dim=-1).clamp_min(1)).transpose(1, 2)
        valid_ratio = weight_sum.mean(dim=-1).squeeze(1) / max(cameras, 1)
        return query_features, valid_ratio

    def forward(self, pts_feats, img_feats, img_metas=None):
        if pts_feats is None and img_feats is None:
            raise ValueError('LiDAR Bézier anchors require image or LiDAR features.')
        curvature, grade, from_lidar = self._predict_geometry(pts_feats, img_feats)
        x_controls, z_controls = self._bezier_controls(curvature, grade)
        anchor_points = self._sample_curves(x_controls, z_controls)
        query_features, projection_valid_ratio = self.sample_image_features(
            img_feats, img_metas, anchor_points)
        return dict(
            curvature=curvature,
            grade=grade,
            curve_x_controls=x_controls,
            curve_z_controls=z_controls,
            anchor_points=anchor_points,
            query_features=query_features,
            projection_valid_ratio=projection_valid_ratio,
            from_lidar=from_lidar)

    def fit_targets(self, lanes_3d, visibility):
        """Fit robust per-sample road curvature and grade from lane GT."""
        lanes = lanes_3d.float()
        weights = (visibility > 0.5).to(lanes.dtype)
        y, x, z = lanes[..., 1], lanes[..., 0], lanes[..., 2]
        x_design = torch.stack((
            torch.ones_like(y), y, 0.5 * y.square()), dim=-1)
        z_design = torch.stack((torch.ones_like(y), y), dim=-1)

        def solve(design, values):
            normal = torch.einsum(
                'bla,blad,blae->blde', weights, design, design)
            rhs = torch.einsum('bla,blad,bla->bld', weights, design, values)
            eye = torch.eye(
                design.size(-1), device=design.device, dtype=design.dtype)
            return torch.linalg.solve(normal + 1e-6 * eye, rhs.unsqueeze(-1))[
                ..., 0]

        x_coefficients = solve(x_design, x)
        z_coefficients = solve(z_design, z)
        lane_valid = weights.sum(dim=-1) >= 4
        lane_curvature = x_coefficients[..., 2].masked_fill(~lane_valid, torch.nan)
        lane_grade = z_coefficients[..., 1].masked_fill(~lane_valid, torch.nan)
        target_valid = lane_valid.any(dim=1)
        target_curvature = torch.nanmedian(lane_curvature, dim=1).values
        target_grade = torch.nanmedian(lane_grade, dim=1).values
        target_curvature = torch.nan_to_num(target_curvature).clamp(
            -self.max_curvature, self.max_curvature)
        target_grade = torch.nan_to_num(target_grade).clamp(
            -self.max_grade, self.max_grade)
        return target_curvature, target_grade, target_valid

    def loss(self, prior, lanes_3d, visibility):
        if not prior['from_lidar']:
            return {}
        target_curvature, target_grade, valid = self.fit_targets(
            lanes_3d, visibility)
        if not valid.any():
            zero = prior['curvature'].sum() * 0
            return dict(
                loss_lidar_curvature=zero,
                loss_lidar_grade=zero)
        curvature_loss = F.smooth_l1_loss(
            prior['curvature'][valid] / self.max_curvature,
            target_curvature[valid] / self.max_curvature)
        grade_loss = F.smooth_l1_loss(
            prior['grade'][valid] / self.max_grade,
            target_grade[valid] / self.max_grade)
        return dict(
            loss_lidar_curvature=curvature_loss * self.curvature_loss_weight,
            loss_lidar_grade=grade_loss * self.grade_loss_weight)
