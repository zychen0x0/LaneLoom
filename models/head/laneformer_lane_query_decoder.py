"""Configurable lane-query decoder."""
import torch
import torch.nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class LaneFormerLaneQueryDecoder(nn.Module):
    """Decode BEV memory into cubic-Bézier 3D lanes sampled at anchors."""

    def __init__(self, embed_dims, bev_h, bev_w, pc_range, num_query,
                 anchor_y_steps, num_layers=3, num_heads=8,
                 feedforward_channels=1024, dropout=0.1, num_category=21,
                 dn_num_groups=0, dn_noise_scale=0.5,
                 objectness_prior_probability=0.01,
                 visibility_temperature=0.75,
                 visibility_start_prior=0.14,
                 visibility_end_prior=0.58,
                 **kwargs):
        super().__init__()
        self.pc_range = pc_range
        self.num_query = num_query
        self.num_layers = int(num_layers)
        self.register_buffer('anchor_y_steps', torch.tensor(anchor_y_steps, dtype=torch.float32))
        self.num_anchors = len(anchor_y_steps)
        self.num_control_points = 4
        anchor_span = self.anchor_y_steps[-1] - self.anchor_y_steps[0]
        if anchor_span <= 0:
            raise ValueError('anchor_y_steps must be strictly increasing.')
        t = (self.anchor_y_steps - self.anchor_y_steps[0]) / anchor_span
        self.register_buffer(
            'bezier_basis',
            torch.stack((
                (1 - t) ** 3,
                3 * (1 - t) ** 2 * t,
                3 * (1 - t) * t ** 2,
                t ** 3,
            ), dim=-1))
        self.num_category = int(num_category)
        self.dn_num_groups = int(dn_num_groups)
        if self.dn_num_groups < 0:
            raise ValueError('dn_num_groups must be non-negative.')
        self.dn_noise_scale = float(dn_noise_scale)
        self.visibility_temperature = float(visibility_temperature)
        if self.visibility_temperature <= 0:
            raise ValueError('visibility_temperature must be positive.')
        visibility_start_prior = float(visibility_start_prior)
        visibility_end_prior = float(visibility_end_prior)
        if not 0 <= visibility_start_prior < visibility_end_prior <= 1:
            raise ValueError(
                'visibility priors must satisfy 0 <= start < end <= 1.')
        self.register_buffer(
            'visibility_indices', torch.arange(self.num_anchors, dtype=torch.float32),
            persistent=False)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dims, nhead=num_heads,
            dim_feedforward=feedforward_channels, dropout=dropout,
            activation='relu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.query = nn.Embedding(num_query, embed_dims)
        self.query_pos = nn.Embedding(num_query, embed_dims)
        if self.dn_num_groups > 0:
            self.dn_query = nn.Sequential(
                nn.Linear(self.num_control_points * 2, embed_dims),
                nn.ReLU(inplace=True), nn.Linear(embed_dims, embed_dims))
            self.dn_query_pos = nn.Sequential(
                nn.Linear(self.num_control_points * 2, embed_dims),
                nn.ReLU(inplace=True), nn.Linear(embed_dims, embed_dims))
        else:
            self.dn_query = None
            self.dn_query_pos = None
        self.norm = nn.LayerNorm(embed_dims)
        self.x_head = nn.Linear(embed_dims, self.num_control_points)
        self.z_head = nn.Linear(embed_dims, self.num_control_points)
        self.x_refine_heads = nn.ModuleList([
            nn.Linear(embed_dims, self.num_control_points)
            for _ in range(max(self.num_layers - 1, 0))])
        self.z_refine_heads = nn.ModuleList([
            nn.Linear(embed_dims, self.num_control_points)
            for _ in range(max(self.num_layers - 1, 0))])
        self.vis_head = nn.Linear(embed_dims, self.num_anchors)
        self.vis_bounds_head = nn.Linear(embed_dims, 2)
        self.cls_head = nn.Linear(embed_dims, 1)
        self.category_head = nn.Linear(embed_dims, self.num_category)
        prior = float(objectness_prior_probability)
        if not 0.0 < prior < 1.0:
            raise ValueError('objectness_prior_probability must be in (0, 1).')
        nn.init.constant_(self.cls_head.bias, torch.logit(torch.tensor(prior)))
        nn.init.zeros_(self.vis_head.weight)
        nn.init.zeros_(self.vis_head.bias)
        nn.init.zeros_(self.vis_bounds_head.weight)
        prior_start = visibility_start_prior * (self.num_anchors - 1)
        prior_end = visibility_end_prior * (self.num_anchors - 1)
        start_probability = (prior_start + .5) / self.num_anchors
        end_probability = ((prior_end - prior_start) /
                           (self.num_anchors - .5 - prior_start))
        nn.init.constant_(
            self.vis_bounds_head.bias[0],
            torch.logit(torch.tensor(start_probability)))
        nn.init.constant_(
            self.vis_bounds_head.bias[1],
            torch.logit(torch.tensor(end_probability)))
        for head in (*self.x_refine_heads, *self.z_refine_heads):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _fit_bezier_controls(self, lanes_3d, visibility):
        """Fit fixed-anchor GT lanes to the decoder's four Bézier controls."""
        weights = visibility.to(dtype=lanes_3d.dtype)
        basis = self.bezier_basis.to(dtype=lanes_3d.dtype)
        eye = torch.eye(self.num_control_points, device=lanes_3d.device,
                        dtype=lanes_3d.dtype)
        normal = torch.einsum('ac,bla,ad->blcd', basis, weights, basis)
        normal = normal + 1e-4 * eye

        def solve(values):
            rhs = torch.einsum('ac,bla->blc', basis, weights * values)
            return torch.linalg.solve(normal, rhs.unsqueeze(-1)).squeeze(-1)

        return solve(lanes_3d[..., 0]), solve(lanes_3d[..., 2])

    def _build_dn_queries(self, lanes_3d, visibility):
        valid = visibility.bool().any(dim=-1)
        control_x, control_z = self._fit_bezier_controls(lanes_3d, visibility)
        x_range = self.pc_range[3] - self.pc_range[0]
        z_range = max(abs(self.pc_range[2]), abs(self.pc_range[5]))
        controls = torch.cat((
            (control_x - self.pc_range[0]) / x_range,
            control_z / z_range), dim=-1)
        if self.dn_num_groups > 1:
            controls = controls.repeat(1, self.dn_num_groups, 1)
            valid = valid.repeat(1, self.dn_num_groups)
        noise_scale = controls.new_tensor(
            [self.dn_noise_scale / x_range] * self.num_control_points +
            [self.dn_noise_scale / z_range] * self.num_control_points)
        noisy_controls = controls + torch.randn_like(controls) * noise_scale
        noisy_controls = torch.where(valid.unsqueeze(-1), noisy_controls,
                                     torch.zeros_like(noisy_controls))
        indices = torch.arange(lanes_3d.size(1), device=lanes_3d.device)
        indices = indices.repeat(self.dn_num_groups)
        return (self.dn_query(noisy_controls), self.dn_query_pos(noisy_controls),
                valid, indices)

    def _decode_predictions(self, query, reference_controls=None, layer_idx=0):
        x_range = self.pc_range[3] - self.pc_range[0]
        z_range = max(abs(self.pc_range[2]), abs(self.pc_range[5]))
        x_head = self.x_head if layer_idx == 0 else self.x_refine_heads[layer_idx - 1]
        z_head = self.z_head if layer_idx == 0 else self.z_refine_heads[layer_idx - 1]
        x_logits = x_head(query)
        z_logits = z_head(query)
        if reference_controls is not None:
            reference_x, reference_z = reference_controls
            normalized_x = ((reference_x - self.pc_range[0]) / x_range).clamp(
                1e-4, 1 - 1e-4)
            normalized_z = (reference_z / z_range).clamp(-1 + 1e-4, 1 - 1e-4)
            x_logits = x_logits + torch.logit(normalized_x)
            z_logits = z_logits + torch.atanh(normalized_z)
        curve_x_controls = torch.sigmoid(x_logits) * x_range + self.pc_range[0]
        curve_z_controls = torch.tanh(z_logits) * z_range
        visibility_bounds = self.vis_bounds_head(query)
        boundary_span = float(self.num_anchors)
        visibility_start = (
            torch.sigmoid(visibility_bounds[..., 0]) * boundary_span - 0.5)
        visibility_end = visibility_start + torch.sigmoid(
            visibility_bounds[..., 1]) * (
                self.num_anchors - 0.5 - visibility_start)
        anchor_indices = self.visibility_indices.to(dtype=query.dtype)
        interval_probability = (
            torch.sigmoid(
                (anchor_indices - visibility_start.unsqueeze(-1)) /
                self.visibility_temperature)
            * torch.sigmoid(
                (visibility_end.unsqueeze(-1) - anchor_indices) /
                self.visibility_temperature))
        visibility_logits = (
            torch.logit(interval_probability.clamp(1e-4, 1 - 1e-4))
            + self.vis_head(query))
        return dict(
            pred_x=torch.einsum('ac,bqc->bqa', self.bezier_basis,
                                curve_x_controls),
            pred_z=torch.einsum('ac,bqc->bqa', self.bezier_basis,
                                curve_z_controls),
            curve_x_controls=curve_x_controls,
            curve_z_controls=curve_z_controls,
            pred_vis=visibility_logits,
            pred_vis_start=visibility_start,
            pred_vis_end=visibility_end,
            pred_vis_interval=interval_probability,
            pred_cls=self.cls_head(query).squeeze(-1),
            pred_category=self.category_head(query))

    def forward(self, bev, bev_pos, dn_lanes_3d=None, dn_visibility=None):
        batch = bev.size(0)
        memory = bev.flatten(2).transpose(1, 2)
        memory = memory + bev_pos.weight.unsqueeze(0).expand(batch, -1, -1)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(batch, -1, -1)
        query = self.query.weight.unsqueeze(0).expand(batch, -1, -1)
        dn_query = None
        if (self.training and self.dn_num_groups > 0 and
                dn_lanes_3d is not None and dn_visibility is not None):
            dn_query, dn_query_pos, dn_valid_mask, dn_target_indices = (
                self._build_dn_queries(dn_lanes_3d, dn_visibility))
            query = torch.cat((query, dn_query), dim=1)
            query_pos = torch.cat((query_pos, dn_query_pos), dim=1)
            attention_mask = torch.zeros(
                (query.size(1), query.size(1)), dtype=torch.bool,
                device=query.device)
            attention_mask[:self.num_query, self.num_query:] = True
        else:
            attention_mask = None
        query = query + query_pos
        layer_predictions = []
        main_reference = None
        dn_reference = None
        for layer_idx, layer in enumerate(self.decoder.layers):
            query = layer(query, memory, tgt_mask=attention_mask)
            normalized_query = self.norm(query)
            predictions = self._decode_predictions(
                normalized_query[:, :self.num_query], main_reference,
                layer_idx=layer_idx)
            main_reference = (
                predictions['curve_x_controls'].detach(),
                predictions['curve_z_controls'].detach())
            if dn_query is not None:
                predictions['dn_predictions'] = self._decode_predictions(
                    normalized_query[:, self.num_query:], dn_reference,
                    layer_idx=layer_idx)
                dn_reference = (
                    predictions['dn_predictions']['curve_x_controls'].detach(),
                    predictions['dn_predictions']['curve_z_controls'].detach())
                predictions['dn_valid_mask'] = dn_valid_mask
                predictions['dn_target_indices'] = dn_target_indices
            layer_predictions.append(predictions)
        predictions = layer_predictions[-1]
        predictions['aux_predictions'] = layer_predictions[:-1]
        predictions['bev'] = bev
        return predictions
