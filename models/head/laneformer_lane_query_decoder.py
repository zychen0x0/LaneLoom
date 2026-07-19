"""Configurable lane-query decoder."""
import torch
import torch.nn as nn

from mmdet3d.registry import MODELS


class _PerQueryTransformerLayer(nn.Module):
    """Update lane queries from their own sampled curve evidence."""

    def __init__(self, embed_dims, num_heads, feedforward_channels, dropout):
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dims, feedforward_channels)
        self.linear2 = nn.Linear(feedforward_channels, embed_dims)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, query, query_pos, evidence, evidence_padding_mask,
                self_attention_mask, num_main_queries):
        normalized = self.norm1(query)
        attended, _ = self.self_attn(
            normalized + query_pos, normalized + query_pos, normalized,
            attn_mask=self_attention_mask, need_weights=False)
        query = query + self.dropout1(attended)

        main_query = query[:, :num_main_queries]
        main_pos = query_pos[:, :num_main_queries]
        batch, num_query, num_tokens, channels = evidence.shape
        cross_query = (self.norm2(main_query) + main_pos).reshape(
            batch * num_query, 1, channels)
        evidence = evidence.reshape(batch * num_query, num_tokens, channels)
        evidence_padding_mask = evidence_padding_mask.reshape(
            batch * num_query, num_tokens)
        attended, _ = self.cross_attn(
            cross_query, evidence, evidence,
            key_padding_mask=evidence_padding_mask, need_weights=False)
        main_query = main_query + self.dropout2(
            attended.reshape(batch, num_query, channels))
        query = torch.cat((main_query, query[:, num_main_queries:]), dim=1)

        normalized = self.norm3(query)
        query = query + self.dropout3(
            self.linear2(self.activation(self.linear1(normalized))))
        return query


@MODELS.register_module()
class LaneFormerLaneQueryDecoder(nn.Module):
    """Decode BEV memory into cubic-Bézier 3D lanes sampled at anchors."""

    def __init__(self, embed_dims, bev_h, bev_w, pc_range, num_query,
                 anchor_y_steps, evidence_y_steps=None,
                 num_layers=3, num_heads=8,
                 feedforward_channels=1024, dropout=0.1, num_category=21,
                 dn_num_groups=0,
                 layer_query_budget=None,
                 objectness_prior_probability=0.01,
                 visibility_temperature=0.75,
                 visibility_start_prior=0.14,
                 visibility_end_prior=0.58,
                 reference_x_residual_scale=5.0,
                 reference_z_residual_scale=1.0,
                 **kwargs):
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.pc_range = pc_range
        self.num_query = num_query
        self.num_layers = int(num_layers)
        self.layer_query_budget = (
            None if layer_query_budget is None else int(layer_query_budget))
        if (self.layer_query_budget is not None
                and not 0 < self.layer_query_budget <= self.num_query):
            raise ValueError(
                'layer_query_budget must be in (0, num_query].')
        self.register_buffer('anchor_y_steps', torch.tensor(anchor_y_steps, dtype=torch.float32))
        self.num_anchors = len(anchor_y_steps)
        self.num_control_points = 4
        curve_length = self.anchor_y_steps[-1]
        if curve_length <= 0:
            raise ValueError('anchor_y_steps must be strictly increasing.')
        if not torch.all(
                self.anchor_y_steps[1:] > self.anchor_y_steps[:-1]):
            raise ValueError('anchor_y_steps must be strictly increasing.')
        t = self.anchor_y_steps / curve_length
        self.register_buffer(
            'bezier_basis',
            torch.stack((
                (1 - t) ** 3,
                3 * (1 - t) ** 2 * t,
                3 * (1 - t) * t ** 2,
                t ** 3,
            ), dim=-1))
        if evidence_y_steps is None:
            evidence_y_steps = anchor_y_steps
        evidence_y_steps = torch.as_tensor(
            evidence_y_steps, dtype=torch.float32)
        if (evidence_y_steps.ndim != 1 or evidence_y_steps.numel() < 2
                or evidence_y_steps[0] < 0
                or evidence_y_steps[-1] > curve_length
                or not torch.all(
                    evidence_y_steps[1:] > evidence_y_steps[:-1])):
            raise ValueError(
                'evidence_y_steps must be positive, increasing and within '
                'the output curve range.')
        self.register_buffer('evidence_y_steps', evidence_y_steps)
        evidence_t = evidence_y_steps / curve_length
        self.register_buffer(
            'evidence_bezier_basis',
            torch.stack((
                (1 - evidence_t) ** 3,
                3 * (1 - evidence_t) ** 2 * evidence_t,
                3 * (1 - evidence_t) * evidence_t ** 2,
                evidence_t ** 3,
            ), dim=-1))
        self.num_category = int(num_category)
        if int(dn_num_groups) != 0:
            raise ValueError(
                'denoising queries are not supported by the per-query '
                'evidence decoder; use decoder-layer auxiliary supervision.')
        self.visibility_temperature = float(visibility_temperature)
        self.reference_x_residual_scale = float(reference_x_residual_scale)
        self.reference_z_residual_scale = float(reference_z_residual_scale)
        if (self.reference_x_residual_scale <= 0
                or self.reference_z_residual_scale <= 0):
            raise ValueError('Reference residual scales must be positive.')
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
        self.layers = nn.ModuleList([
            _PerQueryTransformerLayer(
                embed_dims, num_heads, feedforward_channels, dropout)
            for _ in range(self.num_layers)])
        self.query = nn.Embedding(num_query, embed_dims)
        self.query_pos = nn.Embedding(num_query, embed_dims)
        self.null_evidence = nn.Parameter(torch.zeros(1, 1, 1, embed_dims))
        self.norm = nn.LayerNorm(embed_dims)
        self.x_head = nn.Linear(embed_dims, self.num_control_points)
        self.z_head = nn.Linear(embed_dims, self.num_control_points)
        self.x_reference_head = nn.Linear(
            embed_dims, self.num_control_points)
        self.z_reference_head = nn.Linear(
            embed_dims, self.num_control_points)
        self.x_refine_heads = nn.ModuleList([
            nn.Linear(embed_dims, self.num_control_points)
            for _ in range(max(self.num_layers - 1, 0))])
        self.z_refine_heads = nn.ModuleList([
            nn.Linear(embed_dims, self.num_control_points)
            for _ in range(max(self.num_layers - 1, 0))])
        self.vis_bounds_head = nn.Linear(embed_dims, 2)
        self.vis_bounds_refine_heads = nn.ModuleList([
            nn.Linear(embed_dims, 2)
            for _ in range(max(self.num_layers - 1, 0))])
        self.cls_head = nn.Linear(embed_dims, 1)
        self.category_head = nn.Linear(embed_dims, self.num_category)
        prior = float(objectness_prior_probability)
        if not 0.0 < prior < 1.0:
            raise ValueError('objectness_prior_probability must be in (0, 1).')
        nn.init.constant_(self.cls_head.bias, torch.logit(torch.tensor(prior)))
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
        for head in (self.x_reference_head, self.z_reference_head,
                     *self.x_refine_heads, *self.z_refine_heads,
                     *self.vis_bounds_refine_heads):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _decode_predictions(self, query, reference_controls=None,
                            reference_visibility=None, layer_idx=0):
        x_range = self.pc_range[3] - self.pc_range[0]
        z_range = max(abs(self.pc_range[2]), abs(self.pc_range[5]))
        if reference_controls is None:
            if layer_idx != 0:
                raise ValueError(
                    'Only the first decoder layer may omit curve references.')
            x_logits = self.x_head(query)
            z_logits = self.z_head(query)
            curve_x_controls = (
                torch.sigmoid(x_logits) * x_range + self.pc_range[0])
            curve_z_controls = torch.tanh(z_logits) * z_range
        else:
            x_head = (self.x_reference_head if layer_idx == 0
                      else self.x_refine_heads[layer_idx - 1])
            z_head = (self.z_reference_head if layer_idx == 0
                      else self.z_refine_heads[layer_idx - 1])
            x_logits = x_head(query)
            z_logits = z_head(query)
            reference_x, reference_z = reference_controls
            curve_x_controls = reference_x + (
                torch.tanh(x_logits) * self.reference_x_residual_scale)
            curve_z_controls = reference_z + (
                torch.tanh(z_logits) * self.reference_z_residual_scale)
        visibility_head = (
            self.vis_bounds_head if layer_idx == 0
            else self.vis_bounds_refine_heads[layer_idx - 1])
        visibility_bounds = visibility_head(query)
        boundary_span = float(self.num_anchors)
        if reference_visibility is not None:
            reference_start, reference_end = reference_visibility
            normalized_start = ((reference_start + 0.5) / boundary_span).clamp(
                1e-4, 1 - 1e-4)
            normalized_span = (
                (reference_end - reference_start)
                / (self.num_anchors - 0.5 - reference_start)).clamp(
                    1e-4, 1 - 1e-4)
            visibility_bounds = visibility_bounds + torch.stack((
                torch.logit(normalized_start),
                torch.logit(normalized_span)), dim=-1)
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
        visibility_logits = torch.logit(
            interval_probability.clamp(1e-4, 1 - 1e-4))
        pred_objectness = self.cls_head(query).squeeze(-1)
        pred_category = self.category_head(query)
        background = pred_category.new_zeros(*pred_category.shape[:-1], 1)
        pred_logits = torch.cat((
            background, pred_category + pred_objectness.unsqueeze(-1)), dim=-1)
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
            pred_cls=pred_objectness,
            pred_category=pred_category,
            pred_logits=pred_logits)

    def _reference_points(self, reference_controls, evidence=False):
        reference_x, reference_z = reference_controls
        basis = (
            self.evidence_bezier_basis if evidence else self.bezier_basis)
        y_steps = (
            self.evidence_y_steps if evidence else self.anchor_y_steps)
        x = torch.einsum(
            'ac,bqc->bqa', basis, reference_x)
        z = torch.einsum(
            'ac,bqc->bqa', basis, reference_z)
        y = y_steps.to(x).view(1, 1, -1).expand_as(x)
        return torch.stack((x, y, z), dim=-1)

    def _sample_evidence(self, reference_controls, sampler, batch, dtype,
                         device, query_features):
        num_query = query_features.size(1)
        null = self.null_evidence.to(dtype=dtype).expand(
            batch, num_query, -1, -1)
        if reference_controls is None or sampler is None:
            return null, torch.zeros(
                batch, num_query, 1, dtype=torch.bool, device=device), {}
        points = self._reference_points(reference_controls, evidence=True)
        sampled = sampler(points, query_features)
        if torch.is_tensor(sampled):
            sampled = dict(
                tokens=sampled.unsqueeze(2),
                key_padding_mask=torch.zeros(
                    *sampled.shape[:2], 1, dtype=torch.bool,
                    device=sampled.device))
        evidence = sampled['tokens']
        padding_mask = sampled['key_padding_mask']
        expected = (batch, num_query)
        if (tuple(evidence.shape[:2]) != expected
                or evidence.size(-1) != self.embed_dims):
            raise ValueError(
                'query evidence must have shape [B, num_query, M, C], got '
                f'{tuple(evidence.shape)}.')
        evidence = torch.cat((evidence, null), dim=2)
        padding_mask = torch.cat((
            padding_mask,
            torch.zeros(batch, num_query, 1, dtype=torch.bool,
                        device=padding_mask.device)), dim=2)
        diagnostics = {}
        for modality in ('image', 'lidar'):
            valid_mask = sampled.get(f'{modality}_valid_mask')
            if valid_mask is not None:
                diagnostics[f'diag_{modality}_valid_ratio'] = (
                    valid_mask.float().mean())
        return evidence, padding_mask, diagnostics

    @staticmethod
    def _gather_query_tensor(tensor, indices):
        index = indices
        for _ in range(tensor.ndim - 2):
            index = index.unsqueeze(-1)
        return tensor.gather(
            1, index.expand(*indices.shape, *tensor.shape[2:]))

    def _select_layer_queries(self, objectness, required_query_mask):
        budget = min(self.layer_query_budget, objectness.size(1))
        predicted = objectness.detach().topk(budget, dim=-1).indices
        if required_query_mask is None:
            return predicted, predicted
        if tuple(required_query_mask.shape) != tuple(objectness.shape):
            raise ValueError(
                'required_query_mask must match the initial query shape.')
        ranking = objectness.detach().masked_fill(required_query_mask, torch.inf)
        selected = ranking.topk(budget, dim=-1).indices
        return selected, predicted

    def forward(self, bev, bev_pos=None,
                query_features=None, initial_reference_controls=None,
                query_feature_sampler=None, required_query_mask=None,
                required_query_selector=None):
        if bev is not None:
            batch = bev.size(0)
        elif initial_reference_controls is not None:
            batch = initial_reference_controls[0].size(0)
        else:
            raise ValueError(
                'The decoder requires LiDAR BEV or initial curve controls.')
        query_pos = self.query_pos.weight.unsqueeze(0).expand(batch, -1, -1)
        query = self.query.weight.unsqueeze(0).expand(batch, -1, -1)
        source_query_indices = torch.arange(
            self.num_query, device=query.device).view(1, -1).expand(batch, -1)
        if required_query_mask is not None:
            required_query_mask = required_query_mask.to(
                device=query.device, dtype=torch.bool)
            if tuple(required_query_mask.shape) != (batch, self.num_query):
                raise ValueError(
                    'required_query_mask must have shape '
                    f'{(batch, self.num_query)}.')
        if query_features is not None:
            expected_shape = (batch, self.num_query, query.size(-1))
            if tuple(query_features.shape) != expected_shape:
                raise ValueError(
                    f'query_features must have shape {expected_shape}, got '
                    f'{tuple(query_features.shape)}.')
            query = query + query_features
        if initial_reference_controls is not None:
            reference_x, reference_z = initial_reference_controls
            expected_shape = (batch, self.num_query, self.num_control_points)
            if (tuple(reference_x.shape) != expected_shape
                    or tuple(reference_z.shape) != expected_shape):
                raise ValueError(
                    'initial_reference_controls must contain x/z tensors with '
                    f'shape {expected_shape}.')
        layer_predictions = []
        selected_query_indices = None
        layer0_topk_indices = None
        main_reference = initial_reference_controls
        main_visibility_reference = None
        for layer_idx, layer in enumerate(self.layers):
            current_queries = query.size(1)
            evidence, evidence_padding_mask, sampling_diagnostics = (
                self._sample_evidence(
                    main_reference, query_feature_sampler, batch, query.dtype,
                    query.device, self.norm(query)))
            query = layer(
                query, query_pos, evidence, evidence_padding_mask,
                None, current_queries)
            normalized_query = self.norm(query)
            predictions = self._decode_predictions(
                normalized_query, main_reference,
                main_visibility_reference,
                layer_idx=layer_idx)
            predictions.update(sampling_diagnostics)
            main_reference = (
                predictions['curve_x_controls'],
                predictions['curve_z_controls'])
            main_visibility_reference = (
                predictions['pred_vis_start'],
                predictions['pred_vis_end'])
            layer_predictions.append(predictions)
            should_prune = (
                layer_idx == 0 and layer_idx + 1 < self.num_layers
                and self.layer_query_budget is not None
                and self.layer_query_budget < current_queries)
            if should_prune:
                layer_required_mask = None
                if required_query_selector is not None:
                    layer_required_mask = required_query_selector(predictions)
                    layer_required_mask = layer_required_mask.to(
                        device=query.device, dtype=torch.bool)
                    if tuple(layer_required_mask.shape) != (
                            batch, current_queries):
                        raise ValueError(
                            'required_query_selector must return shape '
                            f'{(batch, current_queries)}.')
                if required_query_mask is not None:
                    layer_required_mask = (
                        required_query_mask
                        if layer_required_mask is None
                        else layer_required_mask | required_query_mask)
                selected, predicted = self._select_layer_queries(
                    predictions['pred_cls'], layer_required_mask)
                layer0_topk_indices = self._gather_query_tensor(
                    source_query_indices, predicted)
                source_query_indices = self._gather_query_tensor(
                    source_query_indices, selected)
                selected_query_indices = source_query_indices
                query = self._gather_query_tensor(query, selected)
                query_pos = self._gather_query_tensor(query_pos, selected)
                main_reference = tuple(
                    self._gather_query_tensor(value, selected)
                    for value in main_reference)
                main_visibility_reference = tuple(
                    self._gather_query_tensor(value, selected)
                    for value in main_visibility_reference)
        predictions = layer_predictions[-1]
        predictions['aux_predictions'] = layer_predictions[:-1]
        if selected_query_indices is not None:
            predictions['selected_query_indices'] = selected_query_indices
            predictions['layer0_topk_indices'] = layer0_topk_indices
        predictions['bev'] = bev
        return predictions
