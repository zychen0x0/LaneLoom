"""Use a predicted lane-position/height field to guide spatial template reading."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

from .spatial_bev_shape_prior import SpatialBEVShapePrior
from .continuous_bev_shape_prior import fit_observed_cubic
from ..backbones.road_evidence import ROAD_EVIDENCE_CHANNELS


@MODELS.register_module()
class GuidedSpatialBEVShapePrior(SpatialBEVShapePrior):
    """Five relative cubics, with explicit intermediate lane evidence.

    The field is multi-label: several lanes may occupy the same BEV row.
    It is not a curb detector, nor a five-instance lane classifier. All field
    predictions come from BEV alone, including at inference. An auxiliary
    training loss may use raw visible lane annotations after prediction.

    geometric_readout fits predicted row coordinates into cubics, keeping
    lateral seed translation and an absolute (potentially nonzero) z origin.
    use_road_evidence optionally concatenates native-grid point statistics
    before spatial encoding. These are measurements, not curb pseudo-labels;
    they do not add templates, seeds, or another supervision objective.
    A positive curvature_prior_strength enables global curvature reasoning
    when a short observed span leaves the cubic fit poorly constrained.
    """

    def __init__(self, geometric_readout=False, pc_range=(-25, 0, -5, 25, 100, 5),
                 lane_field_mode='balanced_sigmoid', row_query_references=False,
                 row_query_window_m=1.5, use_road_evidence=False,
                 curvature_prior_strength=0., curvature_prior_scale=20., **kwargs):
        super().__init__(**kwargs)
        self.geometric_readout = bool(geometric_readout)
        self.curvature_prior_strength = float(curvature_prior_strength)
        self.curvature_prior_scale = float(curvature_prior_scale)
        if (self.curvature_prior_strength < 0 or self.curvature_prior_scale <= 0
                or (self.curvature_prior_strength and not self.geometric_readout)):
            raise ValueError('A positive curvature prior requires geometric readout and a positive scale')
        self.use_road_evidence = bool(use_road_evidence)
        if self.use_road_evidence:
            # Keep the measured grid at its native resolution. Interpolated
            # backbone features cannot restore a lost curb/color transition.
            conv = self.feature_encoder[0]
            # Preserve every shared random initialization and the initial
            # function. The new evidence path starts neutral and learns from
            # ordinary training gradients; it is not a new random warm start.
            with torch.random.fork_rng(devices=[]):
                extended = nn.Conv2d(conv.in_channels + ROAD_EVIDENCE_CHANNELS,
                                     conv.out_channels, 3, padding=1, bias=False)
            with torch.no_grad():
                extended.weight[:, :conv.in_channels].copy_(conv.weight)
                extended.weight[:, conv.in_channels:].zero_()
            self.feature_encoder[0] = extended
        if lane_field_mode not in ('balanced_sigmoid', 'row_distribution'):
            raise ValueError('Unsupported lane-field probability model')
        self.lane_field_mode = lane_field_mode
        self.row_query_references = bool(row_query_references)
        self.row_query_window_m = float(row_query_window_m)
        if self.row_query_window_m <= 0:
            raise ValueError('Row-query window must be positive')
        if self.row_query_references and (lane_field_mode != 'row_distribution' or not self.geometric_readout):
            raise ValueError('Row references require a distribution field and geometric readout')
        self.pc_range = tuple(float(value) for value in pc_range)
        if self.pc_range[1] != 0 or self.pc_range[4] != float(self.anchor_y_steps[-1]):
            raise ValueError('Lane-field rows must span the same 0..length interval as the curves')
        channels = self.shape_tokens.embedding_dim
        self.lane_field_head = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 16, (7, 3), padding=(9, 1),
                      dilation=(3, 1), groups=16, bias=False),
            nn.GELU(), nn.Conv2d(16, 2, 1))
        self.field_value = nn.Conv2d(2, channels, 1, bias=False)
        if self.geometric_readout:
            # Read positions, then fit geometry. Do not regress six unconstrained
            # slope parameters from a pooled latent vector in this ablation.
            del self.control_queries, self.control_attention, self.control_norm, self.slope_head
            self.point_residual = nn.Linear(channels, 2)
            nn.init.zeros_(self.point_residual.weight)
            nn.init.zeros_(self.point_residual.bias)
            if self.curvature_prior_strength:
                self.curvature_queries = nn.Embedding(2, channels)
                nn.init.normal_(self.curvature_queries.weight, std=.02)
                self.curvature_attention = nn.MultiheadAttention(
                    channels, self.row_attention.num_heads, batch_first=True)
                self.curvature_norm = nn.LayerNorm(channels)
                self.curvature_head = nn.Linear(channels, 1)
                self.curvature_precision_head = nn.Linear(channels+2, 1)
                for head in (self.curvature_head, self.curvature_precision_head):
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)

    def forward(self, pts_feats, img_feats=None, img_metas=None):
        if pts_feats is None:
            raise ValueError('GuidedSpatialBEVShapePrior requires LiDAR BEV features')
        feature = self._feature(pts_feats)
        encoder_input = feature
        if self.use_road_evidence:
            road = pts_feats.get('road_evidence') if isinstance(pts_feats, dict) else None
            if (road is None or road.ndim != 4
                    or road.shape != (feature.size(0), ROAD_EVIDENCE_CHANNELS, *feature.shape[-2:])):
                raise ValueError('Road evidence must share the BEV batch/grid and have 16 channels')
            road = road.to(feature)
            # Unobserved cells are unknown, not zero-height/black surfaces.
            road = torch.where(road[:, :1] > 0, road, 0)
            encoder_input = torch.cat((feature, road), 1)
        encoded = self.feature_encoder(encoder_input)
        field = self.lane_field_head(encoded)
        logits, height = field[:, :1], field[:, 1:]
        # Predict at the original BEV resolution. Only the attention's row
        # organization is pooled; the supervised field stays at full resolution.
        probability = (logits.softmax(-1) if self.lane_field_mode == 'row_distribution'
                       else logits.sigmoid())
        values = torch.cat((probability, height), 1)
        memory, observed = self._row_memory(encoded + self.field_value(values), pts_feats)
        row_logits = F.adaptive_avg_pool2d(logits, memory.shape[1:3])[:, 0]
        if self.lane_field_mode == 'row_distribution':
            attention_bias = self._position_attention_bias(row_logits)
        else:
            attention_bias = F.logsigmoid(row_logits)
        if self.geometric_readout:
            evidence, attention = self._read_rows(memory, observed, attention_bias, return_weights=True)
            row_height = F.adaptive_avg_pool2d(height, memory.shape[1:3])[:, 0]
            controls, trace = self._geometric_controls(evidence, attention, row_logits, row_height, observed)
            result = self._format_prior(controls)
            result.update(trace)
        else:
            evidence = self._read_rows(memory, observed, attention_bias)
            result = self._format_prior(self._controls_from_evidence(evidence, feature, observed))
        result.update(lane_field_logits=logits, lane_height_field=height,
                      lane_field_mode=self.lane_field_mode)
        return result

    def _position_attention_bias(self, logits):
        """Separate spatial modes with reference points from predicted density.

        References are quantiles of each predicted row, not GT coordinates or
        fixed road-shape anchors. They initialize five parallel reads so an
        ambiguous field need not be averaged into the same trajectory five
        times. Content attention and longitudinal context remain trainable.
        """
        width = logits.size(-1)
        log_probability = logits.log_softmax(-1)
        bias = log_probability + logits.new_tensor(width).log()
        if not self.row_query_references:
            return bias
        with torch.no_grad():
            cdf = log_probability.detach().exp().cumsum(-1).contiguous()
            levels = (torch.arange(self.max_templates, device=logits.device,
                                   dtype=logits.dtype)+.5)/self.max_templates
            levels = levels.expand(*logits.shape[:-1], -1).contiguous()
            index = torch.searchsorted(cdf, levels).clamp_max(width-1)
            cell_width = (self.pc_range[3]-self.pc_range[0])/width
            reference_x = self.pc_range[0]+(index.to(logits.dtype)+.5)*cell_width
        column_x = self.pc_range[0]+(torch.arange(width, device=logits.device,
                                   dtype=logits.dtype)+.5)*cell_width
        locality = -.5*((column_x-reference_x[..., None])/self.row_query_window_m).square()
        return bias[..., None, :] + locality

    def _geometric_controls(self, evidence, attention, logits, height, observed):
        batch, rows, templates, width_plus_null = attention.shape
        width = width_plus_null-1
        probability = attention[..., :width]
        probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-6)
        column_x = (self.pc_range[0] + (torch.arange(width, device=attention.device,
                     dtype=attention.dtype)+.5) * (self.pc_range[3]-self.pc_range[0])/width)
        row_x = (probability * column_x).sum(-1).permute(0, 2, 1)
        row_z = (probability * height[:, :, None]).sum(-1).permute(0, 2, 1)
        # Longitudinal context can correct the measured positions, instead of
        # discarding them when decoding the final curve.
        residual = self.point_residual(evidence).reshape(batch, templates, rows, 2)
        row_x = row_x + residual[..., 0]
        row_z = row_z + residual[..., 1]
        field_probability = (logits.softmax(-1) if self.lane_field_mode == 'row_distribution'
                             else logits.sigmoid())
        weights = (probability * field_probability[:, :, None]).sum(-1).permute(0, 2, 1)
        weights = weights * observed.any(-1)[:, None]
        if self.lane_field_mode == 'row_distribution':
            # A row probability sums to one, rather than W independent values.
            # Keep the cubic fit invariant to this change in confidence scale.
            weights = weights / weights.mean(-1, keepdim=True).clamp_min(1e-6)
        length = self.anchor_y_steps[-1].to(attention)
        y = (torch.arange(rows, device=attention.device, dtype=attention.dtype)+.5)*length/rows
        prior_trace = {}
        if self.curvature_prior_strength:
            prior, precision = self._read_curvature_prior(evidence, weights, y/length)
            x = fit_observed_cubic(row_x, weights, y, length, prior, precision)
            prior_trace = dict(lane_curvature_prior=prior, lane_curvature_precision=precision)
        else:
            x = fit_observed_cubic(row_x, weights, y, length)
        z = fit_observed_cubic(row_z, weights, y, length)
        origin_x = x[..., :1]
        x = x-origin_x
        # Height is an absolute road coordinate: a fitted nonzero intercept
        # must not be subtracted from the entire predicted road surface.
        valid = observed.flatten(1).any(-1)
        x = x * valid[:, None, None]
        z = z * valid[:, None, None]
        lateral, vertical = x.diff(dim=-1)*3/length, z.diff(dim=-1)*3/length
        tokens = evidence.mean(1).reshape(batch, templates, -1) * valid[:, None, None]
        controls = (x, z, lateral, vertical, torch.rad2deg(torch.atan(lateral)),
                    torch.rad2deg(torch.atan(vertical)), tokens, valid)
        return controls, dict(lane_row_x=row_x, lane_row_z=row_z,
                              lane_row_x_origin=origin_x, lane_row_weights=weights, **prior_trace)

    def _read_curvature_prior(self, evidence, weights, normalized_y):
        """Infer near/far curvature and trust from ordered BEV evidence alone.

        The prior fills poorly observed polynomial directions. Translation and
        linear heading remain determined by measurements; there is no new GT
        curvature loss and no forced zero-curvature assumption after learning.
        Prior values are second control differences in meters, not curvature
        in inverse meters; the cubic's second derivative scales by 6/length².
        """
        batch, templates, _ = weights.shape
        position = self.position_encoder(evidence.new_tensor([[0., -1.], [0., 1.]]))
        query = (self.curvature_queries.weight+position)[None].expand(batch*templates, -1, -1)
        controls = self.curvature_norm(query + self.curvature_attention(
            query, evidence, evidence, need_weights=False)[0])
        prior = self.curvature_head(controls).reshape(batch, templates, 2) * self.curvature_prior_scale
        total = weights.sum(-1).clamp_min(1e-6)
        mean_y = (weights*normalized_y).sum(-1)/total
        variance = (weights*(normalized_y-mean_y[..., None]).square()).sum(-1)/total
        support = torch.stack((weights.mean(-1), 4*variance), -1)
        context = controls.mean(1).reshape(batch, templates, -1)
        log_scale = self.curvature_precision_head(torch.cat((context, support), -1)).squeeze(-1)
        # Learn confidence over a bounded 100x range in either direction.
        precision = self.curvature_prior_strength * torch.exp(4.605170186*log_scale.tanh())
        return prior, precision
