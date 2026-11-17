"""Observed height/color boundaries for BEV road-shape extraction."""
import torch
import torch.nn.functional as F


ROAD_EVIDENCE_FIELDS = (
    'observed', 'log_density', 'minimum_z', 'mean_z', 'std_z', 'z_range',
    'red', 'green', 'blue', 'color_observed', 'color_fraction', 'log_intensity',
    'height_dx', 'height_dy', 'color_dx', 'color_dy')
ROAD_EVIDENCE_CHANNELS = len(ROAD_EVIDENCE_FIELDS)


def road_evidence(points, pc_range, height, width, rgb_start=5):
    """Rasterize painted points; empty cells never create boundary evidence.

    RGB is averaged only within 0.3 m of the lowest point in a cell. This is an
    input statistic, not a road/curb label. Heights are kept in ground meters.
    Coarser context and association are learned by the template extractor.
    """
    rasters = []
    for sample in points:
        lower, upper = sample.new_tensor(pc_range[:3]), sample.new_tensor(pc_range[3:])
        valid = (torch.isfinite(sample).all(-1)
                 & (sample[:, :3] >= lower).all(-1)
                 & (sample[:, :3] <= upper).all(-1))
        p = sample[valid].float()
        cells = height * width
        xy = (p[:, :2] - lower[:2]) / (upper[:2] - lower[:2])
        ix = (xy[:, 0] * width).long().clamp(0, width - 1)
        iy = (xy[:, 1] * height).long().clamp(0, height - 1)
        index = iy * width + ix

        def total(values):
            return values.new_zeros((cells, *values.shape[1:])).index_add_(0, index, values)

        count = total(p.new_ones(len(p)))
        seen = count > 0
        denominator = count.clamp_min(1)
        minimum = p.new_full((cells,), torch.inf)
        maximum = p.new_full((cells,), -torch.inf)
        minimum.scatter_reduce_(0, index, p[:, 2], reduce='amin', include_self=True)
        maximum.scatter_reduce_(0, index, p[:, 2], reduce='amax', include_self=True)
        minimum = torch.where(seen, minimum, 0)
        maximum = torch.where(seen, maximum, 0)
        mean = total(p[:, 2]) / denominator
        variance = (total(p[:, 2].square()) / denominator - mean.square()).clamp_min(0)
        std = torch.where(seen, (variance + 1e-6).sqrt() - .001, 0)
        rgb = p[:, rgb_start:rgb_start + 3]
        colored = (rgb.abs().sum(-1) > 0) & (p[:, 2] <= minimum[index] + .3)
        color_count = total(colored.to(p))
        color = total(rgb * colored[:, None]) / color_count.clamp_min(1)[:, None]
        fields = torch.stack((
            seen.to(p), torch.log1p(count) / 5, minimum, mean, std, maximum - minimum,
            *color.unbind(-1), (color_count > 0).to(p), color_count / denominator,
            total(torch.log1p(p[:, 3].clamp_min(0))) / denominator), 0)
        fields = fields.reshape(12, height, width)
        # Differences require observations on both sides; a missing cell is
        # not a zero-height surface or a black road marking.
        observed = fields[0]
        color_seen = fields[9]
        dx = (pc_range[3] - pc_range[0]) / width
        dy = (pc_range[4] - pc_range[1]) / height
        dzx = (fields[2, :, 1:] - fields[2, :, :-1]) / dx
        dzy = (fields[2, 1:] - fields[2, :-1]) / dy
        dcx = torch.linalg.vector_norm(fields[6:9, :, 1:] - fields[6:9, :, :-1], dim=0) / dx
        dcy = torch.linalg.vector_norm(fields[6:9, 1:] - fields[6:9, :-1], dim=0) / dy
        boundaries = torch.stack((
            F.pad(dzx * observed[:, 1:] * observed[:, :-1], (1, 0)),
            F.pad(dzy * observed[1:] * observed[:-1], (0, 0, 1, 0)),
            F.pad(dcx * color_seen[:, 1:] * color_seen[:, :-1], (1, 0)),
            F.pad(dcy * color_seen[1:] * color_seen[:-1], (0, 0, 1, 0))))
        rasters.append(torch.cat((fields, boundaries), 0))
    return torch.stack(rasters)
