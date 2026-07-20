"""One-to-one Hungarian assignment for fixed 3D lane queries."""

import torch
from mmdet3d.registry import TASK_UTILS

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


def batched_hungarian_assignment(cost):
    """Return one column index per matched row, with -1 for unmatched rows."""
    cost_cpu = cost.detach().cpu()
    assignments = torch.full(cost.shape[:2], -1, dtype=torch.long)
    for batch_idx in range(cost.size(0)):
        finite = torch.isfinite(cost_cpu[batch_idx])
        row_indices = torch.where(finite.any(dim=-1))[0]
        column_indices = torch.where(finite.any(dim=0))[0]
        if row_indices.numel() == 0 or column_indices.numel() == 0:
            continue
        sample_cost = cost_cpu[batch_idx][
            row_indices][:, column_indices]
        if linear_sum_assignment is not None:
            local_rows, local_columns = linear_sum_assignment(
                sample_cost.numpy())
            local_rows = torch.as_tensor(local_rows)
            local_columns = torch.as_tensor(local_columns)
        else:  # pragma: no cover
            remaining = sample_cost.clone()
            local_rows, local_columns = [], []
            for _ in range(min(sample_cost.shape)):
                flat_index = remaining.argmin()
                row_idx = flat_index // remaining.size(1)
                column_idx = flat_index % remaining.size(1)
                local_rows.append(row_idx)
                local_columns.append(column_idx)
                remaining[row_idx] = torch.inf
                remaining[:, column_idx] = torch.inf
            local_rows = torch.stack(local_rows)
            local_columns = torch.stack(local_columns)
        assignments[
            batch_idx, row_indices[local_rows]] = column_indices[local_columns]
    return assignments.to(cost.device)


@TASK_UTILS.register_module()
class LaneLoomHungarianAssigner:
    """Assign every visible GT to one exclusive query by 3D geometry."""

    def assign_batch(self, anchor_x, anchor_z, gt_x, gt_z, gt_visibility):
        """Match a batch with one GPU-to-CPU transfer for SciPy."""
        visibility = gt_visibility.bool()
        visible_count = visibility.sum(dim=-1).clamp_min(1)
        distance = torch.linalg.vector_norm(torch.stack((
            anchor_x[:, :, None] - gt_x[:, None],
            anchor_z[:, :, None] - gt_z[:, None]), dim=-1), dim=-1)
        cost = (
            distance * visibility.unsqueeze(1)).sum(dim=-1)
        cost = cost / visible_count.unsqueeze(1)
        valid_gt = visibility.any(dim=-1)
        cost = cost.masked_fill(~valid_gt.unsqueeze(1), torch.inf)
        return batched_hungarian_assignment(cost)

    def assign(self, anchor_x, anchor_z, gt_x, gt_z, gt_visibility):
        return self.assign_batch(
            anchor_x.unsqueeze(0), anchor_z.unsqueeze(0),
            gt_x.unsqueeze(0), gt_z.unsqueeze(0),
            gt_visibility.unsqueeze(0))[0]
