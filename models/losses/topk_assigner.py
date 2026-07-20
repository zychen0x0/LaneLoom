"""Top-k assignment for fixed 3D lane anchors."""

import torch
from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class TopkAssigner:
    """Assign each GT to its nearest anchors without sharing an anchor."""

    def __init__(self, pos_k=2):
        self.pos_k = int(pos_k)
        if self.pos_k <= 0:
            raise ValueError('pos_k must be positive.')

    def assign(self, anchor_x, anchor_z, gt_x, gt_z, gt_visibility):
        """Return one GT index per anchor, with -1 denoting background."""
        assigned_gt = torch.full(
            (anchor_x.size(0),), -1, dtype=torch.long,
            device=anchor_x.device)
        valid_gt = torch.where(gt_visibility.bool().any(dim=-1))[0]
        if valid_gt.numel() == 0:
            return assigned_gt

        gt_x = gt_x[valid_gt]
        gt_z = gt_z[valid_gt]
        visibility = gt_visibility[valid_gt].bool()
        visible_count = visibility.sum(dim=-1).clamp_min(1)
        distance = torch.linalg.vector_norm(torch.stack((
            anchor_x[:, None] - gt_x[None],
            anchor_z[:, None] - gt_z[None]), dim=-1), dim=-1)
        distance = (distance * visibility[None]).sum(dim=-1) / visible_count[None]

        # An anchor first belongs to its nearest GT.  Each GT then keeps its
        # nearest pos_k anchors, matching Anchor3DLane's ownership rule.
        owner = distance.argmin(dim=1)
        for local_gt, original_gt in enumerate(valid_gt):
            candidates = torch.where(owner == local_gt)[0]
            if candidates.numel() == 0:
                continue
            count = min(self.pos_k, candidates.numel())
            nearest = distance[candidates, local_gt].topk(
                count, largest=False).indices
            assigned_gt[candidates[nearest]] = original_gt
        return assigned_gt
