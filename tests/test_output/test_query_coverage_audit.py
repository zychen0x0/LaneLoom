"""Coverage diagnosis must inspect all queries on GT support, regardless of score."""
import torch

from LaneLoom.tools.audit_query_coverage import distances


def test_full_query_pool_finds_low_score_candidate_and_ignores_invisible_gt():
    gt = torch.tensor([[[1., 5., 0.], [2., 10., 0.], [999., 15., 999.]]])
    visibility = torch.tensor([[True, True, False]])
    stage = dict(x=torch.tensor([[[10., 20., 0.], [1., 2., -999.]]]),
                 z=torch.zeros(1, 2, 3), scores=torch.tensor([[.9, .01]]))
    _, ratio, p80 = distances(stage, gt, visibility)
    assert ratio[:, 0].tolist() == [0., 1.]
    assert p80[:, 0].argmin().item() == 1
    assert p80[1, 0] == 0
