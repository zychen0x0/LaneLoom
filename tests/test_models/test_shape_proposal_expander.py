import torch

from LaneLoom.models.head.shape_proposal_expander import (
    ShapeProposalExpander,
)


def test_five_templates_expand_to_505_queries_with_exact_endpoint_seeds():
    expander = ShapeProposalExpander(
        pc_range=[-25., 0., -5., 25., 100., 5.],
        anchor_y_steps=[2. * index for index in range(1, 51)],
        num_x_seeds=101,
        template_feature_channels=8,
        embed_dims=16,
        q0_distance_threshold=.3,
        q0_match_ratio=.8,
    )
    prior = dict(
        shape_x_controls=torch.zeros(2, 5, 4),
        shape_z_controls=torch.zeros(2, 5, 4),
        shape_query_features=torch.randn(2, 5, 8),
    )

    output = expander(prior)

    assert output['curve_x_controls'].shape == (2, 505, 4)
    assert output['curve_z_controls'].shape == (2, 505, 4)
    assert output['query_features'].shape == (2, 505, 16)
    assert expander.x_seeds.numel() == 101
    assert expander.x_seeds[0].item() == -25.
    assert expander.x_seeds[-1].item() == 25.
    assert torch.all(torch.diff(expander.x_seeds) == .5)
    assert output['curve_x_controls'][0, 0, 0].item() == -25.
    assert output['curve_x_controls'][0, 504, 0].item() == 25.


def test_q0_geometry_does_not_use_gt_when_lidar_sample_is_empty():
    expander = ShapeProposalExpander(
        pc_range=[-25., 0., -5., 25., 100., 5.],
        anchor_y_steps=[2. * index for index in range(1, 51)],
        num_x_seeds=101,
        template_feature_channels=8,
        embed_dims=16,
        q0_distance_threshold=.3,
        q0_match_ratio=.8,
    )
    prior = dict(
        shape_x_controls=torch.zeros(1, 5, 4),
        shape_z_controls=torch.zeros(1, 5, 4),
        shape_query_features=torch.randn(1, 5, 8),
        from_lidar=torch.tensor([False]),
    )
    expanded = dict(prior)
    expanded.update(expander(prior))
    lanes = torch.zeros(1, 1, 50, 3)
    lanes[..., 1] = expander.anchor_y_steps
    visibility = torch.ones(1, 1, 50)

    losses = expander.loss(expanded, lanes, visibility)

    assert losses['loss_q0_geometry'] == 0
    assert losses['diag_q0_gt_recall'] == 0
