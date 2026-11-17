"""Global curve reasoning should stabilize short observations without flattening roads."""
import torch

from LaneLoom.models.head.continuous_bev_shape_prior import fit_observed_cubic
from LaneLoom.models.head.guided_spatial_bev_shape_prior import GuidedSpatialBEVShapePrior
from LaneLoom.models.head.shape_proposal_expander import ShapeProposalExpander


def basis_at(y):
    t = y/100.
    return torch.stack(((1-t)**3,3*(1-t)**2*t,3*(1-t)*t*t,t**3),-1)


def test_predicted_curvature_stabilizes_short_noisy_observations_and_is_differentiable():
    y = torch.arange(1.,100.,2.)
    truth = torch.tensor([0.,2.,8.,16.])
    full = basis_at(y)@truth
    visible = y < 26
    values = torch.where(visible,full+.05*torch.sin(y*.9),0.)[None,None].requires_grad_()
    weights = visible.float()[None,None]
    prior = truth.diff().diff()[None,None].clone().requires_grad_()
    precision = torch.tensor([[1e-4]],requires_grad=True)
    free = fit_observed_cubic(values,weights,y,100.)
    guided = fit_observed_cubic(values,weights,y,100.,prior,precision)
    assert (guided[..., -1]-truth[-1]).abs() < (free[..., -1]-truth[-1]).abs()
    assert ((basis_at(y[visible])@guided[0,0])-full[visible]).abs().max() < .2
    guided.square().mean().backward()
    for gradient in (prior.grad,precision.grad,values.grad):
        assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    assert torch.count_nonzero(values.grad[..., ~visible])==0


def test_curvature_prior_preserves_affine_heading_and_translation():
    y = torch.arange(1.,100.,2.)
    values = (3.+.06*y)[None,None]
    weights = torch.ones_like(values)
    precision = torch.ones(1,1)  # Strong prior: a control-norm penalty would flatten this line.
    prior = torch.zeros(1,1,2)
    guided = fit_observed_cubic(values,weights,y,100.,prior,precision)
    torch.testing.assert_close(guided[0,0],torch.tensor([3.,5.,7.,9.]),atol=1e-4,rtol=0)
    shifted = fit_observed_cubic(values+7.,weights,y,100.,prior,precision)
    torch.testing.assert_close(shifted,guided+7.,atol=1e-4,rtol=0)


def test_global_prior_learns_from_bev_without_extra_queries_and_cannot_invent_empty_clouds():
    torch.manual_seed(31)
    model = GuidedSpatialBEVShapePrior(lidar_in_channels=4,hidden_channels=16,
        anchor_y_steps=[5.,20.,50.,100.],max_templates=5,longitudinal_rows=8,
        spatial_layers=1,num_heads=4,geometric_readout=True,curvature_prior_strength=1e-4)
    features = torch.randn(2,4,16,24)
    occupancy = torch.ones(2,1,16,24,dtype=torch.bool);occupancy[1]=False
    output = model(dict(features=features,occupancy=occupancy))
    assert output['lane_curvature_prior'].shape==(2,5,2)
    assert output['lane_curvature_precision'].shape==(2,5)
    assert torch.all(output['lane_curvature_precision']>0)
    assert torch.isfinite(output['shape_points']).all()
    output['shape_points'].square().mean().backward()
    assert model.curvature_head.weight.grad.abs().sum()>0
    assert model.curvature_precision_head.weight.grad.abs().sum()>0
    assert torch.count_nonzero(output['shape_points'][1,...,(0,2)])==0
    expander = ShapeProposalExpander(pc_range=[-25,0,-5,25,100,5],
        anchor_y_steps=[5.,20.,50.,100.],num_x_seeds=101,template_feature_channels=16,embed_dims=16)
    assert expander(output)['curve_x_controls'].shape==(2,505,4)
    with torch.no_grad():
        model.curvature_head.weight.normal_(std=.01)
        a = model(dict(features=features[:1],occupancy=occupancy[:1]))
        b = model(dict(features=features[:1]+torch.randn_like(features[:1]),occupancy=occupancy[:1]))
        assert not torch.allclose(a['lane_curvature_prior'],b['lane_curvature_prior'])
