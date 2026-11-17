import numpy as np

from LaneLoom.tools.audit_q0_representation import fit_relative_cubic, dictionary_coverage


def test_feasible_probe_recovers_an_independent_known_curve():
    y = np.linspace(5., 95., 50)
    points = np.stack((2. + .02 * y + .001 * y**2, y, .0001 * y**2), -1)
    fit = fit_relative_cubic(points)
    assert fit['p80_xz_m'] < 1e-7
    assert fit['x_controls'][0] == fit['z_controls'][0] == 0.
    assert fit['seed'] in np.linspace(-25., 25., 101)


def test_one_shape_covers_multiple_parallel_gt_with_distinct_seed_queries():
    y = np.linspace(5., 95., 50)
    lanes = [dict(json_index=i, points=np.stack((offset + .02 * y, y, .001 * y), -1))
             for i, offset in enumerate((-5., -2., 1., 4., 7., 10.))]
    fits = [fit_relative_cubic(lane['points']) for lane in lanes]
    result = dictionary_coverage(fits, lanes)
    assert result['covered'] == 6
    ids = {(lane['source_template'], lane['seed']) for lane in result['lanes']}
    assert len(ids) == 6


def test_height_origin_is_an_independent_geometric_degree_of_freedom():
    y = np.linspace(5., 95., 50)
    points = np.stack((2. + .02 * y, y, 1. + .00001 * y**3), -1)
    fixed = fit_relative_cubic(points)
    free = fit_relative_cubic(points, free_height_origin=True)
    assert free['p80_xz_m'] < 1e-6
    assert abs(free['z_controls'][0] - 1.) < 1e-6
    assert fixed['p80_xz_m'] > 1e-4


def test_stability_probe_does_not_mistake_extreme_short_support_extrapolation_for_a_road():
    y = np.arange(10., 14.)
    points = np.stack((.1*y*(y-10)*(y-13), y, np.zeros_like(y)), -1)
    unconstrained = fit_relative_cubic(points, free_height_origin=True)
    stable = fit_relative_cubic(points, free_height_origin=True,
                                lateral_control_bound=100., height_control_bound=20.)
    assert unconstrained['p80_xz_m'] < 1e-6
    assert max(abs(v) for v in unconstrained['x_controls']) > 1000
    assert max(abs(v) for v in stable['x_controls']) <= 100.00001
    assert max(abs(v) for v in stable['z_controls']) <= 20.00001
    assert stable['p80_xz_m'] > unconstrained['p80_xz_m'] + .1
    y = np.linspace(5., 95., 50)
    points = np.stack((2.+.02*y+.001*y*y, y, .5+.0001*y*y), -1)
    known = fit_relative_cubic(points, free_height_origin=True,
                               lateral_control_bound=100., height_control_bound=20.)
    assert known['p80_xz_m'] < 1e-6
