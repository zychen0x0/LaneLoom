# Copyright (c) 2022 The OpenLane Dataset Authors.
# Licensed under the Apache License, Version 2.0.
# Modified for LaneLoom's in-process MMEngine integration.
"""OpenLane's minimum-cost-flow lane assignment."""

import numpy as np
from ortools.graph import pywrapgraph


def solve_min_cost_flow(adj_mat, cost_mat):
    """Return ``(gt_index, pred_index, integer_cost)`` assignments."""
    gt_count, pred_count = adj_mat.shape
    if gt_count == 0 or pred_count == 0:
        return []

    nonempty_gt = int(np.sum(np.sum(adj_mat, axis=1) > 0))
    nonempty_pred = int(np.sum(np.sum(adj_mat, axis=0) > 0))
    flow_count = min(nonempty_gt, nonempty_pred)
    if flow_count == 0:
        return []

    source = 0
    first_pred = gt_count + 1
    sink = gt_count + pred_count + 1
    solver = pywrapgraph.SimpleMinCostFlow()

    for gt_index in range(gt_count):
        solver.AddArcWithCapacityAndUnitCost(
            source, gt_index + 1, 1, 0)
        for pred_index in range(pred_count):
            solver.AddArcWithCapacityAndUnitCost(
                gt_index + 1,
                first_pred + pred_index,
                int(adj_mat[gt_index, pred_index]),
                int(cost_mat[gt_index, pred_index]))
    for pred_index in range(pred_count):
        solver.AddArcWithCapacityAndUnitCost(
            first_pred + pred_index, sink, 1, 0)

    solver.SetNodeSupply(source, flow_count)
    for node in range(1, sink):
        solver.SetNodeSupply(node, 0)
    solver.SetNodeSupply(sink, -flow_count)

    if solver.Solve() != solver.OPTIMAL:
        raise RuntimeError('OpenLane minimum-cost-flow assignment failed.')

    matches = []
    for arc in range(solver.NumArcs()):
        tail = solver.Tail(arc)
        head = solver.Head(arc)
        if tail == source or head == sink or solver.Flow(arc) == 0:
            continue
        matches.append((
            tail - 1,
            head - first_pred,
            solver.UnitCost(arc),
        ))
    return matches
