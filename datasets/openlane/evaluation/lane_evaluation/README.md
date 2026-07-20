# OpenLane 3D lane evaluation

This module is an in-process adaptation of the official OpenLane 3D lane
evaluator from commit `ec98fda7cb21ecf51ffdf70c37c411076985dbd6`:

<https://github.com/OpenDriveLab/OpenLane/tree/ec98fda7cb21ecf51ffdf70c37c411076985dbd6/eval/LANE_evaluation/lane3d>

The official geometry, 100-point resampling, 1.5 m distance threshold,
0.75 visibility-overlap thresholds, category rule, and minimum-cost-flow
assignment are retained. The command-line JSON loading layer is replaced by a
small Python interface so MMEngine can evaluate each collected sample without
launching a second process.

The original OpenLane source is licensed under Apache License 2.0. See
`LICENSE` in this directory.
