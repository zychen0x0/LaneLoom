"""DV3DLane-style validation with official OpenLane final testing."""

evaluation_protocol = 'dv3dlane'
score_threshold = 0.5
best_metric = 'DV3DLane/F1'

dv3dlane_evaluator = dict(
    type='OpenLaneAnchorMetric',
    prefix='DV3DLane',
    score_threshold=score_threshold,
    visibility_threshold=0.5,
    distance_threshold=0.5,
    match_ratio_threshold=0.75,
    near_distance=40.0,
    use_temporal_gt_supplement=False,
    collect_device='cpu')

official_evaluator = dict(
    type='OpenLaneOfficialMetric',
    prefix='OpenLaneOfficial',
    score_threshold=score_threshold,
    visibility_threshold=0.5,
    use_temporal_gt_supplement=False,
    collect_device='cpu')

val_evaluator = dv3dlane_evaluator
test_evaluator = official_evaluator
