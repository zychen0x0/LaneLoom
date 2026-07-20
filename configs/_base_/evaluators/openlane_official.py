"""Official OpenLane validation and final-test evaluation."""

evaluation_protocol = 'official'
score_threshold = 0.5
best_metric = 'OpenLaneOfficial/F1'

official_evaluator = dict(
    type='OpenLaneOfficialMetric',
    prefix='OpenLaneOfficial',
    score_threshold=score_threshold,
    visibility_threshold=0.5,
    use_temporal_gt_supplement=False,
    collect_device='cpu')

val_evaluator = official_evaluator
test_evaluator = official_evaluator
