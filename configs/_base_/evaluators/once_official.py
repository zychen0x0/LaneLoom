"""Released ONCE-3DLanes validation protocol."""

evaluation_protocol = 'once_official'
score_threshold = 0.5
score_thresholds = [round(0.10 + 0.05 * index, 2) for index in range(18)]
best_metric = 'ONCEOfficial/F1'

once_official_evaluator = dict(
    type='ONCEOfficialMetric',
    prefix='ONCEOfficial',
    score_threshold=score_threshold,
    score_thresholds=score_thresholds,
    visibility_threshold=0.5,
    distance_threshold=0.3,
    collect_device='cpu')

val_evaluator = once_official_evaluator
test_evaluator = once_official_evaluator
