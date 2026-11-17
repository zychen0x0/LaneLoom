"""Official validation cannot be confused with the old training-scene holdout."""
import hashlib
import pytest
from mmengine.config import Config

from LaneLoom.tools.q0_scenarios import read_validation_scenarios, HARD_GROUPS
from LaneLoom.tools.evaluate_q0_prior import validate_validation_training_manifest, coverage_goal_report


def test_scene_list_selects_validation_only_and_stops_before_duplicate_notes(tmp_path):
    scenes = tmp_path / 'scenes.md'
    path = 'data/openlane/images/validation/segment-test/123.jpg'
    scenes.write_text('## 训练\n弯道\ndata/openlane/images/training/segment-train/1.jpg\n'
                      '## 验证 ########\n弯道\ndata/openlane/images/validation/segment-test\n'
                      + path + '\n---\n' + path)
    assert read_validation_scenarios(scenes) == [dict(group='curve', segment='segment-test', frame='123', image=path)]
    scenes.write_text('## 验证\n弯道\n' + path + '\n' + path)
    with pytest.raises(ValueError, match='Duplicate'):
        read_validation_scenarios(scenes)
    scenes.write_text('## 验证\n弯道\ndata/openlane/images/training/segment-test/123.jpg')
    with pytest.raises(ValueError, match='Unexpected'):
        read_validation_scenarios(scenes)


def test_full_training_configs_restore_training_scenes_without_changing_data_between_heads():
    baseline = Config.fromfile('configs/ablations/laneloom_openlane_q0_fulltrain.py')
    new = Config.fromfile('configs/ablations/laneloom_openlane_q0_fulltrain_field_geometry.py')
    assert baseline.q0_evaluation_split == new.q0_evaluation_split == 'validation'
    assert baseline.q0_excluded_segments == new.q0_excluded_segments == []
    assert baseline.train_dataloader == new.train_dataloader
    assert baseline.train_dataloader.dataset.dataset_cfg.excluded_segments == []
    assert baseline.q0_target_frames == new.q0_target_frames == {}
    assert baseline.optim_wrapper == new.optim_wrapper
    selected = read_validation_scenarios(baseline.q0_validation_scenes)
    assert len(selected) == 47
    assert sum(s['group'] in HARD_GROUPS for s in selected) == 26


def test_validation_audit_checks_actual_training_manifest_and_its_digest(tmp_path):
    scenes = [dict(segment='segment-val', frame='1')]
    manifest = tmp_path / 'training_samples.txt'
    def check(payload):
        manifest.write_bytes(payload)
        protocol = dict(training_labels=1, training_labels_sha256=hashlib.sha256(payload).hexdigest())
        validate_validation_training_manifest(tmp_path / 'step.pth', protocol, scenes)
    check(b'training/segment-train/2.json\n')
    for leaked in [b'validation/segment-val/1.json\n', b'training/segment-val/2.json\n']:
        with pytest.raises(ValueError, match='leaked'):
            check(leaked)
    with pytest.raises(ValueError, match='does not match'):
        validate_validation_training_manifest(tmp_path/'step.pth',
            dict(training_labels=1, training_labels_sha256='wrong'), scenes)


def test_validation_goal_uses_new_denominator_and_does_not_reuse_74():
    keys = ('paired_xz_p80', 'paired_xz_p75_03', 'paired_xz_p80_05', 'paired_xz_p75_05')
    samples = []
    remaining = 240
    for i in range(47):
        gt = 5 if i < 46 else 36
        covered = min(gt, remaining)
        remaining -= covered
        samples.append(dict(frame=str(i), raw_gt=gt,
            baseline={k:dict(gt=gt, covered=0) for k in keys},
            spatial={k:dict(gt=gt, covered=covered) for k in keys}))
    expected = [str(i) for i in range(47)]
    assert coverage_goal_report(dict(samples=samples), expected, expected_total=266)['passed']
    samples[0]['spatial']['paired_xz_p80']['covered'] -= 1
    assert not coverage_goal_report(dict(samples=samples), expected, expected_total=266)['passed']
    with pytest.raises(ValueError):
        coverage_goal_report(dict(samples=samples[:-1]), expected, expected_total=266)
