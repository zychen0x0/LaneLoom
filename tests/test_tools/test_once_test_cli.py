from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace


TEST_PATH = Path(__file__).resolve().parents[2] / 'tools/test_once.py'
SPEC = spec_from_file_location('laneloom_once_test_cli', TEST_PATH)
TEST_MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(TEST_MODULE)


def test_once_default_evaluation_dir_is_next_to_checkpoint(tmp_path):
    checkpoint = tmp_path / 'experiment/checkpoints/best.pth'
    assert TEST_MODULE.resolve_evaluation_dir(checkpoint, None) == (
        tmp_path / 'experiment/once_evaluation')


def test_once_summary_keeps_primary_and_best_threshold_results(tmp_path):
    checkpoint = tmp_path / 'experiment/checkpoints/best.pth'
    metrics = {
        'ONCEOfficial/F1': 0.75,
        'ONCEOfficial/Precision': 0.8,
        'ONCEOfficial/Recall': 0.705882,
        'ONCEOfficial/CDE': 0.06,
        'ONCEOfficial/score_threshold': 0.5,
        'ONCEOfficial/best_F1': 0.78,
        'ONCEOfficial/best_Precision': 0.82,
        'ONCEOfficial/best_Recall': 0.7437,
        'ONCEOfficial/best_CDE': 0.058,
        'ONCEOfficial/best_score_threshold': 0.6,
    }
    payload = TEST_MODULE.write_summary(tmp_path, checkpoint, metrics)

    assert payload['metrics']['F1'] == 0.75
    assert payload['metrics']['best_score_threshold'] == 0.6
    assert json.loads(
        (tmp_path / 'once_evaluation_results.json').read_text()) == payload


def test_once_main_does_not_duplicate_runner_metric_summary(
        tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / 'best.pth'
    checkpoint.write_bytes(b'checkpoint')
    cfg = SimpleNamespace(
        get=lambda key, default=None: (
            'once' if key == 'dataset_name' else default),
        test_evaluator=SimpleNamespace())
    args = SimpleNamespace(
        config=str(tmp_path / 'once.py'),
        checkpoint=str(checkpoint),
        work_dir=str(tmp_path / 'evaluation'),
        prediction_dir=None,
        launcher='none')
    runner = SimpleNamespace(rank=0, test=lambda: {})

    monkeypatch.setattr(TEST_MODULE, 'parse_args', lambda: args)
    monkeypatch.setattr(TEST_MODULE.Config, 'fromfile', lambda _path: cfg)
    monkeypatch.setattr(
        TEST_MODULE.ExperimentRunner, 'from_cfg', lambda _cfg: runner)
    monkeypatch.setattr(
        TEST_MODULE, 'configure_test_experiment',
        lambda _cfg, _checkpoint, output_dir: output_dir)
    monkeypatch.setattr(
        TEST_MODULE, 'remove_runtime_config_artifacts',
        lambda _experiment_dir, _config_path: None)

    TEST_MODULE.main()

    assert capsys.readouterr().out == ''
