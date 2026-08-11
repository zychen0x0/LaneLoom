from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path


TEST_PATH = Path(__file__).resolve().parents[2] / 'tools/test.py'
SPEC = spec_from_file_location('laneloom_test_cli', TEST_PATH)
TEST_MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(TEST_MODULE)


def test_default_evaluation_dir_is_next_to_checkpoint_directories(tmp_path):
    checkpoint = (
        tmp_path / 'exp_date_20260803_081030' / 'checkpoints'
        / 'best_OpenLaneTemporal_F1_epoch_44.pth')

    output_dir = TEST_MODULE.resolve_evaluation_dir(checkpoint, None)

    assert output_dir == (
        tmp_path / 'exp_date_20260803_081030' / 'evaluation')


def test_explicit_work_dir_overrides_checkpoint_default(tmp_path):
    checkpoint = tmp_path / 'experiment' / 'checkpoints' / 'best.pth'
    requested = tmp_path / 'custom_evaluation'

    output_dir = TEST_MODULE.resolve_evaluation_dir(checkpoint, requested)

    assert output_dir == requested.resolve()


def test_evaluation_summary_writes_and_merges_table3_f1_results(tmp_path):
    checkpoint = tmp_path / 'experiment' / 'checkpoints' / 'best.pth'
    output_dir = tmp_path / 'experiment' / 'evaluation'

    TEST_MODULE.write_evaluation_summary(
        output_dir,
        checkpoint,
        {'up_down_case': 0.565331995, 'curve_case': 0.704237342})
    TEST_MODULE.write_evaluation_summary(
        output_dir,
        checkpoint,
        {'night_case': 0.616515815})

    payload = json.loads(
        (output_dir / 'evaluation_results.json').read_text())
    assert payload == {
        'checkpoint': str(checkpoint.resolve()),
        'F1_percent': {
            'Up&Down': 56.5332,
            'Curve': 70.4237,
            'Night': 61.6516,
        },
    }
    assert (output_dir / 'evaluation_results.txt').read_text() == (
        f'Checkpoint: {checkpoint.resolve()}\n'
        '\n'
        'Up&Down          56.5332\n'
        'Curve            70.4237\n'
        'Night            61.6516\n')
