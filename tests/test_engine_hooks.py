from types import SimpleNamespace
from unittest.mock import Mock, patch

from mmengine.fileio import LocalBackend

from LaneLoom.engine.hooks.laneloom_checkpoint_hook import (
    ExperimentCheckpointHook,
)


def test_checkpoint_hook_uses_current_file_backend_api(tmp_path):
    runner = SimpleNamespace(
        work_dir=str(tmp_path / 'logs'),
        logger=Mock(),
        message_hub=SimpleNamespace(runtime_info={}),
    )
    hook = ExperimentCheckpointHook(out_dir=str(tmp_path / 'checkpoints'))

    with patch(
            'mmengine.hooks.checkpoint_hook.FileClient.infer_client',
            side_effect=AssertionError('legacy FileClient must not be used')):
        hook.before_train(runner)

    assert isinstance(hook.file_backend, LocalBackend)
    assert hook.out_dir == str(tmp_path / 'checkpoints')
