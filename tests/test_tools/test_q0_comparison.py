"""The comparison must finish five epochs, not mistake a staged cutoff for completion."""
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from LaneLoom.tools import run_q0_comparison as comparison


def test_staged_runs_continue_only_to_five_complete_epochs(tmp_path, monkeypatch):
    for directory in ('fidelity/baseline', 'fidelity/spatial', 'continuous'):
        run = tmp_path / directory
        run.mkdir(parents=True)
        (run / 'train.pid').write_text('1234')
        torch.save(dict(step=20000, epoch=1, batch=733), run / 'last.pth')
    commands = []

    def launch(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(pid=10000 + len(commands))

    def end_first_cycle(_):
        raise StopIteration

    monkeypatch.setattr(sys, 'argv', ['comparison', '--root', str(tmp_path)])
    monkeypatch.setattr(comparison, 'training_alive', lambda *_: False)
    monkeypatch.setattr(comparison.subprocess, 'Popen', launch)
    monkeypatch.setattr(comparison.time, 'sleep', end_first_cycle)
    with pytest.raises(StopIteration):
        comparison.main()
    assert len(commands) == 3
    for command in commands:
        assert command[command.index('--epochs') + 1] == '10'
        assert command[command.index('--stop-after-epoch') + 1] == '5'
        assert '--resume' in command
        assert '--steps' not in command  # zero default completes the epoch schedule
    plan = json.loads((tmp_path / 'validation/plan.json').read_text())
    assert plan['epochs'] == 5
    assert plan['milestone_epochs'] == [1, 2, 5]


def test_partial_epoch_checkpoint_cannot_be_used_as_epoch_boundary(tmp_path):
    path = tmp_path / 'epoch_1.pth'
    torch.save(dict(step=20000, epoch=1, batch=733), path)
    with pytest.raises(ValueError, match='does not end epoch'):
        comparison.epoch_ready(path, 1)
    torch.save(dict(step=19267, epoch=1, batch=0), path)
    assert comparison.epoch_ready(path, 1)


def test_completed_five_epoch_runs_never_resume_to_ten(tmp_path, monkeypatch):
    for directory in ('fidelity/baseline', 'fidelity/spatial', 'continuous'):
        run = tmp_path / directory
        run.mkdir(parents=True)
        (run / 'train.pid').write_text('1234')
        torch.save(dict(step=5 * 19267, epoch=5, batch=0), run / 'last.pth')
    for epoch in (1, 2, 5):
        torch.save(dict(epoch=epoch, batch=0),
                   tmp_path / 'fidelity/baseline' / f'epoch_{epoch}.pth')
        for variant in ('spatial', 'continuous'):
            output = tmp_path / 'validation' / f'epoch_{epoch}' / variant
            output.mkdir(parents=True)
            (output / f'comparison_step_{epoch * 19267}.json').write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['comparison', '--root', str(tmp_path)])
    monkeypatch.setattr(comparison, 'training_alive', lambda *_: False)

    def unexpected_resume(*args, **kwargs):
        pytest.fail('Training must end after epoch 5')

    monkeypatch.setattr(comparison.subprocess, 'Popen', unexpected_resume)
    comparison.main()
    status = json.loads((tmp_path / 'validation/status.json').read_text())
    assert len(status['completed_validations']) == 6
    assert all(run['state'] == 'training_complete' for run in status['runs'].values())
