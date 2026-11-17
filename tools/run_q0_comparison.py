"""Complete the matched five-epoch Q0 runs and evaluate fixed milestones."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


MILESTONES = (1, 2, 5)
TRAINING_EPOCHS = 5
LR_SCHEDULE_EPOCHS = 10  # Preserve the LR trajectory of the checkpoints already trained.
VARIANTS = ('spatial', 'continuous')
PLY = 'vis/input/openlane/segment-6161542573106757148_585_030_605_030_with_camera_labels/150810876026/04_points.ply'


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def training_alive(pid, directory):
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\x00', ' ')
        return ('train_q0_prior.py' in command
                and (str(directory) in command or str(directory.resolve()) in command))
    except FileNotFoundError:
        return False


def epoch_ready(path, epoch):
    if not path.exists():
        return False
    try:
        saved = torch.load(path, map_location='cpu', weights_only=False)
    except (RuntimeError, EOFError, OSError):
        # The initial processes predate atomic epoch-checkpoint writes.
        return False
    if saved['epoch'] != epoch or saved['batch'] != 0:
        raise ValueError(f'Checkpoint {path} does not end epoch {epoch}')
    return True


def plot_curves(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'svg.fonttype': 'none', 'font.size': 10})
    curves = {name: [] for name in ('baseline', *VARIANTS)}
    for epoch in MILESTONES:
        baseline_added = False
        for name in VARIANTS:
            directory = root / f'epoch_{epoch}' / name
            files = list(directory.glob('comparison_step_*.json'))
            if not files:
                continue
            data = json.loads(files[0].read_text())
            for key, model_name in (('baseline', 'baseline'), ('spatial', name)):
                if model_name == 'baseline' and baseline_added:
                    continue
                counts = []
                for group in ('all', 'curve', 'slopes'):
                    samples = [s for s in data['samples'] if group == 'all'
                               or (s['group'] == 'curve' if group == 'curve' else s['group'] != 'curve')]
                    total = sum(s['raw_gt'] for s in samples)
                    covered = sum(s[key]['paired_xz_p80']['covered'] for s in samples)
                    counts.append(100 * covered / total)
                curves[model_name].append((epoch, counts))
                if model_name == 'baseline':
                    baseline_added = True
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    fig.subplots_adjust(left=.07, right=.98, bottom=.18, top=.78, wspace=.3)
    colors = dict(baseline='.35', spatial='#0072B2', continuous='#D55E00')
    labels = dict(baseline='Original head', spatial='Spatial attention', continuous='Continuous paths')
    for index, title in enumerate(('All 74 raw GT lanes', 'Curves: 40 raw GT lanes', 'Slopes: 34 raw GT lanes')):
        for name, points in curves.items():
            if points:
                axes[index].plot([p[0] for p in points], [p[1][index] for p in points],
                                 'o-', color=colors[name], label=labels[name])
        axes[index].set(title=title, xlabel='Completed epochs', ylabel='Q0 coverage (%)',
                        xlim=(.5, 5.5), ylim=(0, 100), xticks=MILESTONES)
        axes[index].grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
    fig.suptitle('Target-segment holdout | 0.3 m / 80% raw-point coverage | distinct queries', y=.89, fontsize=11)
    from LaneLoom.vis.output import visualize_query_stages as visual
    sys.path.insert(0, str(visual.AUDIT_SCRIPTS))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig, strict=True)
    temporary = root / 'coverage_curve.tmp.svg'
    fig.savefig(temporary, format='svg')
    temporary.replace(root / 'coverage_curve.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('work_dirs/q0_bev_holdout'))
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / 'validation'
    output.mkdir(parents=True, exist_ok=True)
    runs = dict(baseline=root / 'fidelity/baseline', spatial=root / 'fidelity/spatial',
                continuous=root / 'continuous')
    devices = dict(baseline='cuda:0', spatial='cuda:1', continuous='cuda:0')
    pids = {name: int((run / 'train.pid').read_text()) for name, run in runs.items()}
    children, complete = {}, set()
    plan = dict(epochs=TRAINING_EPOCHS, milestone_epochs=list(MILESTONES),
                lr_schedule_epochs=LR_SCHEDULE_EPOCHS,
                schedule_note='Keep the existing LR trajectory; stop all three runs after epoch 5',
                datasets='154143 ordinary training frames; all 19 target segments excluded',
                objectives='Same full-template and Q0 geometry losses; auxiliary direction/curvature weights zero',
                runs={name: str(run) for name, run in runs.items()},
                validation='21 target frames / 74 raw GT lanes, plus supplied validation PLY; no GT in prediction')
    atomic_json(output / 'plan.json', plan)
    while True:
        status = {}
        for name, run in runs.items():
            if not training_alive(pids[name], run):
                saved = torch.load(run / 'last.pth', map_location='cpu', weights_only=False)
                if saved['epoch'] >= TRAINING_EPOCHS and saved['batch'] == 0:
                    status[name] = dict(state='training_complete', step=saved['step'])
                elif saved['step'] == 20000 and name not in children:
                    command = [sys.executable, 'tools/train_q0_prior.py', str(run / 'config.py'),
                               '--work-dir', str(run), '--device', devices[name], '--batch-size', '8',
                               '--num-workers', '4', '--epochs', str(LR_SCHEDULE_EPOCHS),
                               '--stop-after-epoch', str(TRAINING_EPOCHS),
                               '--resume', str(run / 'last.pth')]
                    with (run / 'train.log').open('ab', buffering=0) as log:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                 stdin=subprocess.DEVNULL, start_new_session=True,
                                                 env={**os.environ, 'PYTHONPATH': '/workspace'})
                    children[name] = child
                    pids[name] = child.pid
                    (run / 'train.pid').write_text(str(child.pid) + '\n')
                    status[name] = dict(state='continuing_to_epoch_5', pid=child.pid)
                    print(name, 'resumed at 20000 steps; will stop after epoch 5', flush=True)
                else:
                    raise RuntimeError(f'{name} stopped unexpectedly at step {saved["step"]}; inspect train.log')
            else:
                status[name] = dict(state='training', pid=pids[name])
        for epoch in MILESTONES:
            if all((epoch, name) in complete for name in VARIANTS):
                continue
            baseline = runs['baseline'] / f'epoch_{epoch}.pth'
            if not epoch_ready(baseline, epoch):
                continue
            for name in VARIANTS:
                destination = output / f'epoch_{epoch}' / name
                key = (epoch, name)
                if list(destination.glob('comparison_step_*.json')):
                    complete.add(key)
                    continue
                candidate = runs[name] / f'epoch_{epoch}.pth'
                if not epoch_ready(candidate, epoch):
                    continue
                command = [sys.executable, 'tools/evaluate_q0_prior.py', '--baseline', str(baseline),
                           '--spatial', str(candidate), '--work-dir', str(destination), '--device', 'cuda:1',
                           '--render', '--ply', PLY]
                print('VALIDATING', epoch, name, flush=True)
                subprocess.run(command, check=True, env={**os.environ, 'PYTHONPATH': '/workspace'})
                complete.add(key)
                plot_curves(output)
        atomic_json(output / 'status.json', dict(runs=status, completed_validations=sorted(complete)))
        if all(s['state'] == 'training_complete' for s in status.values()) and len(complete) == 6:
            print('Completed epoch 1/2/5 validations for both architecture variants', flush=True)
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
