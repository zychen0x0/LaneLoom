"""Measure extractor and point-cloud-to-Q0 cost with isolated GPU execution."""
import argparse
import gc
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmdet3d.registry import DATASETS

from LaneLoom.datasets.openlane.dataset import openlane_collate
from LaneLoom.engine import register_all_modules
from LaneLoom.tools.train_q0_prior import Q0Pretrainer
from LaneLoom.vis.output import infer_openlane_sample as infer


def gpu_batch(batch, device):
    return {key: ([p.to(device) for p in value] if key == 'lidar' else
                  value.to(device) if torch.is_tensor(value) else value)
            for key, value in batch.items()}


@torch.inference_mode()
def measure(call, warmup, iterations):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    wall, cuda = [], []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        began = time.perf_counter()
        start.record()
        output = call()
        end.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - began) * 1000)
        cuda.append(start.elapsed_time(end))
        del output
    peak = torch.cuda.max_memory_allocated()
    return dict(wall_mean_ms=float(np.mean(wall)), wall_median_ms=float(np.median(wall)),
                wall_p90_ms=float(np.quantile(wall, .9)),
                cuda_median_ms=float(np.median(cuda)),
                fps_batch1=1000 / float(np.mean(wall)),
                peak_allocated_mib=peak / 2**20,
                temporary_peak_mib=(peak - baseline) / 2**20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--pause-processes', nargs='*', type=int, default=[])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=50)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    register_all_modules()
    configs = [Config.fromfile(p) for p in args.configs]
    # Load/augment data before suspending training. All backward measurements
    # use ordinary training frames from the same excluded-segment split.
    dataset = DATASETS.build(configs[-1].train_dataloader.dataset)
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(7))[:8]
    train_cpu = openlane_collate([dataset[int(index)] for index in order])
    image = Path('data/openlane/images/validation/segment-6161542573106757148_585_030_605_030_with_camera_labels/150810876026325200.jpg')
    inference_cpu = infer._build_sample(configs[0], image)
    inference_cpu = {key: inference_cpu[key] for key in ('img', 'lidar', 'img_metas')}
    rows, paused = [], []
    try:
        for pid in args.pause_processes:
            command = Path(f'/proc/{pid}/cmdline').read_bytes()
            if b'train_q0_prior.py' not in command or os.getpgid(pid) != pid:
                raise ValueError(f'PID {pid} is not an independent Q0 training process group')
            os.killpg(pid, signal.SIGSTOP)
            paused.append(pid)
        # Allow already submitted GPU work to finish; the other processes
        # retain allocations but submit no kernels during the benchmark.
        time.sleep(1)
        for config in configs:
            torch.manual_seed(7)
            model = Q0Pretrainer(config).to(args.device).eval()
            head = model.anchor_prior
            name = type(head).__name__
            inputs = dict(features=torch.randn(1, 128, 250, 250, device=args.device),
                          occupancy=torch.ones(1, 1, 250, 250, device=args.device, dtype=torch.bool))
            if model.pts_bev_encoder.return_road_evidence:
                inputs['road_evidence'] = torch.randn(1, 16, 250, 250, device=args.device) * .1
                inputs['road_evidence'][:, 0] = 1
            row = dict(model=name, head_parameters=sum(p.numel() for p in head.parameters()))
            row['head_inference'] = measure(lambda: head(inputs), args.warmup, args.iterations)
            del inputs
            gc.collect()
            torch.cuda.empty_cache()
            batch = gpu_batch(inference_cpu, args.device)

            def q0():
                features = model.pts_bev_encoder(batch['lidar'], images=batch['img'], img_metas=batch['img_metas'])
                prior = head(features)
                prior.update(model.proposal_expander(prior))
                return prior

            row['points_to_q0_inference'] = measure(q0, args.warmup, args.iterations)
            del batch
            gc.collect()
            torch.cuda.empty_cache()
            batch = gpu_batch(train_cpu, args.device)
            model.train()
            settings = dict(config.optim_wrapper.optimizer)
            settings.pop('type')
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), **settings)

            def training_step():
                optimizer.zero_grad(set_to_none=True)
                losses = model(batch, args.device)
                sum(value for key, value in losses.items() if key.startswith('loss_')).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
                optimizer.step()

            training_step()  # warm backward kernels and initialize optimizer state
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            training_step()
            torch.cuda.synchronize()
            row['q0_training_batch8'] = dict(
                step_ms=(time.perf_counter() - began) * 1000,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
            rows.append(row)
            print(json.dumps(row), flush=True)
            del optimizer, model, head, batch, q0, training_step
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        for pid in paused:
            try:
                os.killpg(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
    result = dict(gpu=torch.cuda.get_device_name(), precision='FP32, TF32 disabled',
                  inference_batch=1, training_batch=8, warmup=args.warmup, iterations=args.iterations,
                  head_input='synthetic 128x250x250, with observed 16-channel evidence for continuous head',
                  inference_frame=image.stem, inference_lidar_points=len(inference_cpu['lidar'][0]),
                  inference_scope='painted temporal points -> BEV -> templates -> 505 Q0; excludes image backbone, decoder, disk IO',
                  training_scope='ordinary held-out-split training batch; encoder, head, losses, backward and AdamW',
                  training_frames=[dataset.label_files[int(i)].stem for i in order],
                  memory_scope='PyTorch peak allocated in benchmark process; excludes other suspended processes and driver/context allocations',
                  configs=[str(p) for p in args.configs], measurements=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
