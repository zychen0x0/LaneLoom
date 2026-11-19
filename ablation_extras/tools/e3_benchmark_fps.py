"""E3: measure LaneLoom inference speed on one GPU with batch size 1.

Two latencies are reported per frame:
  * model latency: ``test_step`` on a loaded batch (host-to-device copy,
    RGB painting, both encoders, decoding and curve NMS);
  * end-to-end latency: model latency plus the time spent waiting for the
    dataloader (file reading and sweep alignment).

A stage breakdown is measured in a separate call on the same frames:
preprocessing, image encoder (all T queued frames; the per-frame cost is also
given), LiDAR encoder (accumulated sweeps, including RGB painting) and lane
head (CQG, curve-aligned sampling, temporal convolution, decoding, NMS).
Comparing the multi-frame and single-frame breakdowns separates the cost of
re-encoding historical images from the cost of curve-aligned fusion.

Usage (from the repository root, PYTHONPATH set to its parent):
    python ablation_extras/tools/e3_benchmark_fps.py CONFIG CHECKPOINT --tag mf

Outputs: ablation_extras/return_back/e3_fps_<tag>.json and .md
"""

import argparse
import json
import statistics
import time

import torch

from LaneLoom.ablation_extras.tools._runtime import (
    RETURN_ROOT, Progress, build_model_and_loader, gpu_name, now)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--tag', required=True,
                        help='Short name used in the output file names.')
    parser.add_argument('--num-warmup', type=int, default=30)
    parser.add_argument('--num-frames', type=int, default=300)
    return parser.parse_args()


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def stage_times(model, data):
    """Milliseconds per stage for one batch, plus the number of frames."""
    _sync()
    start = time.perf_counter()
    batch = model.data_preprocessor(data, False)
    _sync()
    t_pre = time.perf_counter()
    img = batch['img']
    img_metas = batch['img_metas']
    points = batch.get('lidar', batch.get('points'))
    if torch.is_tensor(points):
        points = [points]
    img_feats = model.extract_temporal_img_feat(img, img_metas)
    _sync()
    t_img = time.perf_counter()
    pts_feats = model.extract_pts_feat(
        points, img_feats=img_feats, img_metas=img_metas, images=img)
    _sync()
    t_pts = time.perf_counter()
    model.lane_head.predict(
        img_feats=img_feats, pts_feats=pts_feats, img_metas=img_metas,
        points=points, prev_bev=None)
    _sync()
    t_head = time.perf_counter()
    stages = dict(
        preprocess=(t_pre - start) * 1000.0,
        image_encoder=(t_img - t_pre) * 1000.0,
        lidar_encoder=(t_pts - t_img) * 1000.0,
        lane_head=(t_head - t_pts) * 1000.0)
    return stages, int(img.shape[1])


def summarize(values):
    ordered = sorted(values)
    return dict(
        mean_ms=round(statistics.fmean(values), 3),
        median_ms=round(statistics.median(values), 3),
        p90_ms=round(ordered[int(0.9 * (len(ordered) - 1))], 3),
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('E3 needs a CUDA GPU.')
    torch.backends.cudnn.benchmark = False
    print(f'[{now()}] E3 start: {args.tag} on {gpu_name()}', flush=True)
    model, loader = build_model_and_loader(
        args.config, args.checkpoint, f'e3_fps_{args.tag}', batch_size=1)

    total = args.num_warmup + args.num_frames
    progress = Progress(total, 'E3 frames', every=50)
    model_ms, data_ms = [], []
    stage_ms, frames, breakdown_error = {}, None, None
    iterator = iter(loader)
    with torch.no_grad():
        for index in range(total):
            wait_start = time.perf_counter()
            data = next(iterator)
            data_time = time.perf_counter() - wait_start
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.test_step(data)
            torch.cuda.synchronize()
            step_time = time.perf_counter() - start
            if breakdown_error is None:
                try:
                    stages, frames = stage_times(model, data)
                except Exception as error:  # keep the main timing usable
                    breakdown_error = f'{type(error).__name__}: {error}'
                    print(f'[{now()}] stage breakdown skipped: '
                          f'{breakdown_error}', flush=True)
                else:
                    if index >= args.num_warmup:
                        for name, value in stages.items():
                            stage_ms.setdefault(name, []).append(value)
            if index >= args.num_warmup:
                model_ms.append(step_time * 1000.0)
                data_ms.append(data_time * 1000.0)
            progress.update(index + 1)

    model_stats = summarize(model_ms)
    e2e = [m + d for m, d in zip(model_ms, data_ms)]
    e2e_stats = summarize(e2e)
    result = dict(
        tag=args.tag,
        gpu=gpu_name(),
        torch=torch.__version__,
        config=args.config,
        checkpoint=args.checkpoint,
        batch_size=1,
        warmup_frames=args.num_warmup,
        measured_frames=len(model_ms),
        model_latency=model_stats,
        model_fps=round(1000.0 / model_stats['mean_ms'], 2),
        end_to_end_latency=e2e_stats,
        end_to_end_fps=round(1000.0 / e2e_stats['mean_ms'], 2),
        peak_memory_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1),
        parameters_m=round(
            sum(param.numel() for param in model.parameters()) / 1e6, 2),
    )
    if stage_ms:
        stage_mean = {name: round(statistics.fmean(values), 3)
                      for name, values in stage_ms.items()}
        result['queued_frames'] = frames
        result['stage_latency_mean_ms'] = stage_mean
        result['image_encoder_per_frame_ms'] = round(
            stage_mean['image_encoder'] / frames, 3)
    else:
        result['stage_breakdown_error'] = breakdown_error
    RETURN_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = RETURN_ROOT / f'e3_fps_{args.tag}.json'
    json_path.write_text(json.dumps(result, indent=2) + '\n')
    table = (
        '| Model | GPU | Params (M) | Model latency (ms) | Model FPS | '
        'End-to-end latency (ms) | End-to-end FPS |\n'
        '|---|---|---|---|---|---|---|\n'
        f'| {args.tag} | {result["gpu"]} | {result["parameters_m"]} | '
        f'{model_stats["mean_ms"]} | '
        f'{result["model_fps"]} | {e2e_stats["mean_ms"]} | '
        f'{result["end_to_end_fps"]} |\n')
    if stage_ms:
        stage_mean = result['stage_latency_mean_ms']
        table += (
            '\n| Model | Frames | Preprocess (ms) | Image encoder (ms) '
            '| Image encoder per frame (ms) | LiDAR encoder (ms) '
            '| Lane head (ms) |\n|---|---|---|---|---|---|---|\n'
            f'| {args.tag} | {frames} | {stage_mean["preprocess"]} | '
            f'{stage_mean["image_encoder"]} | '
            f'{result["image_encoder_per_frame_ms"]} | '
            f'{stage_mean["lidar_encoder"]} | {stage_mean["lane_head"]} |\n')
    (RETURN_ROOT / f'e3_fps_{args.tag}.md').write_text(table)
    print(table)
    print(f'[{now()}] E3 SUCCESS. Send this file to Claude: {json_path}',
          flush=True)


if __name__ == '__main__':
    main()
