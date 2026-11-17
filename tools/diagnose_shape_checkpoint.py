"""Audit LiDAR templates, Q0 proposals, and decoder recall on three frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from LaneLoom.datasets.openlane.dataset import OpenLaneDataset
from LaneLoom.datasets.openlane import full_shape_targets as shape_target_builder
from LaneLoom.datasets.openlane.full_shape_targets import build_full_shape_targets
from LaneLoom.models.head.curve_nms import mean_distance_curve_nms
from LaneLoom.vis.output.infer_openlane_sample import _build_model, _build_sample


DEFAULT_CHECKPOINT = (
    ROOT / "work_dirs/laneloom_openlane_image_lidar_temporal_0923"
    / "exp_date_20260923_230337/checkpoints"
    / "epoch_18.pth"
)
SCENES = (
    "segment-14811410906788672189_373_113_393_113_with_camera_labels/150724276945019300",
    "segment-17612470202990834368_2800_000_2820_000_with_camera_labels/155364079579344600",
    "segment-5373876050695013404_3817_170_3837_170_with_camera_labels/151026502726599100",
)
FIGURE_SCRIPTS = Path("/root/.codex/skills/nature-figure/scripts")


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _save_pair(path, title, draw):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
    })
    fig, (bev, yz) = plt.subplots(
        1, 2, figsize=(11, 5),
        gridspec_kw={"width_ratios": [1, 2]},
        constrained_layout=True)
    draw(bev, yz)
    bev.set(xlabel="Lateral x (m)", ylabel="Forward y (m)", ylim=(0, 100))
    yz.set(xlabel="Forward y (m)", ylabel="Height z (m)", xlim=(0, 100))
    bev.set_aspect("equal", adjustable="box")
    for ax in (bev, yz):
        ax.grid(alpha=0.18)
    fig.suptitle(title, fontsize=10)
    fig.canvas.draw()
    # These are diagnostic figures, but keep the rendered two-panel layout
    # auditable using the shared plotting contract.
    sys.path.insert(0, str(FIGURE_SCRIPTS))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(
        fig, json_out=str(path.with_suffix(".alignment.json")),
        overlay_svg=str(path.with_suffix(".alignment.svg")),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def _draw_xyz(bev, yz, xyz, *, color, linestyle="-", alpha=1.0,
              linewidth=1.5, label=None, marker=None):
    bev.plot(xyz[:, 0], xyz[:, 1], color=color, linestyle=linestyle,
             alpha=alpha, linewidth=linewidth, marker=marker,
             markersize=2.2, label=label)
    yz.plot(xyz[:, 1], xyz[:, 2], color=color, linestyle=linestyle,
            alpha=alpha, linewidth=linewidth, marker=marker,
            markersize=2.2, label=label)


def _teacher_data(cfg, image):
    dc = cfg.val_dataloader.dataset
    dataset = OpenLaneDataset(
        dataset_cfg=dc.dataset_cfg,
        temporal_sampling_distance=dc.temporal_sampling_distance,
        smooth_3d_gt=dc.smooth_3d_gt,
        gt_outlier_residual=dc.gt_outlier_residual,
        modality=dc.modality, split=dc.split, max_samples=1)
    label = (Path(dc.dataset_cfg.data_label_dir) / "validation"
             / image.parent.name / f"{image.stem}.json")
    gt = dataset._parse_openlane_gt(dataset._read_json(label))
    lanes = _numpy(gt["gt_lanes_3d"])
    visibility = _numpy(gt["gt_lanes_visibility"]) > 0.5
    targets = build_full_shape_targets(
        lanes, visibility, dataset.anchor_y_steps,
        max_templates=cfg.train_dataloader.dataset.dataset_cfg.max_shape_templates,
        min_lane_points=cfg.train_dataloader.dataset.dataset_cfg.full_shape_min_points,
        min_lane_span=cfg.train_dataloader.dataset.dataset_cfg.full_shape_min_span)
    return lanes, visibility, targets


def _aggregation_audit(lanes_3d, visibility, anchor_y_steps,
                       min_lane_points, min_lane_span):
    """Expose the support behind each median target; no target is changed."""
    builder = shape_target_builder
    y_steps = np.asarray(anchor_y_steps, dtype=np.float64)
    dense_step = float(np.min(np.diff(y_steps))) / 5.0
    lanes = []
    for packed_index in range(len(lanes_3d)):
        valid = visibility[packed_index] & np.isfinite(
            lanes_3d[packed_index]).all(axis=-1)
        anchors = np.asarray(lanes_3d[packed_index, valid], dtype=np.float64)
        if (len(anchors) < min_lane_points or
                float(np.ptp(anchors[:, 1])) < min_lane_span):
            continue
        lanes.append(builder._LaneFit(
            packed_index=packed_index, anchors=anchors,
            filtered=anchors.copy(), dense=anchors.copy()))
    while lanes:
        groups = builder._cluster_lanes(lanes)
        old_count = len(lanes)
        old_points = sum(len(lane.anchors) for lane in lanes)
        builder._filter_visible_anchors(lanes, groups, dense_step)
        lanes = [lane for lane in lanes if
                 len(lane.filtered) >= min_lane_points and
                 float(np.ptp(lane.filtered[:, 1])) >= min_lane_span]
        for lane in lanes:
            lane.anchors = lane.filtered.copy()
        if (len(lanes) == old_count and
                sum(len(lane.anchors) for lane in lanes) == old_points):
            break
    if not lanes:
        return []
    groups = builder._cluster_lanes(lanes)
    _, relative = builder._fit_complete_lanes(lanes, groups, y_steps,
                                               float(y_steps[-1]))
    ranked = sorted(groups, key=lambda group: -sum(
        float(np.ptp(lanes[index].filtered[:, 1])) for index in group))
    result = []
    for group in ranked:
        longest = max(group, key=lambda index: np.ptp(
            lanes[index].filtered[:, 1]))
        median = np.median(np.stack([relative[index] for index in group]),
                           axis=0)
        longest_curve = relative[longest]
        result.append(dict(
            member_gt_ids=[int(lanes[index].packed_index) for index in group],
            support_y_m={str(lanes[index].packed_index): [
                round(float(lanes[index].support[0]), 2),
                round(float(lanes[index].support[1]), 2)] for index in group},
            longest_gt_id=int(lanes[longest].packed_index),
            median_relative_z_at_100_m=round(float(median[-1, 2]), 4),
            longest_relative_z_at_100_m=round(float(longest_curve[-1, 2]), 4),
            delta_longest_minus_median_z_at_100_m=round(float(
                longest_curve[-1, 2] - median[-1, 2]), 4)))
    return result


def _plot_teacher(path, frame, lanes, visible, targets):
    valid = targets["gt_lanes_full_valid"]

    def draw(bev, yz):
        for lane_id in np.flatnonzero(visible.sum(axis=1) >= 2):
            color = plt.get_cmap("tab10")(int(lane_id) % 10)
            if valid[lane_id]:
                fit = targets["gt_lanes_3d_full"][lane_id]
                _draw_xyz(bev, yz, fit, color=color, linestyle="--",
                          linewidth=1.1, alpha=0.75)
            _draw_xyz(bev, yz, lanes[lane_id, visible[lane_id]],
                      color=color, linewidth=2.0, marker="o",
                      label=f"GT {lane_id}")
        bev.set_xlim(-25, 25)
    count = int((visible.sum(axis=1) >= 2).sum())
    _save_pair(path, f"{frame} | fitted GT {int(valid.sum())}/{count} lanes"
               " | solid: visible GT; dashed: full-range target", draw)


def _geometry_stage(x, z, lanes, visible, indices=None):
    x = _numpy(x)
    z = _numpy(z)
    if indices is None:
        indices = np.arange(len(x))
    indices = np.asarray(indices, dtype=np.int64)
    result = {}
    for lane_id in np.flatnonzero(visible.sum(axis=1) >= 2):
        mask = visible[lane_id]
        if len(indices) == 0:
            result[str(lane_id)] = dict(covered=False, best_query=None,
                                        match_ratio=0.0, mean_xz_m=None)
            continue
        delta_x = x[indices][:, mask] - lanes[lane_id, mask, 0]
        delta_z = z[indices][:, mask] - lanes[lane_id, mask, 2]
        dist = np.hypot(delta_x, delta_z)
        ratios = (dist < 0.3).mean(axis=1)
        means = dist.mean(axis=1)
        best = int(np.lexsort((means, -ratios))[0])
        result[str(lane_id)] = dict(
            covered=bool(ratios[best] >= 0.8),
            best_query=int(indices[best]),
            match_ratio=round(float(ratios[best]), 4),
            mean_xz_m=round(float(means[best]), 4))
    return result


def _stage_summary(stage):
    return dict(covered=sum(row["covered"] for row in stage.values()),
                eligible=len(stage))


def _teacher_template_replay(lanes, visible, targets):
    """Replay only supervised GT modes through the Q0 x-only seed grid."""
    modes = targets["gt_shape_templates"][targets["gt_shape_template_valid"]]
    seeds = np.linspace(-25.0, 25.0, 101)
    if len(modes):
        x = (modes[:, None, :, 0] + seeds[None, :, None]).reshape(
            -1, modes.shape[1])
        z = np.broadcast_to(modes[:, None, :, 2],
                            (len(modes), len(seeds), modes.shape[1])).reshape(
                                -1, modes.shape[1])
    else:
        x = z = np.empty((0, lanes.shape[1]), dtype=np.float32)
    stage = _geometry_stage(x, z, lanes, visible)
    with_z_translation = {}
    for lane_id in np.flatnonzero(visible.sum(axis=1) >= 2):
        mask = visible[lane_id]
        target_x = lanes[lane_id, mask, 0]
        target_z = lanes[lane_id, mask, 2]
        best = 0.0
        for template in modes:
            shape_x = template[mask, 0]
            shape_z = template[mask, 2]
            z_offset = np.median(target_z - shape_z)
            distance = np.hypot(
                shape_x[None] + seeds[:, None] - target_x[None],
                shape_z[None] + z_offset - target_z[None])
            best = max(best, float((distance < 0.3).mean(axis=1).max()))
        with_z_translation[str(lane_id)] = round(best, 4)
    return x, z, stage, with_z_translation


def _plot_templates(path, frame, predicted, targets):
    truth = targets["gt_shape_templates"]
    truth_ids = np.flatnonzero(targets["gt_shape_template_valid"])
    costs = np.zeros((len(predicted), len(truth_ids)))
    for index in range(len(predicted)):
        for col, target_id in enumerate(truth_ids):
            costs[index, col] = np.mean(np.linalg.norm(
                predicted[index][:, (0, 2)] - truth[target_id][:, (0, 2)],
                axis=-1))
    rows, cols = linear_sum_assignment(costs) if costs.size else ([], [])
    matches = {int(row): int(truth_ids[col]) for row, col in zip(rows, cols)}

    def draw(bev, yz):
        for index, curve in enumerate(predicted):
            color = plt.get_cmap("tab10")(index)
            _draw_xyz(bev, yz, curve, color=color, linewidth=1.9,
                      label=f"Pred T{index}")
            if index in matches:
                _draw_xyz(bev, yz, truth[matches[index]], color=color,
                          linestyle="--", linewidth=1.1)
    _save_pair(path, f"{frame} | relative 0–100 m templates"
               " | solid: model; dashed: fitted GT", draw)
    result = {}
    for row, target_id in matches.items():
        delta = predicted[row] - truth[target_id]
        result[f"T{row}"] = dict(
            gt_template=target_id,
            mean_xz_m=round(float(np.linalg.norm(delta[:, (0, 2)], axis=-1).mean()), 4),
            mean_abs_x_m=round(float(np.abs(delta[:, 0]).mean()), 4),
            mean_abs_z_m=round(float(np.abs(delta[:, 2]).mean()), 4),
            far_mean_abs_z_m=round(float(np.abs(delta[11:, 2]).mean()), 4),
            z_at_100_error_m=round(float(delta[-1, 2]), 4))
    return result


def _plot_stage(path, frame, name, x, z, lanes, visible, stage, y_steps,
                predicted_visibility=None):
    x, z = _numpy(x), _numpy(z)
    pv = _numpy(predicted_visibility) if predicted_visibility is not None else None

    def draw(bev, yz):
        for lane_id in np.flatnonzero(visible.sum(axis=1) >= 2):
            color = plt.get_cmap("tab10")(int(lane_id) % 10)
            _draw_xyz(bev, yz, lanes[lane_id, visible[lane_id]],
                      color=color, linewidth=2.2, marker="o")
            row = stage[str(lane_id)]
            if row["best_query"] is None:
                continue
            query = row["best_query"]
            xyz = np.column_stack((x[query], y_steps, z[query]))
            _draw_xyz(bev, yz, xyz, color=color, linestyle="--",
                      linewidth=1.1, alpha=0.8)
            if pv is not None:
                vm = pv[query] >= 0.5
                if vm.sum() >= 2:
                    _draw_xyz(bev, yz, xyz[vm], color=color,
                              linewidth=1.2, alpha=0.85)
        bev.set_xlim(-25, 25)
    summary = _stage_summary(stage)
    _save_pair(path, f"{frame} | {name} | Q0-style coverage "
               f"{summary['covered']}/{summary['eligible']}"
               " | thick: GT; dashed: query", draw)


def _plot_final_predictions(path, frame, x, z, lanes, visible, y_steps,
                            visibility, keep_indices):
    def draw(bev, yz):
        for lane_id in np.flatnonzero(visible.sum(axis=1) >= 2):
            color = plt.get_cmap("tab10")(int(lane_id) % 10)
            _draw_xyz(bev, yz, lanes[lane_id, visible[lane_id]],
                      color=color, linewidth=2.2, marker="o")
        for query_id in keep_indices:
            mask = visibility[query_id] >= 0.5
            xyz = np.column_stack((x[query_id], y_steps, z[query_id]))
            if mask.sum() >= 2:
                _draw_xyz(bev, yz, xyz[mask], color="black", linewidth=1.3,
                          linestyle="--", alpha=0.85)
        bev.set_xlim(-25, 25)
    _save_pair(path, f"{frame} | final predictions after score, visibility, NMS"
               f" | {len(keep_indices)} kept | color: GT; black dashed: model", draw)


def _covered_query_scores(x, z, lanes, visible, scores, visibility):
    result = {}
    for lane_id in np.flatnonzero(visible.sum(axis=1) >= 2):
        mask = visible[lane_id]
        distance = np.hypot(x[:, mask] - lanes[lane_id, mask, 0],
                            z[:, mask] - lanes[lane_id, mask, 2])
        matched = np.flatnonzero((distance < 0.3).mean(axis=1) >= 0.8)
        result[str(lane_id)] = dict(
            covered_query_count=int(len(matched)),
            max_covered_query_score=(round(float(scores[matched].max()), 4)
                                     if len(matched) else None),
            max_covered_query_visible_anchors=(
                int((visibility[matched[np.argmax(scores[matched])]] >= 0.5).sum())
                if len(matched) else None))
    return result


def _analyze_model(cfg, checkpoint, image, lanes, visible, targets,
                   output, model):
    batch = _build_sample(cfg, image)
    device = next(model.parameters()).device
    inputs = dict(img=batch["img"].to(device),
                  lidar=[points.to(device) for points in batch["lidar"]],
                  img_metas=batch["img_metas"])
    with torch.inference_mode():
        image_features, lidar_features, metas, _, _ = model._encode_batch(inputs)
        head = model.lane_head
        prior = head._anchor_prior(image_features, lidar_features, metas)
        bev = head.forward_bev(image_features, lidar_features, metas)
        occupancy = head._lidar_occupancy(lidar_features, bev)
        decoded = head._predict(
            bev, anchor_prior=prior,
            query_feature_sampler=head._query_feature_sampler(
                image_features, bev, metas, occupancy))
        layer0 = decoded["aux_predictions"][0]
        scores = decoded["pred_cls"].sigmoid()[0]
        visibility = decoded["pred_vis"].sigmoid()[0]
        score_mask = scores >= head.score_threshold
        visible_mask = (visibility >= 0.5).sum(dim=-1) >= 2
        candidate = score_mask & visible_mask
        nms_cfg = head.curve_nms_cfg
        if nms_cfg and nms_cfg.get("mean_distance_threshold") is not None:
            keep = mean_distance_curve_nms(
                decoded["pred_x"], decoded["pred_z"],
                decoded["pred_vis"].sigmoid(), decoded["pred_cls"].sigmoid(),
                distance_threshold=float(nms_cfg["mean_distance_threshold"]),
                visibility_threshold=0.5,
                candidate_mask=candidate.unsqueeze(0))[0]
        else:
            keep = torch.ones_like(candidate)

    y_steps = _numpy(head.anchor_y_steps)
    q0 = _numpy(prior["anchor_points"][0])
    x0, z0 = q0[..., 0], q0[..., 2]
    x1, z1 = _numpy(layer0["pred_x"][0]), _numpy(layer0["pred_z"][0])
    x2, z2 = _numpy(decoded["pred_x"][0]), _numpy(decoded["pred_z"][0])
    selected = _numpy(decoded["selected_query_indices"][0])
    masks = {
        "q0": np.arange(len(x0)),
        "layer0": np.arange(len(x1)),
        "layer0_selected64": selected,
        "decoder64": np.arange(len(x2)),
        "score_pass": _numpy(torch.where(score_mask)[0]),
        "score_and_visibility_pass": _numpy(torch.where(candidate)[0]),
        "after_nms": _numpy(torch.where(candidate & keep)[0]),
    }
    stages = {}
    for name, indices in masks.items():
        x, z = (x0, z0) if name == "q0" else (
            (x1, z1) if name.startswith("layer0") else (x2, z2))
        stages[name] = _geometry_stage(x, z, lanes, visible, indices)
    frame = image.stem
    template_matches = _plot_templates(
        output / "model_templates", frame,
        _numpy(prior["shape_points"][0]), targets)
    _plot_stage(output / "model_q0", frame, "Q0 505 proposals",
                x0, z0, lanes, visible, stages["q0"], y_steps)
    _plot_stage(output / "model_decoder", frame, "decoder 64 queries",
                x2, z2, lanes, visible, stages["decoder64"], y_steps,
                _numpy(visibility))
    _plot_final_predictions(output / "model_after_nms", frame, x2, z2,
                            lanes, visible, y_steps, _numpy(visibility),
                            masks["after_nms"])
    threshold_sweep = {}
    for threshold in (0.2, 0.3, 0.4, 0.5, 0.6):
        mask = (scores >= threshold) & visible_mask
        if nms_cfg and nms_cfg.get("mean_distance_threshold") is not None:
            threshold_keep = mean_distance_curve_nms(
                decoded["pred_x"], decoded["pred_z"],
                decoded["pred_vis"].sigmoid(),
                decoded["pred_cls"].sigmoid(),
                distance_threshold=float(nms_cfg["mean_distance_threshold"]),
                visibility_threshold=0.5,
                candidate_mask=mask.unsqueeze(0))[0]
        else:
            threshold_keep = torch.ones_like(mask)
        kept = _numpy(torch.where(mask & threshold_keep)[0])
        threshold_sweep[str(threshold)] = dict(
            kept_predictions=int(len(kept)),
            geometry_coverage=_stage_summary(
                _geometry_stage(x2, z2, lanes, visible, kept)))
    return dict(
        template_matches=template_matches,
        stage_coverage={key: _stage_summary(value) for key, value in stages.items()},
        per_lane=stages,
        decoder_covered_query_scores=_covered_query_scores(
            x2, z2, lanes, visible, _numpy(scores), _numpy(visibility)),
        score_threshold_sweep=threshold_sweep,
        score_threshold=head.score_threshold,
        counts={key: int(len(value)) for key, value in masks.items()},
        layer0_selected_indices=selected.tolist(),
        note="Coverage uses >=80% GT-visible anchors within 0.3 m x-z; "
             "this is Q0-style coverage, not official evaluator Recall.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work_dirs/shape_checkpoint_diagnostic_epoch18")
    parser.add_argument("--gt-only", action="store_true",
                        help="Audit target construction when CUDA is unavailable")
    parser.add_argument("--cpu-native", action="store_true",
                        help="Run exact checkpoint inference with spconv Native on CPU")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    snapshot = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.fromstring(snapshot["meta"]["cfg"], ".py")
    del snapshot
    if not args.gt_only and not args.cpu_native and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --cpu-native for exact CPU inference.")
    device = torch.device("cpu" if args.cpu_native else "cuda:0")
    model = None if args.gt_only else _build_model(cfg, checkpoint, device)
    if args.cpu_native and model is not None:
        from spconv.core import ConvAlgo
        from spconv.pytorch.conv import SparseConvolution
        for module in model.modules():
            if isinstance(module, SparseConvolution):
                module.algo = ConvAlgo.Native
    results = dict(checkpoint=str(checkpoint), checkpoint_epoch=18,
                   model_inference=not args.gt_only,
                   inference_backend=("none" if args.gt_only else
                                      "cpu_native" if args.cpu_native else "cuda"),
                   scenes={})
    for scene in SCENES:
        image = ROOT / "data/openlane/images/validation" / f"{scene}.jpg"
        output = args.output_dir.resolve() / image.stem
        output.mkdir(parents=True, exist_ok=True)
        lanes, visible, targets = _teacher_data(cfg, image)
        _plot_teacher(output / "gt_full_targets", image.stem,
                      lanes, visible, targets)
        result = dict(
            visible_lane_count=int((visible.sum(axis=1) >= 2).sum()),
            full_target_lane_count=int(targets["gt_lanes_full_valid"].sum()),
            full_target_mode_count=int(targets["gt_shape_template_valid"].sum()),
            visible_anchor_counts={str(i): int(visible[i].sum()) for i in
                                   np.flatnonzero(visible.sum(axis=1) >= 2)})
        result["aggregation_audit"] = _aggregation_audit(
            lanes, visible,
            np.asarray(cfg.model.lane_head.lane_decoder.anchor_y_steps),
            int(cfg.train_dataloader.dataset.dataset_cfg.full_shape_min_points),
            float(cfg.train_dataloader.dataset.dataset_cfg.full_shape_min_span))
        oracle_x, oracle_z, oracle_stage, oracle_z_shift = (
            _teacher_template_replay(lanes, visible, targets))
        result["teacher_template_replay"] = dict(
            coverage=_stage_summary(oracle_stage),
            per_lane=oracle_stage,
            with_oracle_z_translation=oracle_z_shift,
            note="Uses only supervised fitted GT modes and the real 101 x seeds. "
                 "It is a target replay, not checkpoint inference; the model's "
                 "unsupervised template slots may learn additional shapes.")
        _plot_stage(output / "gt_teacher_q0_replay", image.stem,
                    "fitted GT mode x-seed replay", oracle_x, oracle_z,
                    lanes, visible, oracle_stage,
                    np.asarray(cfg.model.lane_head.lane_decoder.anchor_y_steps))
        if model is not None:
            result.update(_analyze_model(cfg, checkpoint, image, lanes,
                                         visible, targets, output, model))
        results["scenes"][image.stem] = result
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
