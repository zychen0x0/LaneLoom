from pathlib import Path

from mmengine.config import Config


def test_0923_training_contract():
    config_path = (
        Path(__file__).resolve().parents[2]
        / 'configs/ablations/laneloom_openlane_image_lidar_temporal_0923.py'
    )
    cfg = Config.fromfile(config_path)

    assert cfg.model.pts_bev_encoder.pillar_size == [0.1, 0.2]
    assert (cfg.model.lane_head.bev_encoder.bev_h,
            cfg.model.lane_head.bev_encoder.bev_w) == (250, 250)
    assert cfg.model.lane_head.lane_decoder.evidence_y_steps == [
        float(y) for y in range(5, 101, 5)]
    assert cfg.model.lane_head.anchor_prior.anchor_y_steps == [
        float(y) for y in range(5, 101, 5)]
    assert cfg.model.lane_head.proposal_expander.anchor_y_steps == [
        float(y) for y in range(5, 101, 5)]
    assert cfg.model.lane_head.lane_decoder.anchor_y_steps == [
        float(y) for y in range(5, 101, 5)]
    assert cfg.model.lane_head.anchor_prior.supervise_shapes is True
    assert cfg.model.lane_head.anchor_prior.shape_supervision == 'full_templates'
    assert cfg.model.lane_head.use_enhanced_prior_gt is False
    assert cfg.model.lane_head.loss_module.use_enhanced_gt is False
    assert cfg.train_dataloader.batch_size == 8
    assert cfg.train_dataloader.dataset.dataset_cfg.load_interval == 1
    assert cfg.val_dataloader.dataset.dataset_cfg.load_interval == 1
    assert cfg.test_dataloader.dataset.dataset_cfg.load_interval == 1
    assert len(cfg.train_dataloader.dataset.dataset_cfg.anchor_y_steps) == 20
    assert cfg.train_dataloader.dataset.dataset_cfg.full_shape_template_targets is True
    assert cfg.train_dataloader.dataset.dataset_cfg.full_shape_min_points == 4
    assert cfg.train_dataloader.dataset.dataset_cfg.full_shape_min_span == 15.0
    assert cfg.train_dataloader.dataset.dataset_cfg.full_shape_guarded_short_merge is True
    assert cfg.train_dataloader.dataset.dataset_cfg.full_shape_z_extrapolation_grace_m == 20.0
    assert cfg.train_dataloader.dataset.dataset_cfg.full_shape_z_extrapolation_decay_m == 20.0
    assert cfg.work_dir.endswith('_z_confidence_full_50e')
    assert cfg.optim_wrapper.accumulative_counts == 1
    assert cfg.param_scheduler[0].type == 'LinearLR'
    assert cfg.param_scheduler[0].end == 500
    assert cfg.train_cfg.max_epochs == 50
    assert cfg.param_scheduler[1].T_max == 50
    assert cfg.val_evaluator.prefix == 'OpenLaneOfficial'
    assert cfg.val_evaluator.use_temporal_gt_supplement is False
    assert cfg.default_hooks.checkpoint.save_best == 'OpenLaneOfficial/F1'
