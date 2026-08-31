"""Three-line training-progress logger compatible with LaneLoom losses."""
from collections import deque
from pathlib import Path
import time

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class TrainingLoggerHook(Hook):
    """Log epoch progress, ETA, and configurable training-loss detail."""

    priority = 'VERY_LOW'

    DEFAULT_BRIEF_METRICS = (
        'loss_score',
        'loss_category',
        'loss_x',
        'loss_z',
        'loss_visibility',
        'loss_tube',
        'loss_lidar_shape_cover',
        'loss_lidar_shape_mode',
        'loss_lidar_shape_diversity',
        'loss_q0_geometry',
    )

    def __init__(self, interval=50, mode='detailed', brief_metrics=None,
                 window_size=50):
        if mode not in ('brief', 'detailed'):
            raise ValueError(
                f'Logger mode must be "brief" or "detailed", got {mode!r}.')
        if isinstance(brief_metrics, str):
            raise TypeError('brief_metrics must be a sequence of metric names.')
        if (isinstance(window_size, bool)
                or not isinstance(window_size, int)
                or window_size <= 0):
            raise ValueError('window_size must be a positive integer.')
        self.interval = interval
        self.mode = mode
        self.brief_metrics = tuple(
            self.DEFAULT_BRIEF_METRICS
            if brief_metrics is None else brief_metrics)
        if not all(isinstance(name, str) for name in self.brief_metrics):
            raise TypeError('brief_metrics must contain only metric names.')
        self.window_size = window_size
        self.metric_windows = {}
        self.start_time = None
        self.val_start_time = None
        self.test_start_time = None

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(seconds))
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

    @staticmethod
    def _as_float(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().mean().cpu())
        return float(value)

    def before_train(self, runner):
        self.start_time = time.monotonic()
        self.metric_windows.clear()

    def _update_metric_windows(self, outputs):
        for name in ('loss', *self.brief_metrics):
            if name not in outputs:
                continue
            window = self.metric_windows.setdefault(
                name, deque(maxlen=self.window_size))
            value = outputs[name]
            if isinstance(value, torch.Tensor):
                value = value.detach().mean()
            else:
                value = float(value)
            window.append(value)

    def _window_mean(self, name):
        values = self.metric_windows[name]
        if isinstance(values[0], torch.Tensor):
            return torch.stack(tuple(values)).mean()
        return sum(values) / len(values)

    def _console_metrics(self, outputs):
        if self.mode == 'detailed':
            return (
                (name, value) for name, value in outputs.items()
                if name != 'loss')
        return (
            (name, self._window_mean(name))
            for name in self.brief_metrics if self.metric_windows.get(name))

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if runner.rank != 0:
            return
        outputs = outputs or {}
        if self.mode == 'brief':
            self._update_metric_windows(outputs)
        if not self.every_n_train_iters(runner, self.interval):
            return
        elapsed = time.monotonic() - self.start_time
        completed = runner.iter + 1
        total_iters = max(runner.max_iters, 1)
        expected = elapsed * total_iters / max(completed, 1)
        eta = expected - elapsed
        loader_len = len(runner.train_dataloader)
        progress = 100.0 * (batch_idx + 1) / max(loader_len, 1)
        losses = [
            f'{name}:{self._as_float(value):.4f}'
            for name, value in self._console_metrics(outputs)
        ]
        if self.mode == 'brief' and self.metric_windows.get('loss'):
            loss_window = self.metric_windows['loss']
            total_loss = self._as_float(self._window_mean('loss'))
            loss_label = f'Loss(avg{len(loss_window)})'
        else:
            total_loss = self._as_float(outputs.get('loss', 0.0))
            loss_label = 'Loss'
        scalar_losses = {
            f'train/{name}': self._as_float(value)
            for name, value in outputs.items()
        }
        runner.visualizer.add_scalars(scalar_losses, step=runner.iter + 1)
        runner.logger.info(
            'Epoch: [{}/{}][{}/{}] {:.2f}%\n'
            'Elapsed/Expected {} / {}    ETA {}\n'
            '{} {:.8f} | {}'.format(
                runner.epoch + 1, runner.max_epochs, batch_idx + 1,
                loader_len, progress, self._format_duration(elapsed),
                self._format_duration(expected), self._format_duration(eta),
                loss_label, total_loss, ' | '.join(losses)))

    def _log_evaluation_progress(self, runner, batch_idx, dataloader,
                                 start_time, label):
        total = len(dataloader)
        completed = batch_idx + 1
        if (runner.rank != 0 or
                (completed % self.interval and completed != total)):
            return
        elapsed = time.monotonic() - start_time
        expected = elapsed * total / max(completed, 1)
        eta = expected - elapsed
        runner.logger.info(
            '{}: [{}/{}] {:.2f}%\n'
            'Elapsed/Expected {} / {}    ETA {}\n'
            'Time {:.2f} s/iter'.format(
                label, completed, total,
                100.0 * completed / max(total, 1),
                self._format_duration(elapsed),
                self._format_duration(expected),
                self._format_duration(eta),
                elapsed / max(completed, 1)))

    def before_val_epoch(self, runner):
        self.val_start_time = time.monotonic()

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Show validation progress while the metric accumulates predictions."""
        self._log_evaluation_progress(
            runner, batch_idx, runner.val_dataloader,
            self.val_start_time, 'Val')

    def before_test_epoch(self, runner):
        self.test_start_time = time.monotonic()

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Show progress and ETA for the selected checkpoint test case."""
        case_name = Path(runner.work_dir).parent.name
        self._log_evaluation_progress(
            runner, batch_idx, runner.test_dataloader,
            self.test_start_time, f'Test {case_name}')

    @staticmethod
    def _metric_value(metrics, name):
        for key, value in metrics.items():
            if key == f'OpenLane/{name}' or key.endswith(f'/{name}'):
                return float(value)
        return 0.0

    def after_val_epoch(self, runner, metrics=None):
        """Write and print the independently aggregated lane metrics."""
        if runner.rank != 0 or not metrics:
            return
        runner.visualizer.add_scalars(
            {f'val/{name}': self._as_float(value)
             for name, value in metrics.items()}, step=runner.iter)
        self._log_evaluation_summary(runner, metrics)

    def _log_evaluation_summary(self, runner, metrics):
        if any(key.startswith('ONCEOfficial/') for key in metrics):
            self._log_once_evaluation_summary(runner, metrics)
            return
        self._log_openlane_evaluation_summary(runner, metrics)

    def _log_once_evaluation_summary(self, runner, metrics):
        runner.logger.info(
            'F1 = {:.4f}%\n'
            'Precision = {:.4f}%\n'
            'Recall = {:.4f}%\n'
            'CDE = {:.4f} cm'.format(
                100.0 * self._metric_value(metrics, 'best_F1'),
                100.0 * self._metric_value(metrics, 'best_Precision'),
                100.0 * self._metric_value(metrics, 'best_Recall'),
                100.0 * self._metric_value(metrics, 'best_CDE')))

    def _log_openlane_evaluation_summary(self, runner, metrics):
        runner.logger.info(
            '===> Evaluation laneline F1: {:.8f}\n'
            '===> Evaluation laneline Recall: {:.8f}\n'
            '===> Evaluation laneline Precision: {:.8f}\n'
            '===> Evaluation laneline Category Accuracy: {:.8f}\n'
            '===> Evaluation laneline GT / Pred / TP: {:.0f} / {:.0f} / {:.0f}\n'
            '===> Evaluation predictions before NMS / suppressed / ratio: '
            '{:.0f} / {:.0f} / {:.4f}\n'
            '===> Evaluation laneline x error (close): {:.8f} m\n'
            '===> Evaluation laneline x error (far): {:.8f} m\n'
            '===> Evaluation laneline z error (close): {:.8f} m\n'
            '===> Evaluation laneline z error (far): {:.8f} m'.format(
                self._metric_value(metrics, 'F1'),
                self._metric_value(metrics, 'Recall'),
                self._metric_value(metrics, 'Precision'),
                self._metric_value(metrics, 'Category Accuracy'),
                self._metric_value(metrics, 'num_gt'),
                self._metric_value(metrics, 'num_pred'),
                self._metric_value(metrics, 'num_tp'),
                self._metric_value(metrics, 'num_pred_before_nms'),
                self._metric_value(metrics, 'num_nms_suppressed'),
                self._metric_value(metrics, 'nms_suppression_ratio'),
                self._metric_value(metrics, 'x_error_near'),
                self._metric_value(metrics, 'x_error_far'),
                self._metric_value(metrics, 'z_error_near'),
                self._metric_value(metrics, 'z_error_far')))

    def after_test_epoch(self, runner, metrics=None):
        """Write and print final per-case test metrics."""
        if runner.rank != 0 or not metrics:
            return
        runner.visualizer.add_scalars(
            {f'test/{name}': self._as_float(value)
             for name, value in metrics.items()}, step=runner.iter)
        if sum(name.endswith('/F1') for name in metrics) == 1:
            self._log_evaluation_summary(runner, metrics)
