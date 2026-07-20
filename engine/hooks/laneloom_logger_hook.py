"""Three-line training-progress logger compatible with LaneLoom losses."""
from pathlib import Path
import time

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class LaneLoomLoggerHook(Hook):
    """Log epoch progress, ETA, total loss, and every returned loss field."""

    priority = 'VERY_LOW'

    def __init__(self, interval=50):
        self.interval = interval
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

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if runner.rank != 0 or not self.every_n_train_iters(runner, self.interval):
            return
        outputs = outputs or {}
        elapsed = time.monotonic() - self.start_time
        completed = runner.iter + 1
        total_iters = max(runner.max_iters, 1)
        expected = elapsed * total_iters / max(completed, 1)
        eta = expected - elapsed
        loader_len = len(runner.train_dataloader)
        progress = 100.0 * (batch_idx + 1) / max(loader_len, 1)
        losses = [
            f'{name}:{self._as_float(value):.4f}'
            for name, value in outputs.items() if name != 'loss'
        ]
        total_loss = self._as_float(outputs.get('loss', 0.0))
        scalar_losses = {
            f'train/{name}': self._as_float(value)
            for name, value in outputs.items()
        }
        runner.visualizer.add_scalars(scalar_losses, step=runner.iter + 1)
        runner.logger.info(
            'Epoch: [{}/{}][{}/{}] {:.2f}%\n'
            'Elapsed/Expected {} / {}    ETA {}\n'
            'Loss {:.8f} | {}'.format(
                runner.epoch + 1, runner.max_epochs, batch_idx + 1,
                loader_len, progress, self._format_duration(elapsed),
                self._format_duration(expected), self._format_duration(eta),
                total_loss, ' | '.join(losses)))

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
        """Keep final test metrics next to training curves in TensorBoard."""
        if runner.rank != 0 or not metrics:
            return
        runner.visualizer.add_scalars(
            {f'test/{name}': self._as_float(value)
             for name, value in metrics.items()}, step=runner.iter)
