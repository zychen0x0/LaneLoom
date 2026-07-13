"""Checkpoint hook with an explicit experiment checkpoint directory."""
from collections import deque

from mmengine.dist import is_main_process
from mmengine.fileio import get_file_backend
from mmengine.hooks import CheckpointHook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class LaneFormerCheckpointHook(CheckpointHook):
    """Keep ``out_dir`` literal instead of appending the runner log directory.

    MMEngine's generic hook still creates a deprecated ``FileClient`` even
    when the modern file backend API is selected. LaneFormer uses
    ``get_file_backend`` directly and keeps the configured directory literal.
    """

    def before_train(self, runner) -> None:
        if self.out_dir is None:
            self.out_dir = runner.work_dir

        self.file_client = None
        self.file_backend = get_file_backend(
            self.out_dir, backend_args=self.backend_args)
        runner.logger.info('Checkpoints will be saved to %s.', self.out_dir)

        if self.save_best is not None:
            if len(self.key_indicators) == 1:
                self.best_ckpt_path = (
                    runner.message_hub.get_info('best_ckpt')
                    if 'best_ckpt' in runner.message_hub.runtime_info else None)
            else:
                for key_indicator in self.key_indicators:
                    best_key = f'best_ckpt_{key_indicator}'
                    self.best_ckpt_path_dict[key_indicator] = (
                        runner.message_hub.get_info(best_key)
                        if best_key in runner.message_hub.runtime_info else None)

        if self.max_keep_ckpts > 0:
            keep_ckpt_ids = []
            if 'keep_ckpt_ids' in runner.message_hub.runtime_info:
                keep_ckpt_ids = runner.message_hub.get_info('keep_ckpt_ids')
                while len(keep_ckpt_ids) > self.max_keep_ckpts:
                    step = keep_ckpt_ids.pop(0)
                    if is_main_process():
                        path = self.file_backend.join_path(
                            self.out_dir, self.filename_tmpl.format(step))
                        if self.file_backend.isfile(path):
                            self.file_backend.remove(path)
                        elif self.file_backend.isdir(path):
                            self.file_backend.rmtree(path)
            self.keep_ckpt_ids = deque(keep_ckpt_ids, self.max_keep_ckpts)
