"""LaneFormer runner customizations."""
from mmengine.runner import Runner


class LaneFormerRunner(Runner):
    """Write runtime logs directly into ``work_dir``."""

    @property
    def timestamp(self):
        # Runner joins work_dir and timestamp while creating log_dir.  Using
        # the current directory for that one access prevents a nested folder.
        if hasattr(self, '_experiment_name') and not hasattr(self, '_log_dir'):
            return '.'
        return self._timestamp

    def build_logger(self, log_level='INFO', log_file=None, **kwargs):
        self._log_dir = self.work_dir
        return super().build_logger(
            log_level=log_level, log_file=log_file, **kwargs)
