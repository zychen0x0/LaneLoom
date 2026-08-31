"""Explicitly register LaneLoom components with MMEngine."""


def register_all_modules():
    """Import every module that contributes to an MMEngine registry."""
    import LaneLoom.datasets  # noqa: F401
    import LaneLoom.detectors  # noqa: F401
    import LaneLoom.evaluation  # noqa: F401
    import LaneLoom.models  # noqa: F401
