import sys
from pathlib import Path

import numpy as np


VIS_INPUT = Path(__file__).resolve().parents[1] / "vis/input"
if str(VIS_INPUT) not in sys.path:
    sys.path.insert(0, str(VIS_INPUT))

from export_aligned_pointpainting import align_ground_points_to_current  # noqa: E402


def test_aligns_ground_points_through_vehicle_global_poses():
    points_ground = np.asarray([[2.0, 5.0, 1.0]], dtype=np.float64)
    source_vehicle_to_global = np.eye(4, dtype=np.float64)
    source_vehicle_to_global[:3, 3] = [10.0, 2.0, 3.0]
    current_vehicle_to_global = np.eye(4, dtype=np.float64)

    aligned = align_ground_points_to_current(
        points_ground,
        source_vehicle_to_global,
        current_vehicle_to_global,
    )

    np.testing.assert_allclose(aligned, [[0.0, 15.0, 4.0]], atol=1e-8)
