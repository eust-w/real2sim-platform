import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from physics_metrics import shape_deformation


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    def test_rigid_motion_is_not_deformation(self):
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.assertLess(shape_deformation(self.vertices, self.vertices @ rotation + 5), 1e-12)

    def test_compression_is_deformation(self):
        self.assertGreater(shape_deformation(self.vertices, self.vertices * [1., 1., 0.5]), 0.1)


if __name__ == "__main__":
    unittest.main()
