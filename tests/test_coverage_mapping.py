import unittest

import numpy as np

from core.coverage_mapping import default_coverage_polygon_image, propose_coverage_polygon_image


class CoverageMappingTests(unittest.TestCase):
    def test_default_polygon_stays_inside_frame(self) -> None:
        polygon = default_coverage_polygon_image(1280, 720)
        self.assertGreaterEqual(len(polygon), 4)
        for x, y in polygon:
            self.assertGreaterEqual(x, 0.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(x, 1280)
            self.assertLessEqual(y, 720)

    def test_simple_floor_frame_yields_non_empty_polygon(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:140, :] = 30
        frame[140:, :] = (135, 135, 135)
        frame[180:340, 230:410] = (60, 60, 160)
        proposal = propose_coverage_polygon_image(frame, (640, 480))
        self.assertGreaterEqual(len(proposal.polygon_image), 3)
        self.assertGreater(proposal.confidence, 0.0)

    def test_black_bars_are_excluded_from_fallback(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[80:400, 60:580] = (120, 120, 120)
        proposal = propose_coverage_polygon_image(frame, (640, 480))
        self.assertGreaterEqual(len(proposal.polygon_image), 3)
        min_y = min(point[1] for point in proposal.polygon_image)
        self.assertGreaterEqual(min_y, 70.0)


if __name__ == "__main__":
    unittest.main()
