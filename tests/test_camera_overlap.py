import unittest

from core.camera_overlap import build_camera_overlap_graph
from core.models import CameraConfig, OverlapDedupConfig


class CameraOverlapTests(unittest.TestCase):
    def test_overlap_relation_contains_intersection_polygon(self) -> None:
        cameras = [
            CameraConfig(
                camera_id="camera-a",
                calibration_valid=True,
                coverage_polygon_world=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
            ),
            CameraConfig(
                camera_id="camera-b",
                calibration_valid=True,
                coverage_polygon_world=[(2.0, 1.0), (6.0, 1.0), (6.0, 5.0), (2.0, 5.0)],
            ),
        ]

        graph = build_camera_overlap_graph(cameras, OverlapDedupConfig())
        relation = graph.relation_for("camera-a", "camera-b")

        self.assertIsNotNone(relation)
        assert relation is not None
        self.assertGreater(relation.overlap_area_m2, 0.0)
        self.assertGreaterEqual(len(relation.intersection_polygon_world), 3)

    def test_adjacent_without_real_overlap_has_empty_intersection_polygon(self) -> None:
        cameras = [
            CameraConfig(
                camera_id="camera-a",
                calibration_valid=True,
                coverage_polygon_world=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
            ),
            CameraConfig(
                camera_id="camera-b",
                calibration_valid=True,
                coverage_polygon_world=[(4.4, 0.0), (8.4, 0.0), (8.4, 4.0), (4.4, 4.0)],
            ),
        ]
        fusion = OverlapDedupConfig(boundary_gap_m=0.5)

        graph = build_camera_overlap_graph(cameras, fusion)
        relation = graph.relation_for("camera-a", "camera-b")

        self.assertIsNotNone(relation)
        assert relation is not None
        self.assertTrue(relation.is_adjacent)
        self.assertEqual(relation.intersection_polygon_world, [])

    def test_invalid_calibration_cameras_do_not_produce_overlap(self) -> None:
        cameras = [
            CameraConfig(
                camera_id="camera-a",
                calibration_valid=False,
                coverage_polygon_world=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
            ),
            CameraConfig(
                camera_id="camera-b",
                calibration_valid=True,
                coverage_polygon_world=[(2.0, 1.0), (6.0, 1.0), (6.0, 5.0), (2.0, 5.0)],
            ),
        ]

        graph = build_camera_overlap_graph(cameras, OverlapDedupConfig())
        relation = graph.relation_for("camera-a", "camera-b")

        self.assertIsNotNone(relation)
        assert relation is not None
        self.assertFalse(relation.is_adjacent)
        self.assertEqual(relation.overlap_area_m2, 0.0)


if __name__ == "__main__":
    unittest.main()
