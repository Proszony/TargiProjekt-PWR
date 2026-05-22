import unittest

from core.models import CameraConfig, Detection
from core.tracking import SimpleWorldTracker


def _detection(
    anchor_x: float,
    anchor_y: float,
    bbox: tuple[int, int, int, int],
    confidence: float = 0.9,
    track_id: int | None = None,
) -> Detection:
    return Detection(
        camera_id="camera-1",
        timestamp=0.0,
        person_bbox_xyxy=bbox,
        confidence=confidence,
        ground_anchor_image=((bbox[0] + bbox[2]) / 2.0, bbox[3]),
        ground_anchor_world=(anchor_x, anchor_y),
        track_id=track_id,
    )


class TrackingTests(unittest.TestCase):
    def test_stationary_person_keeps_same_id(self) -> None:
        tracker = SimpleWorldTracker(max_world_distance_m=0.45, max_missed_frames=12, min_iou=0.1, confirmation_frames=1)

        tracks, _expired = tracker.update("camera-1", 0.0, [_detection(1.0, 1.0, (100, 100, 180, 260))])
        self.assertEqual(sorted(tracks), [1])

        tracks, _expired = tracker.update("camera-1", 0.1, [_detection(1.02, 1.01, (102, 100, 182, 260))])
        self.assertEqual(sorted(tracks), [1])
        self.assertAlmostEqual(tracks[1].smoothed_ground_anchor_world[0], 1.01, places=2)

    def test_short_detection_drop_does_not_create_new_id(self) -> None:
        tracker = SimpleWorldTracker(max_world_distance_m=0.45, max_missed_frames=12, min_iou=0.1, confirmation_frames=1)

        tracks, _expired = tracker.update("camera-1", 0.0, [_detection(1.0, 1.0, (100, 100, 180, 260))])
        self.assertEqual(sorted(tracks), [1])

        tracks, _expired = tracker.update("camera-1", 0.1, [])
        self.assertEqual(sorted(tracks), [1])
        self.assertEqual(tracks[1].missed_frames, 1)

        tracks, _expired = tracker.update("camera-1", 0.2, [_detection(1.03, 1.02, (101, 101, 181, 261))])
        self.assertEqual(sorted(tracks), [1])

    def test_far_detection_creates_new_track(self) -> None:
        tracker = SimpleWorldTracker(max_world_distance_m=0.25, max_missed_frames=12, min_iou=0.1, confirmation_frames=1)

        tracks, _expired = tracker.update("camera-1", 0.0, [_detection(1.0, 1.0, (100, 100, 180, 260))])
        self.assertEqual(sorted(tracks), [1])

        tracks, _expired = tracker.update("camera-1", 0.1, [_detection(2.0, 2.0, (300, 100, 380, 260))])
        self.assertEqual(sorted(tracks), [1, 2])

    def test_tracked_detection_uses_ultralytics_track_id(self) -> None:
        tracker = SimpleWorldTracker(max_missed_frames=12, confirmation_frames=1)

        tracks, _expired = tracker.update(
            "camera-1",
            0.0,
            [_detection(1.0, 1.0, (100, 100, 180, 260), track_id=42)],
        )

        self.assertEqual(sorted(tracks), [42])
        self.assertEqual(tracks[42].display_track_id, None)

    def test_camera_config_defaults_cover_product_fields(self) -> None:
        config = CameraConfig.from_dict({"camera_id": "camera-1"})
        self.assertEqual(config.source_type, "udp")
        self.assertEqual(config.source_value, "udp://0.0.0.0:5000")
        self.assertEqual(config.runtime_mode, "local")
        self.assertEqual(config.remote_worker_id, "")
        self.assertTrue(config.enabled)
        self.assertFalse(config.loop_file)
        self.assertEqual(config.overlap_camera_ids, [])


if __name__ == "__main__":
    unittest.main()
