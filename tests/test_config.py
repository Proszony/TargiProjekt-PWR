import unittest

from core.models import CameraConfig


class ConfigCompatibilityTests(unittest.TestCase):
    def test_old_udp_only_config_loads_into_new_source_fields(self) -> None:
        config = CameraConfig.from_dict(
            {
                "camera_id": "camera-1",
                "udp_url": "udp://0.0.0.0:5000",
            }
        )
        self.assertEqual(config.source_type, "udp")
        self.assertEqual(config.source_value, "udp://0.0.0.0:5000")
        self.assertFalse(config.loop_file)
        self.assertEqual(config.detector_model_path, "models/yolo26m.pt")
        self.assertFalse(config.detector_use_augmentation)
        self.assertEqual(config.tracker_backend, "botsort")
        self.assertTrue(config.tracker_reid_enabled)

    def test_new_source_fields_round_trip(self) -> None:
        config = CameraConfig.from_dict(
            {
                "camera_id": "camera-1",
                "source_type": "file",
                "source_value": "/tmp/demo.mp4",
                "loop_file": True,
                "detector_model_path": "models/yolov8n.pt",
                "detector_use_augmentation": False,
                "tracker_backend": "bytetrack",
                "tracker_reid_enabled": False,
                "tracker_track_buffer": 90,
                "tracker_match_thresh": 0.75,
                "tracker_new_track_thresh": 0.4,
                "tracker_proximity_thresh": 0.55,
                "tracker_appearance_thresh": 0.3,
            }
        )
        dumped = config.to_dict()
        loaded = CameraConfig.from_dict(dumped)
        self.assertEqual(loaded.source_type, "file")
        self.assertEqual(loaded.source_value, "/tmp/demo.mp4")
        self.assertTrue(loaded.loop_file)
        self.assertEqual(loaded.detector_model_path, "models/yolov8n.pt")
        self.assertFalse(loaded.detector_use_augmentation)
        self.assertEqual(loaded.tracker_backend, "bytetrack")
        self.assertFalse(loaded.tracker_reid_enabled)
        self.assertEqual(loaded.tracker_track_buffer, 90)
        self.assertAlmostEqual(loaded.tracker_match_thresh, 0.75)
        self.assertAlmostEqual(loaded.tracker_new_track_thresh, 0.4)
        self.assertAlmostEqual(loaded.tracker_proximity_thresh, 0.55)
        self.assertAlmostEqual(loaded.tracker_appearance_thresh, 0.3)


if __name__ == "__main__":
    unittest.main()
