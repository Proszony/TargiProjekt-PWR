import json
import tempfile
import unittest
from pathlib import Path

from core.config import ConfigRepository
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
        self.assertEqual(config.detector_family, "yolo26")
        self.assertEqual(config.detector_variant, "m")
        self.assertEqual(config.detector_model_path, "models/yolo26m.pt")
        self.assertFalse(config.detector_use_augmentation)
        self.assertEqual(config.tracker_family, "botsort")
        self.assertEqual(config.tracker_backend, "botsort")
        self.assertTrue(config.tracker_with_reid)
        self.assertTrue(config.tracker_reid_enabled)
        self.assertTrue(config.allow_auto_overlap)

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
        self.assertEqual(loaded.detector_family, "yolo")
        self.assertEqual(loaded.detector_variant, "n")
        self.assertEqual(loaded.detector_model_path, "models/yolov8n.pt")
        self.assertFalse(loaded.detector_use_augmentation)
        self.assertEqual(loaded.tracker_family, "bytetrack")
        self.assertEqual(loaded.tracker_backend, "bytetrack")
        self.assertFalse(loaded.tracker_with_reid)
        self.assertFalse(loaded.tracker_reid_enabled)
        self.assertEqual(loaded.tracker_track_buffer, 90)
        self.assertAlmostEqual(loaded.tracker_match_thresh, 0.75)
        self.assertAlmostEqual(loaded.tracker_new_track_thresh, 0.4)
        self.assertAlmostEqual(loaded.tracker_proximity_thresh, 0.55)
        self.assertAlmostEqual(loaded.tracker_appearance_thresh, 0.3)
        self.assertAlmostEqual(loaded.bbox_publish_ttl_s, 0.15)
        self.assertTrue(loaded.allow_auto_overlap)
        self.assertFalse(loaded.calibration_valid)

    def test_repository_loads_multi_camera_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = ConfigRepository(root)
            project = repo.ensure_defaults()
            self.assertEqual(len(project.cameras), 1)
            self.assertTrue(project.playback_sync.enabled_for_file_sources)
            self.assertAlmostEqual(project.playback_sync.target_fps, 30.0)
            self.assertTrue(project.reid.enabled)
            self.assertTrue(project.identity.single_camera_restitch_enabled)

            second_camera = CameraConfig(camera_id="camera-2", name="Camera 2", display_order=1)
            project.cameras.append(second_camera)
            project.playback_sync.sync_tolerance_s = 0.05
            repo.save_project(project)

            loaded = repo.load_project()
            self.assertEqual([camera.camera_id for camera in loaded.cameras], ["camera-1", "camera-2"])
            self.assertAlmostEqual(loaded.playback_sync.sync_tolerance_s, 0.05)
            self.assertTrue(loaded.reid.download_if_missing)

    def test_project_round_trip_preserves_reid_and_identity_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = ConfigRepository(root)
            project = repo.ensure_defaults()
            project.reid.enabled = False
            project.reid.backend = "torchreid_model"
            project.reid.model_name = "osnet_ain_x1_0"
            project.reid.single_camera_restitch_threshold = 0.66
            project.reid.gallery_size_per_camera = 24
            project.identity.single_camera_max_gap_s = 3.5
            project.identity.single_camera_restitch_threshold = 0.63
            repo.save_project(project)

            loaded = repo.load_project()

            self.assertFalse(loaded.reid.enabled)
            self.assertEqual(loaded.reid.backend, "torchreid_model")
            self.assertEqual(loaded.reid.model_name, "osnet_ain_x1_0")
            self.assertAlmostEqual(loaded.reid.single_camera_restitch_threshold, 0.66)
            self.assertEqual(loaded.reid.gallery_size_per_camera, 24)
            self.assertAlmostEqual(loaded.identity.single_camera_max_gap_s, 3.5)
            self.assertAlmostEqual(loaded.identity.single_camera_restitch_threshold, 0.63)

    def test_project_round_trip_preserves_map_dedup_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = ConfigRepository(root)
            project = repo.ensure_defaults()
            project.map_dedup.enabled = True
            project.map_dedup.max_distance_m = 0.72
            project.map_dedup.max_time_gap_s = 0.18
            project.map_dedup.similarity_threshold = 0.57
            project.map_dedup.margin_min = 0.03
            project.map_dedup.confirmation_frames = 1
            project.map_dedup.presence_publish_ttl_s = 0.18
            repo.save_project(project)

            loaded = repo.load_project()

            self.assertTrue(loaded.map_dedup.enabled)
            self.assertAlmostEqual(loaded.map_dedup.max_distance_m, 0.72)
            self.assertAlmostEqual(loaded.map_dedup.max_time_gap_s, 0.18)
            self.assertAlmostEqual(loaded.map_dedup.similarity_threshold, 0.57)
            self.assertAlmostEqual(loaded.map_dedup.margin_min, 0.03)
            self.assertEqual(loaded.map_dedup.confirmation_frames, 1)
            self.assertAlmostEqual(loaded.map_dedup.presence_publish_ttl_s, 0.18)

    def test_project_round_trip_preserves_overlap_reid_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = ConfigRepository(root)
            project = repo.ensure_defaults()
            project.reid.overlap_reid_min_bbox_height_px = 64
            project.reid.overlap_reid_min_confidence = 0.32
            repo.save_project(project)

            loaded = repo.load_project()

            self.assertEqual(loaded.reid.overlap_reid_min_bbox_height_px, 64)
            self.assertAlmostEqual(loaded.reid.overlap_reid_min_confidence, 0.32)

    def test_new_calibration_fields_round_trip(self) -> None:
        config = CameraConfig.from_dict(
            {
                "camera_id": "camera-1",
                "homography_image_to_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "coverage_polygon_image": [[10, 10], [20, 10], [20, 20], [10, 20]],
                "coverage_auto_generated": True,
                "coverage_confidence": 0.73,
                "coverage_warning_text": "Review polygon.",
                "calibration_rmse_px": 9.5,
                "calibration_max_error_px": 14.0,
                "calibration_warning_text": "Coverage extends outside venue bounds.",
                "calibration_valid": True,
            }
        )
        loaded = CameraConfig.from_dict(config.to_dict())
        self.assertTrue(loaded.calibration_valid)
        self.assertAlmostEqual(loaded.calibration_rmse_px or 0.0, 9.5)
        self.assertAlmostEqual(loaded.calibration_max_error_px or 0.0, 14.0)
        self.assertEqual(len(loaded.coverage_polygon_image or []), 4)
        self.assertTrue(loaded.coverage_auto_generated)
        self.assertAlmostEqual(loaded.coverage_confidence or 0.0, 0.73)
        self.assertNotIn("coverage_polygon_world", config.to_dict())
        self.assertNotIn("coverage_polygon_world_raw", config.to_dict())

    def test_world_only_coverage_is_ignored_on_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            cameras_dir = config_dir / "cameras"
            cameras_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "venue.json").write_text(
                json.dumps({"zones": []}),
                encoding="utf-8",
            )
            legacy_camera = CameraConfig(
                camera_id="camera-1",
                homography_image_to_world=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                coverage_polygon_world=[(-1.0, 0.0), (11.0, 0.0), (11.0, 11.0), (-1.0, 11.0)],
                calibration_valid=True,
            )
            (cameras_dir / "camera-1.json").write_text(
                json.dumps(legacy_camera.to_dict(), indent=2),
                encoding="utf-8",
            )
            repo = ConfigRepository(root)

            project = repo.load_project()
            camera = project.cameras[0]

            self.assertIsNone(camera.coverage_polygon_world)
            self.assertIsNone(camera.coverage_polygon_world_raw)
            self.assertIsNone(camera.coverage_polygon_image)

    def test_legacy_osnet_backend_alias_maps_to_torchreid_osnet(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = ConfigRepository(root)
            project = repo.ensure_defaults()
            project.reid.backend = "osnet"
            repo.save_project(project)

            loaded = repo.load_project()

            self.assertEqual(loaded.reid.backend, "torchreid_osnet")


if __name__ == "__main__":
    unittest.main()
