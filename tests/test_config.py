import json
import tempfile
import unittest
from pathlib import Path

from core.config import ConfigRepository
from core.models import CameraConfig, ProjectConfig, ReIDConfig, VenueMapConfig


class ConfigCompatibilityTests(unittest.TestCase):
    def test_old_udp_only_config_loads_into_product_camera_fields(self) -> None:
        config = CameraConfig.from_dict(
            {
                "camera_id": "camera-1",
                "udp_url": "udp://0.0.0.0:5000",
            }
        )
        self.assertEqual(config.source_type, "udp")
        self.assertEqual(config.source_value, "udp://0.0.0.0:5000")
        self.assertFalse(config.loop_file)
        self.assertEqual(config.runtime_mode, "local")
        self.assertEqual(config.remote_worker_id, "")

    def test_camera_persisted_schema_contains_operational_fields_only(self) -> None:
        camera = CameraConfig.from_dict(
            {
                "camera_id": "camera-1",
                "name": "Entrance",
                "source_type": "file",
                "source_value": "/tmp/demo.mp4",
                "loop_file": True,
                "runtime_mode": "remote",
                "remote_worker_id": "edge-1",
                "detector_model_path": "models/custom.pt",
                "tracker_backend": "bytetrack",
                "coverage_polygon_world": [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0]],
                "overlap_camera_ids": ["camera-2"],
            }
        )

        dumped = camera.to_persisted_dict()

        self.assertEqual(dumped["source_type"], "file")
        self.assertEqual(dumped["remote_worker_id"], "edge-1")
        self.assertEqual(dumped["overlap_camera_ids"], ["camera-2"])
        self.assertEqual(dumped["detector_model_path"], "models/custom.pt")
        self.assertNotIn("tracker_backend", dumped)
        self.assertNotIn("coverage_polygon_world", dumped)

    def test_path_like_config_values_are_normalized_to_forward_slashes(self) -> None:
        camera = CameraConfig.from_dict(
            {
                "camera_id": "camera-1",
                "source_type": "file",
                "source_value": r"videos\demo.mp4",
                "detector_model_path": r"models\custom.pt",
            }
        )
        venue = VenueMapConfig.from_dict({"map_image_path": r"assets\venue_maps\map.png"})
        reid = ReIDConfig.from_dict({"weights_path": r"models\reid\weights.pth"})

        self.assertEqual(camera.source_value, "videos/demo.mp4")
        self.assertEqual(camera.detector_model_path, "models/custom.pt")
        self.assertEqual(camera.to_persisted_dict()["source_value"], "videos/demo.mp4")
        self.assertEqual(camera.to_persisted_dict()["detector_model_path"], "models/custom.pt")
        self.assertEqual(venue.map_image_path, "assets/venue_maps/map.png")
        self.assertEqual(venue.to_dict()["map_image_path"], "assets/venue_maps/map.png")
        self.assertEqual(reid.weights_path, "models/reid/weights.pth")
        self.assertEqual(reid.to_dict()["weights_path"], "models/reid/weights.pth")

    def test_repository_migrates_legacy_project_to_canonical_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            legacy_project = {
                "venue_map": {
                    "map_image_path": "venue.png",
                    "zones": [],
                    "metric_mode": "relative",
                },
                "cameras": [
                    {
                        "camera_id": "camera-1",
                        "name": "Camera 1",
                        "source_type": "udp",
                        "source_value": "udp://0.0.0.0:5000",
                        "panel_color": "#2563eb",
                        "tracker_backend": "bytetrack",
                    }
                ],
                "analytics": {
                    "zone_entry_min_duration_s": 0.45,
                    "zone_exit_grace_s": 0.9,
                    "live_snapshot_rate_hz": 7.0,
                },
                "distributed_runtime": {
                    "enabled": True,
                    "server_port": 6201,
                },
            }
            (config_dir / "project.json").write_text(json.dumps(legacy_project, indent=2), encoding="utf-8")

            repo = ConfigRepository(root)
            project = repo.load_project()
            repo.save_project(project)

            saved_project = json.loads((config_dir / "project.json").read_text(encoding="utf-8"))
            saved_camera = json.loads((config_dir / "cameras" / "camera-1.json").read_text(encoding="utf-8"))
            saved_venue = json.loads((config_dir / "venue.json").read_text(encoding="utf-8"))

            self.assertEqual(saved_project, project.to_persisted_dict())
            self.assertNotIn("venue_map", saved_project)
            self.assertNotIn("cameras", saved_project)
            self.assertNotIn("distributed_runtime", saved_project)
            self.assertNotIn("panel_color", saved_camera)
            self.assertNotIn("tracker_backend", saved_camera)
            self.assertEqual(saved_venue["map_image_path"], "venue.png")
            self.assertEqual(project.analytics.zone_entry_min_duration_s, 0.45)
            self.assertTrue(project.analytics.heatmap_enabled)
            self.assertEqual(project.analytics.heatmap_sample_interval_s, 0.5)
            self.assertEqual(project.analytics.heatmap_grid_columns, 160)
            self.assertTrue(project.distributed_runtime.enabled)
            self.assertEqual(project.distributed_runtime.server_port, 6201)

    def test_repository_round_trip_keeps_runtime_defaults_internal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = ConfigRepository(root)
            project = repo.ensure_defaults()
            project.analytics.zone_entry_min_duration_s = 0.6
            project.analytics.zone_exit_grace_s = 1.1
            project.analytics.heatmap_enabled = False
            project.analytics.heatmap_sample_interval_s = 2.0
            project.analytics.heatmap_grid_columns = 96
            project.analytics.heatmap_min_rows = 24
            project.analytics.heatmap_max_rows = 120
            project.cameras[0].source_type = "file"
            project.cameras[0].source_value = "/tmp/demo.mp4"
            project.cameras[0].loop_file = True
            repo.save_project(project)

            loaded = repo.load_project()
            saved_project = json.loads((root / "config" / "project.json").read_text(encoding="utf-8"))

            self.assertEqual(saved_project, loaded.to_persisted_dict())
            self.assertEqual(loaded.cameras[0].source_type, "file")
            self.assertTrue(loaded.cameras[0].loop_file)
            self.assertTrue(loaded.playback_sync.enabled_for_file_sources)
            self.assertTrue(loaded.reid.enabled)
            self.assertFalse(loaded.analytics.heatmap_enabled)
            self.assertEqual(loaded.analytics.heatmap_sample_interval_s, 2.0)
            self.assertEqual(loaded.analytics.heatmap_grid_columns, 96)
            self.assertEqual(loaded.analytics.heatmap_min_rows, 24)
            self.assertEqual(loaded.analytics.heatmap_max_rows, 120)

    def test_world_only_coverage_is_ignored_on_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            cameras_dir = config_dir / "cameras"
            cameras_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "project.json").write_text(
                json.dumps(ProjectConfig().to_persisted_dict(), indent=2),
                encoding="utf-8",
            )
            (config_dir / "venue.json").write_text(
                json.dumps(VenueMapConfig().to_dict(), indent=2),
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


if __name__ == "__main__":
    unittest.main()
