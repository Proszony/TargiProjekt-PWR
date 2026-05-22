from __future__ import annotations

import json
from pathlib import Path

from core import runtime_defaults as rd
from core.calibration import recompute_camera_coverage
from core.models import CameraConfig, ProjectConfig, VenueMapConfig


class ConfigRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.config_dir = root / "config"
        self.cameras_dir = self.config_dir / "cameras"
        self.venue_path = self.config_dir / "venue.json"
        self.project_path = self.config_dir / "project.json"

    def ensure_defaults(self) -> ProjectConfig:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.cameras_dir.mkdir(parents=True, exist_ok=True)

        if not self.venue_path.exists():
            self.save_venue(VenueMapConfig())

        default_camera_path = self.cameras_dir / "camera-1.json"
        if not default_camera_path.exists():
            self.save_camera(CameraConfig(), path=default_camera_path)

        if not self.project_path.exists():
            self.save_project(ProjectConfig())

        project = self.load_project()
        self.save_project(project)
        return project

    def load_project(self) -> ProjectConfig:
        project_payload = self._load_json(self.project_path) if self.project_path.exists() else {}
        project = ProjectConfig.from_dict(project_payload)
        project.venue_map = self._load_venue_with_fallback(project_payload)
        project.cameras = self._load_cameras_with_fallback(project_payload)
        self._migrate_project(project)
        return project

    def save_project(self, project: ProjectConfig) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.cameras_dir.mkdir(parents=True, exist_ok=True)
        self.save_venue(project.venue_map)
        self.save_cameras(project.cameras)
        self.project_path.write_text(
            json.dumps(project.to_persisted_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def load_venue(self) -> VenueMapConfig:
        payload = self._load_json(self.venue_path)
        return VenueMapConfig.from_dict(payload)

    def save_venue(self, venue_map: VenueMapConfig) -> None:
        self.venue_path.write_text(
            json.dumps(venue_map.to_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def load_camera(self, camera_id: str) -> CameraConfig:
        path = self.cameras_dir / f"{camera_id}.json"
        return CameraConfig.from_dict(self._load_json(path))

    def load_first_camera(self) -> CameraConfig:
        camera_files = sorted(self.cameras_dir.glob("*.json"))
        if not camera_files:
            default_camera = CameraConfig()
            self.save_camera(default_camera)
            return default_camera
        return CameraConfig.from_dict(self._load_json(camera_files[0]))

    def load_cameras(self) -> list[CameraConfig]:
        camera_files = sorted(self.cameras_dir.glob("*.json"))
        if not camera_files:
            default_camera = CameraConfig()
            self.save_camera(default_camera)
            return [default_camera]
        cameras = [CameraConfig.from_dict(self._load_json(path)) for path in camera_files]
        return sorted(cameras, key=lambda camera: (camera.display_order, camera.camera_id))

    def save_camera(self, camera_config: CameraConfig, path: Path | None = None) -> None:
        destination = path or self.cameras_dir / f"{camera_config.camera_id}.json"
        destination.write_text(
            json.dumps(camera_config.to_persisted_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def save_cameras(self, cameras: list[CameraConfig]) -> None:
        self.cameras_dir.mkdir(parents=True, exist_ok=True)
        existing = {path.stem: path for path in self.cameras_dir.glob("*.json")}
        valid_ids = {camera.camera_id for camera in cameras}
        for camera in cameras:
            self.save_camera(camera)
        for camera_id, path in existing.items():
            if camera_id not in valid_ids:
                path.unlink(missing_ok=True)

    def _load_venue_with_fallback(self, project_payload: dict[str, object]) -> VenueMapConfig:
        if self.venue_path.exists():
            return self.load_venue()
        legacy_venue = project_payload.get("venue_map", {})
        if isinstance(legacy_venue, dict):
            return VenueMapConfig.from_dict(legacy_venue)
        return VenueMapConfig()

    def _load_cameras_with_fallback(self, project_payload: dict[str, object]) -> list[CameraConfig]:
        camera_files = sorted(self.cameras_dir.glob("*.json"))
        if camera_files:
            cameras = [CameraConfig.from_dict(self._load_json(path)) for path in camera_files]
            return sorted(cameras, key=lambda camera: (camera.display_order, camera.camera_id))

        legacy_cameras = project_payload.get("cameras", [])
        if isinstance(legacy_cameras, list) and legacy_cameras:
            cameras = [
                CameraConfig.from_dict(item)
                for item in legacy_cameras
                if isinstance(item, dict)
            ]
            return sorted(cameras, key=lambda camera: (camera.display_order, camera.camera_id))
        return [CameraConfig()]

    @staticmethod
    def _load_json(path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _migrate_project(project: ProjectConfig) -> None:
        project.cameras = sorted(project.cameras, key=lambda camera: (camera.display_order, camera.camera_id))
        for display_order, camera in enumerate(project.cameras):
            camera.display_order = display_order
            if camera.runtime_mode not in {"local", "remote"}:
                camera.runtime_mode = "local"
            if camera.source_type not in {"udp", "file"}:
                camera.source_type = "udp"
            if not camera.source_value:
                camera.source_value = "udp://0.0.0.0:5000"
            if camera.runtime_mode == "local":
                camera.remote_worker_id = ""
            if camera.coverage_polygon_image:
                coverage_result = recompute_camera_coverage(camera)
                camera.coverage_polygon_world_raw = coverage_result.raw_polygon_world or None
                camera.coverage_polygon_world = coverage_result.sanitized_polygon_world or None
                warnings = list(
                    dict.fromkeys(
                        [warning for warning in [camera.calibration_warning_text, camera.coverage_warning_text] if warning]
                        + coverage_result.warnings
                    )
                )
                camera.coverage_warning_text = " | ".join(coverage_result.warnings)
                camera.calibration_warning_text = " | ".join(warnings)
                camera.calibration_valid = bool(camera.homography_image_to_world) and coverage_result.is_valid
            else:
                camera.coverage_polygon_world_raw = None
                camera.coverage_polygon_world = None

        project.playback_sync.enabled_for_file_sources = bool(project.playback_sync.enabled_for_file_sources)
        project.playback_sync.target_fps = max(project.playback_sync.target_fps, 1.0)
        project.playback_sync.sync_tolerance_s = max(project.playback_sync.sync_tolerance_s, 0.001)
        project.playback_sync.late_frame_drop_threshold_s = max(
            project.playback_sync.late_frame_drop_threshold_s,
            project.playback_sync.sync_tolerance_s,
        )
        project.playback_sync.stale_packet_threshold_s = max(project.playback_sync.stale_packet_threshold_s, 0.001)
        project.playback_sync.camera_missing_timeout_s = max(project.playback_sync.camera_missing_timeout_s, 0.05)
        project.playback_sync.max_buffered_packets_per_camera = max(
            project.playback_sync.max_buffered_packets_per_camera,
            1,
        )

        project.distributed_runtime.enabled = bool(project.distributed_runtime.enabled)
        project.distributed_runtime.server_bind_host = (
            project.distributed_runtime.server_bind_host or rd.DEFAULT_DISTRIBUTED_BIND_HOST
        )
        project.distributed_runtime.server_port = max(project.distributed_runtime.server_port, 1)
        project.distributed_runtime.preview_jpeg_quality = min(
            max(project.distributed_runtime.preview_jpeg_quality, 20),
            95,
        )
        project.distributed_runtime.preview_fps = max(project.distributed_runtime.preview_fps, 0.5)
        project.distributed_runtime.worker_heartbeat_interval_s = max(
            project.distributed_runtime.worker_heartbeat_interval_s,
            0.2,
        )
        project.distributed_runtime.worker_timeout_s = max(
            project.distributed_runtime.worker_timeout_s,
            project.distributed_runtime.worker_heartbeat_interval_s,
        )
