from __future__ import annotations

import json
from pathlib import Path

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
            self.save_project(
                ProjectConfig(
                    venue_map=self.load_venue(),
                    cameras=self.load_cameras(),
                )
            )

        return self.load_project()

    def load_project(self) -> ProjectConfig:
        if self.project_path.exists():
            project = ProjectConfig.from_dict(json.loads(self.project_path.read_text(encoding="utf-8")))
        else:
            project = ProjectConfig()
        if not project.cameras:
            project.cameras = self.load_cameras()
        if not project.venue_map.zones and self.venue_path.exists():
            project.venue_map = self.load_venue()
        self._migrate_project(project)
        return project

    def save_project(self, project: ProjectConfig) -> None:
        self.save_venue(project.venue_map)
        self.save_cameras(project.cameras)
        self.project_path.write_text(
            json.dumps(project.to_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def load_venue(self) -> VenueMapConfig:
        return VenueMapConfig.from_dict(json.loads(self.venue_path.read_text(encoding="utf-8")))

    def save_venue(self, venue_map: VenueMapConfig) -> None:
        self.venue_path.write_text(
            json.dumps(venue_map.to_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def load_camera(self, camera_id: str) -> CameraConfig:
        path = self.cameras_dir / f"{camera_id}.json"
        return CameraConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_first_camera(self) -> CameraConfig:
        camera_files = sorted(self.cameras_dir.glob("*.json"))
        if not camera_files:
            default_camera = CameraConfig()
            self.save_camera(default_camera)
            return default_camera
        return CameraConfig.from_dict(json.loads(camera_files[0].read_text(encoding="utf-8")))

    def load_cameras(self) -> list[CameraConfig]:
        camera_files = sorted(self.cameras_dir.glob("*.json"))
        if not camera_files:
            default_camera = CameraConfig()
            self.save_camera(default_camera)
            return [default_camera]
        cameras = [
            CameraConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in camera_files
        ]
        return sorted(cameras, key=lambda camera: (camera.display_order, camera.camera_id))

    def save_camera(self, camera_config: CameraConfig, path: Path | None = None) -> None:
        destination = path or self.cameras_dir / f"{camera_config.camera_id}.json"
        destination.write_text(
            json.dumps(camera_config.to_dict(), indent=2, ensure_ascii=True),
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

    @staticmethod
    def _migrate_project(project: ProjectConfig) -> None:
        project.cameras = sorted(project.cameras, key=lambda camera: (camera.display_order, camera.camera_id))
        for display_order, camera in enumerate(project.cameras):
            camera.display_order = display_order
            if camera.coverage_polygon_image:
                coverage_result = recompute_camera_coverage(camera)
                camera.coverage_polygon_world_raw = coverage_result.raw_polygon_world or None
                camera.coverage_polygon_world = coverage_result.sanitized_polygon_world or None
                warnings = list(dict.fromkeys(
                    [warning for warning in [camera.calibration_warning_text, camera.coverage_warning_text] if warning]
                    + coverage_result.warnings
                ))
                camera.coverage_warning_text = " | ".join(coverage_result.warnings)
                camera.calibration_warning_text = " | ".join(warnings)
                camera.calibration_valid = bool(camera.homography_image_to_world) and coverage_result.is_valid
            else:
                camera.coverage_polygon_world_raw = None
                camera.coverage_polygon_world = None
        if project.camera_layout.selected_camera_id is None and project.cameras:
            project.camera_layout.selected_camera_id = project.cameras[0].camera_id
