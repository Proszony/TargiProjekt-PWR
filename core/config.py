from __future__ import annotations

import json
from pathlib import Path

from core.models import CameraConfig, VenueMapConfig


class ConfigRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.config_dir = root / "config"
        self.cameras_dir = self.config_dir / "cameras"
        self.venue_path = self.config_dir / "venue.json"

    def ensure_defaults(self) -> tuple[VenueMapConfig, CameraConfig]:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.cameras_dir.mkdir(parents=True, exist_ok=True)

        if not self.venue_path.exists():
            self.save_venue(VenueMapConfig())

        default_camera_path = self.cameras_dir / "camera-1.json"
        if not default_camera_path.exists():
            self.save_camera(CameraConfig(), path=default_camera_path)

        return self.load_venue(), self.load_first_camera()

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

    def save_camera(self, camera_config: CameraConfig, path: Path | None = None) -> None:
        destination = path or self.cameras_dir / f"{camera_config.camera_id}.json"
        destination.write_text(
            json.dumps(camera_config.to_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
