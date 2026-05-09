from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Point = tuple[float, float]


@dataclass(slots=True)
class CalibrationPair:
    image_point: Point
    world_point: Point

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationPair":
        return cls(
            image_point=(float(data["image_point"][0]), float(data["image_point"][1])),
            world_point=(float(data["world_point"][0]), float(data["world_point"][1])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_point": [self.image_point[0], self.image_point[1]],
            "world_point": [self.world_point[0], self.world_point[1]],
        }


@dataclass(slots=True)
class ZoneDefinition:
    zone_id: str
    name: str
    kind: str
    polygon_world: list[Point]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZoneDefinition":
        return cls(
            zone_id=str(data["zone_id"]),
            name=str(data["name"]),
            kind=str(data.get("kind", "neutral")),
            polygon_world=[
                (float(point[0]), float(point[1])) for point in data.get("polygon_world", [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "kind": self.kind,
            "polygon_world": [[x, y] for x, y in self.polygon_world],
        }


@dataclass(slots=True)
class VenueMapConfig:
    map_image_path: str = ""
    world_width_m: float = 20.0
    world_height_m: float = 12.0
    zones: list[ZoneDefinition] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VenueMapConfig":
        return cls(
            map_image_path=str(data.get("map_image_path", "")),
            world_width_m=float(data.get("world_width_m", 20.0)),
            world_height_m=float(data.get("world_height_m", 12.0)),
            zones=[ZoneDefinition.from_dict(zone) for zone in data.get("zones", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "map_image_path": self.map_image_path,
            "world_width_m": self.world_width_m,
            "world_height_m": self.world_height_m,
            "zones": [zone.to_dict() for zone in self.zones],
        }


@dataclass(slots=True)
class CameraConfig:
    camera_id: str = "camera-1"
    name: str = "Camera 1"
    udp_url: str = "udp://0.0.0.0:5000"
    source_type: str = "udp"
    source_value: str = "udp://0.0.0.0:5000"
    loop_file: bool = False
    detector_model_path: str = "models/yolo26m.pt"
    detector_use_augmentation: bool = False
    tracker_backend: str = "botsort"
    tracker_config_path: str = "config/trackers/botsort.yaml"
    tracker_persist: bool = True
    tracker_reid_enabled: bool = True
    tracker_track_buffer: int = 60
    tracker_match_thresh: float = 0.8
    tracker_new_track_thresh: float = 0.5
    tracker_proximity_thresh: float = 0.5
    tracker_appearance_thresh: float = 0.25
    enabled: bool = True
    frame_width: int = 640
    frame_height: int = 480
    homography_image_to_world: list[list[float]] | None = None
    entry_zone_ids: list[str] = field(default_factory=list)
    exit_zone_ids: list[str] = field(default_factory=list)
    calibration_pairs: list[CalibrationPair] = field(default_factory=list)
    track_timeout_s: float = 3.0
    zone_entry_min_duration_s: float = 0.35
    return_threshold_s: float = 30.0
    global_match_distance_m: float = 1.5
    global_match_time_s: float = 2.0
    tracker_max_missed_frames: int = 24
    tracker_max_world_distance_m: float = 0.5
    tracker_max_image_distance_px: float = 85.0
    tracker_min_iou: float = 0.08
    tracker_anchor_weight: float = 0.65
    tracker_iou_weight: float = 0.25
    tracker_confidence_weight: float = 0.05
    global_reid_grace_period_s: float = 20.0
    global_reid_max_distance_m: float = 1.2
    global_reid_max_image_distance_px: float = 140.0
    global_reid_min_appearance_similarity: float = 0.72
    global_reid_location_weight: float = 0.4
    global_reid_appearance_weight: float = 0.6

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraConfig":
        homography = data.get("homography_image_to_world")
        udp_url = str(data.get("udp_url", "udp://0.0.0.0:5000"))
        source_type = str(data.get("source_type", "udp"))
        source_value = str(data.get("source_value", udp_url))
        return cls(
            camera_id=str(data.get("camera_id", "camera-1")),
            name=str(data.get("name", "Camera 1")),
            udp_url=udp_url,
            source_type=source_type,
            source_value=source_value,
            loop_file=bool(data.get("loop_file", False)),
            detector_model_path=str(data.get("detector_model_path", "models/yolo26m.pt")),
            detector_use_augmentation=bool(data.get("detector_use_augmentation", False)),
            tracker_backend=str(data.get("tracker_backend", "botsort")),
            tracker_config_path=str(data.get("tracker_config_path", "config/trackers/botsort.yaml")),
            tracker_persist=bool(data.get("tracker_persist", True)),
            tracker_reid_enabled=bool(data.get("tracker_reid_enabled", True)),
            tracker_track_buffer=int(data.get("tracker_track_buffer", 60)),
            tracker_match_thresh=float(data.get("tracker_match_thresh", 0.8)),
            tracker_new_track_thresh=float(data.get("tracker_new_track_thresh", 0.5)),
            tracker_proximity_thresh=float(data.get("tracker_proximity_thresh", 0.5)),
            tracker_appearance_thresh=float(data.get("tracker_appearance_thresh", 0.25)),
            enabled=bool(data.get("enabled", True)),
            frame_width=int(data.get("frame_width", 640)),
            frame_height=int(data.get("frame_height", 480)),
            homography_image_to_world=homography if homography else None,
            entry_zone_ids=[str(value) for value in data.get("entry_zone_ids", [])],
            exit_zone_ids=[str(value) for value in data.get("exit_zone_ids", [])],
            calibration_pairs=[
                CalibrationPair.from_dict(item) for item in data.get("calibration_pairs", [])
            ],
            track_timeout_s=float(data.get("track_timeout_s", 3.0)),
            zone_entry_min_duration_s=float(data.get("zone_entry_min_duration_s", 0.35)),
            return_threshold_s=float(data.get("return_threshold_s", 30.0)),
            global_match_distance_m=float(data.get("global_match_distance_m", 1.5)),
            global_match_time_s=float(data.get("global_match_time_s", 2.0)),
            tracker_max_missed_frames=int(data.get("tracker_max_missed_frames", 24)),
            tracker_max_world_distance_m=float(data.get("tracker_max_world_distance_m", 0.5)),
            tracker_max_image_distance_px=float(data.get("tracker_max_image_distance_px", 85.0)),
            tracker_min_iou=float(data.get("tracker_min_iou", 0.08)),
            tracker_anchor_weight=float(data.get("tracker_anchor_weight", 0.65)),
            tracker_iou_weight=float(data.get("tracker_iou_weight", 0.25)),
            tracker_confidence_weight=float(data.get("tracker_confidence_weight", 0.05)),
            global_reid_grace_period_s=float(data.get("global_reid_grace_period_s", 20.0)),
            global_reid_max_distance_m=float(data.get("global_reid_max_distance_m", 1.2)),
            global_reid_max_image_distance_px=float(
                data.get("global_reid_max_image_distance_px", 140.0)
            ),
            global_reid_min_appearance_similarity=float(
                data.get("global_reid_min_appearance_similarity", 0.72)
            ),
            global_reid_location_weight=float(data.get("global_reid_location_weight", 0.4)),
            global_reid_appearance_weight=float(data.get("global_reid_appearance_weight", 0.6)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "udp_url": self.udp_url,
            "source_type": self.source_type,
            "source_value": self.source_value,
            "loop_file": self.loop_file,
            "detector_model_path": self.detector_model_path,
            "detector_use_augmentation": self.detector_use_augmentation,
            "tracker_backend": self.tracker_backend,
            "tracker_config_path": self.tracker_config_path,
            "tracker_persist": self.tracker_persist,
            "tracker_reid_enabled": self.tracker_reid_enabled,
            "tracker_track_buffer": self.tracker_track_buffer,
            "tracker_match_thresh": self.tracker_match_thresh,
            "tracker_new_track_thresh": self.tracker_new_track_thresh,
            "tracker_proximity_thresh": self.tracker_proximity_thresh,
            "tracker_appearance_thresh": self.tracker_appearance_thresh,
            "enabled": self.enabled,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "homography_image_to_world": self.homography_image_to_world,
            "entry_zone_ids": self.entry_zone_ids,
            "exit_zone_ids": self.exit_zone_ids,
            "calibration_pairs": [pair.to_dict() for pair in self.calibration_pairs],
            "track_timeout_s": self.track_timeout_s,
            "zone_entry_min_duration_s": self.zone_entry_min_duration_s,
            "return_threshold_s": self.return_threshold_s,
            "global_match_distance_m": self.global_match_distance_m,
            "global_match_time_s": self.global_match_time_s,
            "tracker_max_missed_frames": self.tracker_max_missed_frames,
            "tracker_max_world_distance_m": self.tracker_max_world_distance_m,
            "tracker_max_image_distance_px": self.tracker_max_image_distance_px,
            "tracker_min_iou": self.tracker_min_iou,
            "tracker_anchor_weight": self.tracker_anchor_weight,
            "tracker_iou_weight": self.tracker_iou_weight,
            "tracker_confidence_weight": self.tracker_confidence_weight,
            "global_reid_grace_period_s": self.global_reid_grace_period_s,
            "global_reid_max_distance_m": self.global_reid_max_distance_m,
            "global_reid_max_image_distance_px": self.global_reid_max_image_distance_px,
            "global_reid_min_appearance_similarity": self.global_reid_min_appearance_similarity,
            "global_reid_location_weight": self.global_reid_location_weight,
            "global_reid_appearance_weight": self.global_reid_appearance_weight,
        }


@dataclass(slots=True)
class Detection:
    camera_id: str
    timestamp: float
    person_bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    ground_anchor_image: Point
    ground_anchor_world: Point | None
    track_id: int | None = None
    appearance_descriptor: list[float] = field(default_factory=list)


@dataclass(slots=True)
class LocalTrack:
    camera_id: str
    local_track_id: int
    positions_world: list[Point] = field(default_factory=list)
    last_seen_ts: float = 0.0
    velocity: Point = (0.0, 0.0)
    active_zone_id: str | None = None
    visited_zone_ids: list[str] = field(default_factory=list)
    current_bbox_xyxy: tuple[int, int, int, int] | None = None
    ground_anchor_image: Point | None = None
    ground_anchor_world: Point | None = None
    confidence: float = 0.0
    first_seen_ts: float = 0.0
    active: bool = True
    candidate_zone_id: str | None = None
    candidate_zone_since: float | None = None
    zone_entered_at: float | None = None
    missed_frames: int = 0
    last_bbox_xyxy_for_matching: tuple[int, int, int, int] | None = None
    smoothed_ground_anchor_world: Point | None = None
    match_score: float | None = None
    display_track_id: str | None = None
    appearance_descriptor: list[float] = field(default_factory=list)
    observed_frames: int = 0
    confirmed: bool = False


@dataclass(slots=True)
class GlobalTrack:
    global_track_id: str
    member_local_tracks: list[str] = field(default_factory=list)
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    positions_world: list[Point] = field(default_factory=list)
    velocity: Point = (0.0, 0.0)
    return_count: int = 0
    return_counts_by_zone: dict[str, int] = field(default_factory=dict)
    zone_visits: dict[str, int] = field(default_factory=dict)
    dwell_times: dict[str, float] = field(default_factory=dict)
    current_zone_id: str | None = None
    current_bbox_xyxy: tuple[int, int, int, int] | None = None
    ground_anchor_world: Point | None = None
    smoothed_ground_anchor_world: Point | None = None
    ground_anchor_image: Point | None = None
    camera_id: str | None = None
    active: bool = True
    appearance_descriptor: list[float] = field(default_factory=list)
    inactive_since_ts: float | None = None
    reactivation_deadline_ts: float | None = None


@dataclass(slots=True)
class AnalyticsEvent:
    event_type: str
    global_track_id: str
    zone_id: str | None
    timestamp: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ZoneMetrics:
    zone_id: str
    zone_name: str
    zone_kind: str
    current_occupancy: int = 0
    unique_entries: int = 0
    total_dwell_s: float = 0.0
    avg_dwell_s: float = 0.0
    return_count: int = 0


@dataclass(slots=True)
class AnalyticsSnapshot:
    timestamp: float
    active_zone_counts: dict[str, int] = field(default_factory=dict)
    unique_zone_entries: dict[str, int] = field(default_factory=dict)
    dwell_times: dict[str, float] = field(default_factory=dict)
    return_counts: dict[str, int] = field(default_factory=dict)
    active_global_tracks: dict[str, GlobalTrack] = field(default_factory=dict)
    recent_events: list[AnalyticsEvent] = field(default_factory=list)
    avg_dwell_times: dict[str, float] = field(default_factory=dict)
    total_entries: int = 0
    session_total_returns: int = 0
    zone_metrics: dict[str, ZoneMetrics] = field(default_factory=dict)
