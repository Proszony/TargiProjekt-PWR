from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Point = tuple[float, float]


@dataclass(slots=True)
class SharedAnchor:
    anchor_id: str
    name: str
    world_point: Point

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharedAnchor":
        return cls(
            anchor_id=str(data["anchor_id"]),
            name=str(data.get("name", data["anchor_id"])),
            world_point=(float(data["world_point"][0]), float(data["world_point"][1])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "name": self.name,
            "world_point": [self.world_point[0], self.world_point[1]],
        }


@dataclass(slots=True)
class CameraAnchorObservation:
    anchor_id: str
    image_point: Point

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraAnchorObservation":
        return cls(
            anchor_id=str(data["anchor_id"]),
            image_point=(float(data["image_point"][0]), float(data["image_point"][1])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "image_point": [self.image_point[0], self.image_point[1]],
        }


@dataclass(slots=True)
class CameraOverlapRelation:
    camera_a_id: str
    camera_b_id: str
    overlap_area_m2: float
    min_boundary_distance_m: float
    is_adjacent: bool
    intersection_polygon_world: list[Point] = field(default_factory=list)


@dataclass(slots=True)
class CameraOverlapOverlay:
    camera_a_id: str
    camera_b_id: str
    polygon_world: list[Point]
    overlap_area_m2: float
    label: str


@dataclass(slots=True)
class CameraCoverageOverlay:
    camera_id: str
    camera_name: str
    color: str
    polygon_world: list[Point]
    raw_polygon_world: list[Point] = field(default_factory=list)
    calibration_valid: bool = False
    calibration_warning_text: str = ""


@dataclass(slots=True)
class CameraOverlapGraph:
    relations: dict[tuple[str, str], CameraOverlapRelation] = field(default_factory=dict)

    def relation_for(self, camera_a_id: str, camera_b_id: str) -> CameraOverlapRelation | None:
        return self.relations.get((camera_a_id, camera_b_id))

    def neighbors_of(self, camera_id: str) -> set[str]:
        return {
            relation.camera_b_id
            for (camera_a_id, _camera_b_id), relation in self.relations.items()
            if camera_a_id == camera_id and relation.is_adjacent
        }


@dataclass(slots=True)
class CameraLayoutConfig:
    selected_camera_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraLayoutConfig":
        return cls(selected_camera_id=str(data["selected_camera_id"]) if data.get("selected_camera_id") else None)

    def to_dict(self) -> dict[str, Any]:
        return {"selected_camera_id": self.selected_camera_id}


@dataclass(slots=True)
class PlaybackSyncConfig:
    enabled_for_file_sources: bool = True
    target_fps: float = 30.0
    sync_tolerance_s: float = 0.040
    late_frame_drop_threshold_s: float = 0.120
    stale_packet_threshold_s: float = 0.150
    camera_missing_timeout_s: float = 0.500
    max_buffered_packets_per_camera: int = 12

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlaybackSyncConfig":
        return cls(
            enabled_for_file_sources=bool(data.get("enabled_for_file_sources", True)),
            target_fps=float(data.get("target_fps", 30.0)),
            sync_tolerance_s=float(data.get("sync_tolerance_s", 0.040)),
            late_frame_drop_threshold_s=float(data.get("late_frame_drop_threshold_s", 0.120)),
            stale_packet_threshold_s=float(data.get("stale_packet_threshold_s", 0.150)),
            camera_missing_timeout_s=float(data.get("camera_missing_timeout_s", 0.500)),
            max_buffered_packets_per_camera=int(data.get("max_buffered_packets_per_camera", 12)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled_for_file_sources": self.enabled_for_file_sources,
            "target_fps": self.target_fps,
            "sync_tolerance_s": self.sync_tolerance_s,
            "late_frame_drop_threshold_s": self.late_frame_drop_threshold_s,
            "stale_packet_threshold_s": self.stale_packet_threshold_s,
            "camera_missing_timeout_s": self.camera_missing_timeout_s,
            "max_buffered_packets_per_camera": self.max_buffered_packets_per_camera,
        }


@dataclass(slots=True)
class ReIDConfig:
    enabled: bool = True
    backend: str = "torchreid_osnet"
    model_name: str = "osnet_x1_0"
    weights_path: str = "models/reid/osnet_x1_0_msmt17.pth"
    weights_url: str = (
        "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
        "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
    )
    download_if_missing: bool = True
    min_bbox_height_px: int = 96
    min_confidence: float = 0.45
    overlap_reid_min_bbox_height_px: int = 72
    overlap_reid_min_confidence: float = 0.35
    embedding_similarity_threshold: float = 0.72
    single_camera_restitch_threshold: float = 0.68
    cross_camera_match_threshold: float = 0.74
    embedding_momentum: float = 0.85
    max_embedding_memory: int = 30
    gallery_size_per_camera: int = 30
    gallery_size_global: int = 60
    match_topk: int = 5
    match_reduce: str = "mean_top3"
    re_rank_enabled: bool = False
    input_width: int = 128
    input_height: int = 256
    input_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    input_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReIDConfig":
        backend = str(data.get("backend", "torchreid_osnet"))
        if backend == "osnet":
            backend = "torchreid_osnet"
        return cls(
            enabled=bool(data.get("enabled", True)),
            backend=backend,
            model_name=str(data.get("model_name", "osnet_x1_0")),
            weights_path=str(data.get("weights_path", "models/reid/osnet_x1_0_msmt17.pth")),
            weights_url=str(
                data.get(
                    "weights_url",
                    "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
                    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
                )
            ),
            download_if_missing=bool(data.get("download_if_missing", True)),
            min_bbox_height_px=int(data.get("min_bbox_height_px", 96)),
            min_confidence=float(data.get("min_confidence", 0.45)),
            overlap_reid_min_bbox_height_px=int(data.get("overlap_reid_min_bbox_height_px", 72)),
            overlap_reid_min_confidence=float(data.get("overlap_reid_min_confidence", 0.35)),
            embedding_similarity_threshold=float(data.get("embedding_similarity_threshold", 0.72)),
            single_camera_restitch_threshold=float(data.get("single_camera_restitch_threshold", 0.68)),
            cross_camera_match_threshold=float(data.get("cross_camera_match_threshold", 0.74)),
            embedding_momentum=float(data.get("embedding_momentum", 0.85)),
            max_embedding_memory=int(data.get("max_embedding_memory", 30)),
            gallery_size_per_camera=int(data.get("gallery_size_per_camera", 30)),
            gallery_size_global=int(data.get("gallery_size_global", 60)),
            match_topk=int(data.get("match_topk", 5)),
            match_reduce=str(data.get("match_reduce", "mean_top3")),
            re_rank_enabled=bool(data.get("re_rank_enabled", False)),
            input_width=int(data.get("input_width", 128)),
            input_height=int(data.get("input_height", 256)),
            input_mean=[float(value) for value in data.get("input_mean", [0.485, 0.456, 0.406])],
            input_std=[float(value) for value in data.get("input_std", [0.229, 0.224, 0.225])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "model_name": self.model_name,
            "weights_path": self.weights_path,
            "weights_url": self.weights_url,
            "download_if_missing": self.download_if_missing,
            "min_bbox_height_px": self.min_bbox_height_px,
            "min_confidence": self.min_confidence,
            "overlap_reid_min_bbox_height_px": self.overlap_reid_min_bbox_height_px,
            "overlap_reid_min_confidence": self.overlap_reid_min_confidence,
            "embedding_similarity_threshold": self.embedding_similarity_threshold,
            "single_camera_restitch_threshold": self.single_camera_restitch_threshold,
            "cross_camera_match_threshold": self.cross_camera_match_threshold,
            "embedding_momentum": self.embedding_momentum,
            "max_embedding_memory": self.max_embedding_memory,
            "gallery_size_per_camera": self.gallery_size_per_camera,
            "gallery_size_global": self.gallery_size_global,
            "match_topk": self.match_topk,
            "match_reduce": self.match_reduce,
            "re_rank_enabled": self.re_rank_enabled,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "input_mean": list(self.input_mean),
            "input_std": list(self.input_std),
        }


@dataclass(slots=True)
class IdentityConfig:
    single_camera_restitch_enabled: bool = True
    single_camera_max_gap_s: float = 2.0
    single_camera_base_distance_m: float = 0.8
    single_camera_speed_m_per_s: float = 1.0
    single_camera_restitch_threshold: float = 0.68
    confirmation_frames: int = 3
    lost_track_retention_s: float = 1.20

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdentityConfig":
        return cls(
            single_camera_restitch_enabled=bool(data.get("single_camera_restitch_enabled", True)),
            single_camera_max_gap_s=float(data.get("single_camera_max_gap_s", 2.0)),
            single_camera_base_distance_m=float(data.get("single_camera_base_distance_m", 0.8)),
            single_camera_speed_m_per_s=float(data.get("single_camera_speed_m_per_s", 1.0)),
            single_camera_restitch_threshold=float(data.get("single_camera_restitch_threshold", 0.68)),
            confirmation_frames=int(data.get("confirmation_frames", 3)),
            lost_track_retention_s=float(data.get("lost_track_retention_s", 1.20)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "single_camera_restitch_enabled": self.single_camera_restitch_enabled,
            "single_camera_max_gap_s": self.single_camera_max_gap_s,
            "single_camera_base_distance_m": self.single_camera_base_distance_m,
            "single_camera_speed_m_per_s": self.single_camera_speed_m_per_s,
            "single_camera_restitch_threshold": self.single_camera_restitch_threshold,
            "confirmation_frames": self.confirmation_frames,
            "lost_track_retention_s": self.lost_track_retention_s,
        }


@dataclass(slots=True)
class AnalyticsConfig:
    zone_entry_min_duration_s: float = 0.35
    zone_exit_grace_s: float = 0.75
    dedup_overlap_enabled: bool = True
    dedup_confirmation_frames: int = 3
    dedup_similarity_threshold: float = 0.74
    dedup_margin_min: float = 0.08
    live_snapshot_rate_hz: float = 4.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalyticsConfig":
        return cls(
            zone_entry_min_duration_s=float(data.get("zone_entry_min_duration_s", 0.35)),
            zone_exit_grace_s=float(data.get("zone_exit_grace_s", 0.75)),
            dedup_overlap_enabled=bool(data.get("dedup_overlap_enabled", True)),
            dedup_confirmation_frames=int(data.get("dedup_confirmation_frames", 3)),
            dedup_similarity_threshold=float(data.get("dedup_similarity_threshold", 0.74)),
            dedup_margin_min=float(data.get("dedup_margin_min", 0.08)),
            live_snapshot_rate_hz=float(data.get("live_snapshot_rate_hz", 4.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_entry_min_duration_s": self.zone_entry_min_duration_s,
            "zone_exit_grace_s": self.zone_exit_grace_s,
            "dedup_overlap_enabled": self.dedup_overlap_enabled,
            "dedup_confirmation_frames": self.dedup_confirmation_frames,
            "dedup_similarity_threshold": self.dedup_similarity_threshold,
            "dedup_margin_min": self.dedup_margin_min,
            "live_snapshot_rate_hz": self.live_snapshot_rate_hz,
        }


@dataclass(slots=True)
class OverlapDedupConfig:
    enabled: bool = True
    overlap_area_min_m2: float = 0.5
    boundary_gap_m: float = 1.2
    max_distance_m: float = 1.0
    max_time_gap_s: float = 1.0
    appearance_weight: float = 0.50
    world_weight: float = 0.30
    timing_weight: float = 0.15
    geometry_weight: float = 0.05
    similarity_threshold: float = 0.74
    margin_min: float = 0.08
    confirmation_frames: int = 3

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OverlapDedupConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            overlap_area_min_m2=float(data.get("overlap_area_min_m2", 0.5)),
            boundary_gap_m=float(data.get("boundary_gap_m", 1.2)),
            max_distance_m=float(data.get("max_distance_m", 1.0)),
            max_time_gap_s=float(data.get("max_time_gap_s", 1.0)),
            appearance_weight=float(data.get("appearance_weight", 0.50)),
            world_weight=float(data.get("world_weight", 0.30)),
            timing_weight=float(data.get("timing_weight", 0.15)),
            geometry_weight=float(data.get("geometry_weight", 0.05)),
            similarity_threshold=float(data.get("similarity_threshold", 0.74)),
            margin_min=float(data.get("margin_min", 0.08)),
            confirmation_frames=int(data.get("confirmation_frames", 3)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "overlap_area_min_m2": self.overlap_area_min_m2,
            "boundary_gap_m": self.boundary_gap_m,
            "max_distance_m": self.max_distance_m,
            "max_time_gap_s": self.max_time_gap_s,
            "appearance_weight": self.appearance_weight,
            "world_weight": self.world_weight,
            "timing_weight": self.timing_weight,
            "geometry_weight": self.geometry_weight,
            "similarity_threshold": self.similarity_threshold,
            "margin_min": self.margin_min,
            "confirmation_frames": self.confirmation_frames,
        }


@dataclass(slots=True)
class MapDedupConfig:
    enabled: bool = True
    overlap_presence_boundary_buffer_m: float = 0.50
    max_distance_m: float = 0.80
    max_time_gap_s: float = 0.25
    similarity_threshold: float = 0.60
    margin_min: float = 0.05
    confirmation_frames: int = 2
    world_alignment_weight: float = 0.55
    overlap_membership_weight: float = 0.20
    timing_weight: float = 0.20
    appearance_weight: float = 0.15
    motion_weight: float = 0.10
    tracklet_window_s: float = 0.20
    tracklet_min_points: int = 2
    tracklet_keepalive_s: float = 0.40
    overlap_batch_window_s: float = 0.20
    overlap_batch_step_s: float = 0.10
    presence_hold_s: float = 0.30
    presence_reassign_window_s: float = 0.50
    presence_publish_ttl_s: float = 0.25
    max_exact_assignment_size: int = 8

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MapDedupConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            overlap_presence_boundary_buffer_m=float(data.get("overlap_presence_boundary_buffer_m", 0.50)),
            max_distance_m=float(data.get("max_distance_m", 0.80)),
            max_time_gap_s=float(data.get("max_time_gap_s", 0.25)),
            similarity_threshold=float(data.get("similarity_threshold", 0.60)),
            margin_min=float(data.get("margin_min", 0.05)),
            confirmation_frames=int(data.get("confirmation_frames", 2)),
            world_alignment_weight=float(data.get("world_alignment_weight", 0.55)),
            overlap_membership_weight=float(data.get("overlap_membership_weight", 0.20)),
            timing_weight=float(data.get("timing_weight", 0.20)),
            appearance_weight=float(data.get("appearance_weight", 0.15)),
            motion_weight=float(data.get("motion_weight", 0.10)),
            tracklet_window_s=float(data.get("tracklet_window_s", 0.20)),
            tracklet_min_points=int(data.get("tracklet_min_points", 2)),
            tracklet_keepalive_s=float(data.get("tracklet_keepalive_s", 0.40)),
            overlap_batch_window_s=float(data.get("overlap_batch_window_s", 0.20)),
            overlap_batch_step_s=float(data.get("overlap_batch_step_s", 0.10)),
            presence_hold_s=float(data.get("presence_hold_s", 0.30)),
            presence_reassign_window_s=float(data.get("presence_reassign_window_s", 0.50)),
            presence_publish_ttl_s=float(data.get("presence_publish_ttl_s", 0.25)),
            max_exact_assignment_size=int(data.get("max_exact_assignment_size", 8)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "overlap_presence_boundary_buffer_m": self.overlap_presence_boundary_buffer_m,
            "max_distance_m": self.max_distance_m,
            "max_time_gap_s": self.max_time_gap_s,
            "similarity_threshold": self.similarity_threshold,
            "margin_min": self.margin_min,
            "confirmation_frames": self.confirmation_frames,
            "world_alignment_weight": self.world_alignment_weight,
            "overlap_membership_weight": self.overlap_membership_weight,
            "timing_weight": self.timing_weight,
            "appearance_weight": self.appearance_weight,
            "motion_weight": self.motion_weight,
            "tracklet_window_s": self.tracklet_window_s,
            "tracklet_min_points": self.tracklet_min_points,
            "tracklet_keepalive_s": self.tracklet_keepalive_s,
            "overlap_batch_window_s": self.overlap_batch_window_s,
            "overlap_batch_step_s": self.overlap_batch_step_s,
            "presence_hold_s": self.presence_hold_s,
            "presence_reassign_window_s": self.presence_reassign_window_s,
            "presence_publish_ttl_s": self.presence_publish_ttl_s,
            "max_exact_assignment_size": self.max_exact_assignment_size,
        }


@dataclass(slots=True)
class CalibrationResult:
    homography_image_to_world: list[list[float]] | None = None
    reprojection_rmse_px: float | None = None
    max_reprojection_error_px: float | None = None
    used_anchor_count: int = 0
    is_valid: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CoverageProposal:
    polygon_image: list[Point] = field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    mask_bbox_xyxy: tuple[int, int, int, int] | None = None
    method: str = "fallback"


@dataclass(slots=True)
class CameraCoverageComputationResult:
    raw_polygon_world: list[Point] = field(default_factory=list)
    sanitized_polygon_world: list[Point] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_valid: bool = False


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
class WorldViewport:
    min_x: float = 0.0
    min_y: float = 0.0
    max_x: float = 10.0
    max_y: float = 10.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldViewport":
        return cls(
            min_x=float(data.get("min_x", 0.0)),
            min_y=float(data.get("min_y", 0.0)),
            max_x=float(data.get("max_x", 10.0)),
            max_y=float(data.get("max_y", 10.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_x": self.min_x,
            "min_y": self.min_y,
            "max_x": self.max_x,
            "max_y": self.max_y,
        }


@dataclass(slots=True)
class VenueMapConfig:
    map_image_path: str = ""
    zones: list[ZoneDefinition] = field(default_factory=list)
    metric_mode: str = "relative"
    manual_viewport_override: WorldViewport | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VenueMapConfig":
        return cls(
            map_image_path=str(data.get("map_image_path", "")),
            zones=[ZoneDefinition.from_dict(zone) for zone in data.get("zones", [])],
            metric_mode=str(data.get("metric_mode", "relative")),
            manual_viewport_override=(
                WorldViewport.from_dict(data["manual_viewport_override"])
                if data.get("manual_viewport_override")
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "map_image_path": self.map_image_path,
            "zones": [zone.to_dict() for zone in self.zones],
            "metric_mode": self.metric_mode,
        }
        if self.manual_viewport_override is not None:
            payload["manual_viewport_override"] = self.manual_viewport_override.to_dict()
        return payload


@dataclass(slots=True)
class CameraConfig:
    camera_id: str = "camera-1"
    name: str = "Camera 1"
    display_order: int = 0
    panel_color: str = "#2563eb"
    udp_url: str = "udp://0.0.0.0:5000"
    source_type: str = "udp"
    source_value: str = "udp://0.0.0.0:5000"
    loop_file: bool = False
    detector_family: str = "yolo26"
    detector_variant: str = "m"
    detector_model_path: str = "models/yolo26m.pt"
    detector_use_augmentation: bool = False
    tracker_family: str = "botsort"
    tracker_backend: str = "botsort"
    tracker_config_path: str = "config/trackers/botsort.yaml"
    tracker_persist: bool = True
    tracker_with_reid: bool = True
    tracker_reid_enabled: bool = True
    tracker_track_buffer: int = 16
    tracker_match_thresh: float = 0.75
    tracker_new_track_thresh: float = 0.35
    tracker_proximity_thresh: float = 0.6
    tracker_appearance_thresh: float = 0.2
    reid_enabled: bool = True
    camera_identity_enabled: bool = True
    enabled: bool = True
    frame_width: int = 640
    frame_height: int = 480
    homography_image_to_world: list[list[float]] | None = None
    coverage_polygon_image: list[Point] | None = None
    coverage_auto_generated: bool = False
    coverage_confidence: float | None = None
    coverage_warning_text: str = ""
    coverage_polygon_world: list[Point] | None = None
    coverage_polygon_world_raw: list[Point] | None = None
    calibration_rmse_px: float | None = None
    calibration_max_error_px: float | None = None
    calibration_warning_text: str = ""
    calibration_valid: bool = False
    entry_zone_ids: list[str] = field(default_factory=list)
    exit_zone_ids: list[str] = field(default_factory=list)
    anchor_observations: list[CameraAnchorObservation] = field(default_factory=list)
    overlap_camera_ids: list[str] = field(default_factory=list)
    allow_auto_overlap: bool = True
    track_timeout_s: float = 1.20
    tracker_max_missed_frames: int = 10
    bbox_publish_ttl_s: float = 0.15
    tracker_max_world_distance_m: float = 0.5
    tracker_max_image_distance_px: float = 85.0
    tracker_min_iou: float = 0.08
    tracker_anchor_weight: float = 0.65
    tracker_iou_weight: float = 0.25
    tracker_confidence_weight: float = 0.05

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraConfig":
        homography = data.get("homography_image_to_world")
        coverage_polygon_image = [
            (float(point[0]), float(point[1]))
            for point in (data.get("coverage_polygon_image") or [])
        ] or None
        udp_url = str(data.get("udp_url", "udp://0.0.0.0:5000"))
        source_type = str(data.get("source_type", "udp"))
        source_value = str(data.get("source_value", udp_url))
        detector_model_path = str(data.get("detector_model_path", "models/yolo26m.pt"))
        detector_family = str(data.get("detector_family", _infer_detector_family(detector_model_path)))
        detector_variant = str(data.get("detector_variant", _infer_detector_variant(detector_model_path)))
        tracker_backend = str(data.get("tracker_backend", "botsort"))
        calibration_valid = bool(
            data.get(
                "calibration_valid",
                bool(homography) and len(coverage_polygon_image or []) >= 3,
            )
        )
        return cls(
            camera_id=str(data.get("camera_id", "camera-1")),
            name=str(data.get("name", "Camera 1")),
            display_order=int(data.get("display_order", 0)),
            panel_color=str(data.get("panel_color", "#2563eb")),
            udp_url=udp_url,
            source_type=source_type,
            source_value=source_value,
            loop_file=bool(data.get("loop_file", False)),
            detector_family=detector_family,
            detector_variant=detector_variant,
            detector_model_path=detector_model_path,
            detector_use_augmentation=bool(data.get("detector_use_augmentation", False)),
            tracker_family=str(data.get("tracker_family", tracker_backend)),
            tracker_backend=tracker_backend,
            tracker_config_path=str(data.get("tracker_config_path", "config/trackers/botsort.yaml")),
            tracker_persist=bool(data.get("tracker_persist", True)),
            tracker_with_reid=bool(data.get("tracker_with_reid", data.get("tracker_reid_enabled", True))),
            tracker_reid_enabled=bool(data.get("tracker_reid_enabled", True)),
            tracker_track_buffer=int(data.get("tracker_track_buffer", 16)),
            tracker_match_thresh=float(data.get("tracker_match_thresh", 0.75)),
            tracker_new_track_thresh=float(data.get("tracker_new_track_thresh", 0.35)),
            tracker_proximity_thresh=float(data.get("tracker_proximity_thresh", 0.6)),
            tracker_appearance_thresh=float(data.get("tracker_appearance_thresh", 0.2)),
            reid_enabled=bool(data.get("reid_enabled", True)),
            camera_identity_enabled=bool(data.get("camera_identity_enabled", True)),
            enabled=bool(data.get("enabled", True)),
            frame_width=int(data.get("frame_width", 640)),
            frame_height=int(data.get("frame_height", 480)),
            homography_image_to_world=homography if homography else None,
            coverage_polygon_image=coverage_polygon_image,
            coverage_auto_generated=bool(data.get("coverage_auto_generated", False)),
            coverage_confidence=(
                float(data["coverage_confidence"])
                if data.get("coverage_confidence") is not None
                else None
            ),
            coverage_warning_text=str(data.get("coverage_warning_text", "")),
            calibration_rmse_px=(
                float(data["calibration_rmse_px"])
                if data.get("calibration_rmse_px") is not None
                else None
            ),
            calibration_max_error_px=(
                float(data["calibration_max_error_px"])
                if data.get("calibration_max_error_px") is not None
                else None
            ),
            calibration_warning_text=str(data.get("calibration_warning_text", "")),
            calibration_valid=calibration_valid,
            entry_zone_ids=[str(value) for value in data.get("entry_zone_ids", [])],
            exit_zone_ids=[str(value) for value in data.get("exit_zone_ids", [])],
            anchor_observations=[
                CameraAnchorObservation.from_dict(item) for item in data.get("anchor_observations", [])
            ],
            overlap_camera_ids=[str(value) for value in data.get("overlap_camera_ids", [])],
            allow_auto_overlap=bool(data.get("allow_auto_overlap", True)),
            track_timeout_s=float(data.get("track_timeout_s", 1.20)),
            tracker_max_missed_frames=int(data.get("tracker_max_missed_frames", 10)),
            bbox_publish_ttl_s=float(data.get("bbox_publish_ttl_s", 0.15)),
            tracker_max_world_distance_m=float(data.get("tracker_max_world_distance_m", 0.5)),
            tracker_max_image_distance_px=float(data.get("tracker_max_image_distance_px", 85.0)),
            tracker_min_iou=float(data.get("tracker_min_iou", 0.08)),
            tracker_anchor_weight=float(data.get("tracker_anchor_weight", 0.65)),
            tracker_iou_weight=float(data.get("tracker_iou_weight", 0.25)),
            tracker_confidence_weight=float(data.get("tracker_confidence_weight", 0.05)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "display_order": self.display_order,
            "panel_color": self.panel_color,
            "udp_url": self.udp_url,
            "source_type": self.source_type,
            "source_value": self.source_value,
            "loop_file": self.loop_file,
            "detector_family": self.detector_family,
            "detector_variant": self.detector_variant,
            "detector_model_path": self.detector_model_path,
            "detector_use_augmentation": self.detector_use_augmentation,
            "tracker_family": self.tracker_family,
            "tracker_backend": self.tracker_backend,
            "tracker_config_path": self.tracker_config_path,
            "tracker_persist": self.tracker_persist,
            "tracker_with_reid": self.tracker_with_reid,
            "tracker_reid_enabled": self.tracker_reid_enabled,
            "tracker_track_buffer": self.tracker_track_buffer,
            "tracker_match_thresh": self.tracker_match_thresh,
            "tracker_new_track_thresh": self.tracker_new_track_thresh,
            "tracker_proximity_thresh": self.tracker_proximity_thresh,
            "tracker_appearance_thresh": self.tracker_appearance_thresh,
            "reid_enabled": self.reid_enabled,
            "camera_identity_enabled": self.camera_identity_enabled,
            "enabled": self.enabled,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "homography_image_to_world": self.homography_image_to_world,
            "coverage_polygon_image": self.coverage_polygon_image,
            "coverage_auto_generated": self.coverage_auto_generated,
            "coverage_confidence": self.coverage_confidence,
            "coverage_warning_text": self.coverage_warning_text,
            "calibration_rmse_px": self.calibration_rmse_px,
            "calibration_max_error_px": self.calibration_max_error_px,
            "calibration_warning_text": self.calibration_warning_text,
            "calibration_valid": self.calibration_valid,
            "entry_zone_ids": self.entry_zone_ids,
            "exit_zone_ids": self.exit_zone_ids,
            "anchor_observations": [item.to_dict() for item in self.anchor_observations],
            "overlap_camera_ids": self.overlap_camera_ids,
            "allow_auto_overlap": self.allow_auto_overlap,
            "track_timeout_s": self.track_timeout_s,
            "tracker_max_missed_frames": self.tracker_max_missed_frames,
            "bbox_publish_ttl_s": self.bbox_publish_ttl_s,
            "tracker_max_world_distance_m": self.tracker_max_world_distance_m,
            "tracker_max_image_distance_px": self.tracker_max_image_distance_px,
            "tracker_min_iou": self.tracker_min_iou,
            "tracker_anchor_weight": self.tracker_anchor_weight,
            "tracker_iou_weight": self.tracker_iou_weight,
            "tracker_confidence_weight": self.tracker_confidence_weight,
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
class TrackletObservation:
    camera_id: str
    tracker_track_id: int
    timestamp: float
    bbox_xyxy: tuple[int, int, int, int]
    ground_anchor_world: Point | None
    ground_anchor_image: Point | None
    confidence: float
    appearance_embedding: list[float] = field(default_factory=list)
    frame_index: int = 0
    media_time_s: float | None = None
    entry_edge: str | None = None
    exit_edge: str | None = None

    @property
    def local_key(self) -> str:
        return f"{self.camera_id}:{self.tracker_track_id}"


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
    last_exit_edge: str | None = None
    last_entry_edge: str | None = None
    edge_proximity_score: float = 0.0
    bbox_center_image: Point | None = None


@dataclass(slots=True)
class CameraIdentityTrack:
    camera_person_id: str
    camera_id: str
    member_tracklet_keys: list[str] = field(default_factory=list)
    active_tracklet_keys: list[str] = field(default_factory=list)
    active_tracker_track_ids: list[int] = field(default_factory=list)
    appearance_prototype: list[float] = field(default_factory=list)
    appearance_memory: list[list[float]] = field(default_factory=list)
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    active: bool = True
    current_bbox_xyxy: tuple[int, int, int, int] | None = None
    ground_anchor_world: Point | None = None
    ground_anchor_image: Point | None = None
    smoothed_ground_anchor_world: Point | None = None
    confidence: float = 0.0
    state: str = "tentative"
    world_trajectory: list[Point] = field(default_factory=list)
    positions_world: list[Point] = field(default_factory=list)
    velocity: Point = (0.0, 0.0)
    last_entry_edge: str | None = None
    last_exit_edge: str | None = None
    edge_proximity_score: float = 0.0
    bbox_center_image: Point | None = None
    inactive_since_ts: float | None = None
    support_count: int = 0
    current_local_track_id: int | None = None
    display_track_id: str | None = None
    observed_frames: int = 0
    last_match_score: float | None = None
    last_match_reason: str | None = None
    raw_tracker_track_ids: list[int] = field(default_factory=list)
    entered_overlap_ts: float | None = None
    left_overlap_ts: float | None = None
    overlap_presence_count: int = 0
    overlap_exit_side: str | None = None
    appearance_gallery: list["GalleryEntry"] = field(default_factory=list)

    @property
    def appearance_descriptor(self) -> list[float]:
        return self.appearance_prototype


@dataclass(slots=True)
class IdentityDebugRecord:
    camera_id: str
    tracker_track_id: int
    camera_person_id: str
    reason: str
    score: float
    global_candidate_id: str | None = None
    stage: str = "single_camera"
    passed_threshold: bool = False
    appearance_score: float | None = None
    world_score: float | None = None
    timing_score: float | None = None
    motion_score: float | None = None
    transition_score: float | None = None
    second_best_margin: float | None = None
    overlap_inside_left: bool | None = None
    overlap_inside_right: bool | None = None
    world_distance_m: float | None = None
    appearance_available: bool | None = None
    normalized_total_score: float | None = None
    confirmation_progress: int | None = None


@dataclass(slots=True)
class GalleryEntry:
    embedding: list[float]
    camera_id: str
    camera_person_id: str
    timestamp: float
    quality_score: float
    overlap_state: str


@dataclass(slots=True)
class OverlapTracklet:
    tracklet_id: str
    camera_id: str
    camera_person_id: str
    start_ts: float
    end_ts: float
    last_world_point: Point
    smoothed_world_points: list[Point] = field(default_factory=list)
    sample_timestamps: list[float] = field(default_factory=list)
    overlap_inside_ratio: float = 0.0
    appearance_samples: list[list[float]] = field(default_factory=list)
    mean_velocity: Point = (0.0, 0.0)
    zone_id_hint: str | None = None


@dataclass(slots=True)
class AnalyticsTrack:
    analytics_track_id: str
    source_camera_ids: list[str] = field(default_factory=list)
    active_camera_person_ids: dict[str, str] = field(default_factory=dict)
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    active: bool = True
    dedup_mode: str = "local_only"
    ground_anchor_world: Point | None = None
    smoothed_ground_anchor_world: Point | None = None
    primary_camera_id: str | None = None
    appearance_descriptor: list[float] = field(default_factory=list)
    support_count: int = 1


@dataclass(slots=True)
class MapPresence:
    presence_id: str
    world_point: Point
    source_camera_ids: list[str] = field(default_factory=list)
    source_camera_person_ids: dict[str, str] = field(default_factory=dict)
    merged_for_counting: bool = False
    booth_zone_id: str | None = None
    confidence: float = 0.0
    dedup_mode: str = "local_only"
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    confirmed: bool = True
    hold_until_ts: float = 0.0
    member_tracklets: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StableMapPresence:
    presence_id: str
    member_tracklets: list[str] = field(default_factory=list)
    world_point: Point = (0.0, 0.0)
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    confirmed: bool = False
    hold_until_ts: float = 0.0
    confidence: float = 0.0
    zone_id: str | None = None
    source_camera_ids: list[str] = field(default_factory=list)
    source_camera_person_ids: dict[str, str] = field(default_factory=dict)
    merged_for_counting: bool = False
    dedup_mode: str = "local_only"


@dataclass(slots=True)
class MapDedupDebugStats:
    overlap_tracklets_active: int = 0
    candidates_considered: int = 0
    matches_committed: int = 0
    matches_rejected_geometry: int = 0
    matches_rejected_margin: int = 0
    matches_rejected_time: int = 0
    matches_without_appearance: int = 0


@dataclass(slots=True)
class BoothVisitSession:
    visit_id: str
    zone_id: str
    analytics_track_id: str
    entered_at: float
    left_at: float | None = None
    dwell_s: float = 0.0
    source_camera_ids: list[str] = field(default_factory=list)
    dedup_mode: str = "local_only"


@dataclass(slots=True)
class AnalyticsEvent:
    event_type: str
    analytics_track_id: str
    zone_id: str | None
    timestamp: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ZoneMetrics:
    zone_id: str
    zone_name: str
    zone_kind: str
    current_occupancy: int = 0
    unique_visits: int = 0
    total_dwell_s: float = 0.0
    avg_dwell_s: float = 0.0
    median_dwell_s: float = 0.0
    peak_occupancy: int = 0
    booth_utilization_ratio: float = 0.0


@dataclass(slots=True)
class AnalyticsSnapshot:
    timestamp: float
    active_zone_counts: dict[str, int] = field(default_factory=dict)
    unique_zone_entries: dict[str, int] = field(default_factory=dict)
    dwell_times: dict[str, float] = field(default_factory=dict)
    active_analytics_tracks: dict[str, AnalyticsTrack] = field(default_factory=dict)
    active_map_presences: list[MapPresence] = field(default_factory=list)
    recent_events: list[AnalyticsEvent] = field(default_factory=list)
    avg_dwell_times: dict[str, float] = field(default_factory=dict)
    median_dwell_times: dict[str, float] = field(default_factory=dict)
    peak_occupancy_by_zone: dict[str, int] = field(default_factory=dict)
    finalized_visit_sessions_recent: list[BoothVisitSession] = field(default_factory=list)
    total_entries: int = 0
    total_current_occupancy: int = 0
    zone_metrics: dict[str, ZoneMetrics] = field(default_factory=dict)


@dataclass(slots=True)
class CameraTrackingPacket:
    camera_id: str
    timestamp: float
    wall_time_s: float = 0.0
    media_time_s: float | None = None
    frame_index: int = 0
    source_kind: str = "live"
    source_fps: float | None = None
    sync_ready: bool = True
    dropped_frame_count: int = 0
    processing_latency_s: float = 0.0
    tracklet_observations: list[TrackletObservation] = field(default_factory=list)
    local_tracks: dict[int, LocalTrack] = field(default_factory=dict)
    expired_tracks: list[LocalTrack] = field(default_factory=list)
    camera_identity_tracks: dict[str, CameraIdentityTrack] = field(default_factory=dict)
    expired_camera_identity_tracks: list[CameraIdentityTrack] = field(default_factory=list)
    identity_debug_records: list[IdentityDebugRecord] = field(default_factory=list)
    reid_backend_ready: bool = False
    frame_size: tuple[int, int] = (0, 0)
    coverage_polygon_image: list[Point] | None = None
    coverage_polygon_world_raw: list[Point] | None = None
    coverage_polygon_world: list[Point] | None = None
    coverage_auto_generated: bool = False
    coverage_confidence: float | None = None
    coverage_warning_text: str = ""
    fps: float = 0.0
    status_text: str = ""


@dataclass(slots=True)
class MultiCameraRuntimeSnapshot:
    timestamp: float
    analytics_snapshot: AnalyticsSnapshot = field(default_factory=lambda: AnalyticsSnapshot(timestamp=0.0))
    camera_packets: dict[str, CameraTrackingPacket] = field(default_factory=dict)
    overlap_graph: CameraOverlapGraph = field(default_factory=CameraOverlapGraph)
    session_sync_mode: str = "all_live_unsynced"
    session_media_time_s: float | None = None
    sync_drift_by_camera_s: dict[str, float] = field(default_factory=dict)
    dropped_frames_by_camera: dict[str, int] = field(default_factory=dict)
    missing_cameras: list[str] = field(default_factory=list)
    dedup_mode: str = "overlap_only"
    overlap_dedup_ready: bool = False
    overlap_debug_records: list[IdentityDebugRecord] = field(default_factory=list)
    active_analytics_track_count: int = 0
    deduped_overlap_track_count: int = 0
    active_map_presence_count: int = 0
    merged_map_presence_count: int = 0
    map_presence_debug_pairs_considered: int = 0
    map_presence_debug_pairs_merged: int = 0
    overlap_tracklets_active: int = 0
    map_presence_matches_committed: int = 0
    map_presence_matches_rejected_geometry: int = 0
    map_presence_matches_rejected_margin: int = 0
    map_presence_matches_rejected_time: int = 0
    map_presence_matches_without_appearance: int = 0


@dataclass(slots=True)
class SynchronizedCameraFrameSet:
    media_time_s: float
    camera_packets: dict[str, CameraTrackingPacket] = field(default_factory=dict)
    dropped_packets_by_camera: dict[str, int] = field(default_factory=dict)
    missing_cameras: list[str] = field(default_factory=list)
    drift_by_camera_s: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ProjectConfig:
    venue_map: VenueMapConfig = field(default_factory=VenueMapConfig)
    cameras: list[CameraConfig] = field(default_factory=list)
    shared_anchors: list[SharedAnchor] = field(default_factory=list)
    camera_layout: CameraLayoutConfig = field(default_factory=CameraLayoutConfig)
    overlap_dedup: OverlapDedupConfig = field(default_factory=OverlapDedupConfig)
    map_dedup: MapDedupConfig = field(default_factory=MapDedupConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    playback_sync: PlaybackSyncConfig = field(default_factory=PlaybackSyncConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectConfig":
        return cls(
            venue_map=VenueMapConfig.from_dict(data.get("venue_map", {})),
            cameras=[CameraConfig.from_dict(item) for item in data.get("cameras", [])],
            shared_anchors=[SharedAnchor.from_dict(item) for item in data.get("shared_anchors", [])],
            camera_layout=CameraLayoutConfig.from_dict(data.get("camera_layout", {})),
            overlap_dedup=OverlapDedupConfig.from_dict(data.get("overlap_dedup", {})),
            map_dedup=MapDedupConfig.from_dict(data.get("map_dedup", {})),
            analytics=AnalyticsConfig.from_dict(data.get("analytics", {})),
            playback_sync=PlaybackSyncConfig.from_dict(data.get("playback_sync", {})),
            reid=ReIDConfig.from_dict(data.get("reid", {})),
            identity=IdentityConfig.from_dict(data.get("identity", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue_map": self.venue_map.to_dict(),
            "cameras": [camera.to_dict() for camera in self.cameras],
            "shared_anchors": [anchor.to_dict() for anchor in self.shared_anchors],
            "camera_layout": self.camera_layout.to_dict(),
            "overlap_dedup": self.overlap_dedup.to_dict(),
            "map_dedup": self.map_dedup.to_dict(),
            "analytics": self.analytics.to_dict(),
            "playback_sync": self.playback_sync.to_dict(),
            "reid": self.reid.to_dict(),
            "identity": self.identity.to_dict(),
        }


def _infer_detector_family(model_path: str) -> str:
    stem = model_path.lower()
    if "rtdetr" in stem:
        return "rtdetr"
    if "yolo26" in stem:
        return "yolo26"
    if "yolo" in stem:
        return "yolo"
    return "custom"


def _infer_detector_variant(model_path: str) -> str:
    stem = model_path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    if "-" in stem:
        return stem.split("-")[-1]
    if stem and stem[-1] in {"n", "s", "m", "l", "x"}:
        return stem[-1]
    return stem
