from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core import runtime_defaults as rd

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
class PlaybackSyncConfig:
    enabled_for_file_sources: bool = rd.DEFAULT_PLAYBACK_SYNC_ENABLED_FOR_FILE_SOURCES
    target_fps: float = rd.DEFAULT_PLAYBACK_SYNC_TARGET_FPS
    sync_tolerance_s: float = rd.DEFAULT_PLAYBACK_SYNC_TOLERANCE_S
    late_frame_drop_threshold_s: float = rd.DEFAULT_PLAYBACK_LATE_FRAME_DROP_THRESHOLD_S
    stale_packet_threshold_s: float = rd.DEFAULT_PLAYBACK_STALE_PACKET_THRESHOLD_S
    camera_missing_timeout_s: float = rd.DEFAULT_PLAYBACK_CAMERA_MISSING_TIMEOUT_S
    max_buffered_packets_per_camera: int = rd.DEFAULT_PLAYBACK_MAX_BUFFERED_PACKETS_PER_CAMERA

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlaybackSyncConfig":
        return cls(
            enabled_for_file_sources=bool(
                data.get("enabled_for_file_sources", rd.DEFAULT_PLAYBACK_SYNC_ENABLED_FOR_FILE_SOURCES)
            ),
            target_fps=float(data.get("target_fps", rd.DEFAULT_PLAYBACK_SYNC_TARGET_FPS)),
            sync_tolerance_s=float(data.get("sync_tolerance_s", rd.DEFAULT_PLAYBACK_SYNC_TOLERANCE_S)),
            late_frame_drop_threshold_s=float(
                data.get("late_frame_drop_threshold_s", rd.DEFAULT_PLAYBACK_LATE_FRAME_DROP_THRESHOLD_S)
            ),
            stale_packet_threshold_s=float(
                data.get("stale_packet_threshold_s", rd.DEFAULT_PLAYBACK_STALE_PACKET_THRESHOLD_S)
            ),
            camera_missing_timeout_s=float(
                data.get("camera_missing_timeout_s", rd.DEFAULT_PLAYBACK_CAMERA_MISSING_TIMEOUT_S)
            ),
            max_buffered_packets_per_camera=int(
                data.get("max_buffered_packets_per_camera", rd.DEFAULT_PLAYBACK_MAX_BUFFERED_PACKETS_PER_CAMERA)
            ),
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
class DistributedRuntimeConfig:
    enabled: bool = rd.DEFAULT_DISTRIBUTED_ENABLED
    server_bind_host: str = rd.DEFAULT_DISTRIBUTED_BIND_HOST
    server_port: int = rd.DEFAULT_DISTRIBUTED_PORT
    preview_fps: float = rd.DEFAULT_DISTRIBUTED_PREVIEW_FPS
    preview_jpeg_quality: int = rd.DEFAULT_DISTRIBUTED_PREVIEW_JPEG_QUALITY
    worker_heartbeat_interval_s: float = rd.DEFAULT_DISTRIBUTED_HEARTBEAT_INTERVAL_S
    worker_timeout_s: float = rd.DEFAULT_DISTRIBUTED_WORKER_TIMEOUT_S
    protocol_version: str = rd.DEFAULT_DISTRIBUTED_PROTOCOL_VERSION

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DistributedRuntimeConfig":
        return cls(
            enabled=bool(data.get("enabled", rd.DEFAULT_DISTRIBUTED_ENABLED)),
            server_bind_host=str(data.get("server_bind_host", rd.DEFAULT_DISTRIBUTED_BIND_HOST)),
            server_port=int(data.get("server_port", rd.DEFAULT_DISTRIBUTED_PORT)),
            preview_fps=float(data.get("preview_fps", rd.DEFAULT_DISTRIBUTED_PREVIEW_FPS)),
            preview_jpeg_quality=int(
                data.get("preview_jpeg_quality", rd.DEFAULT_DISTRIBUTED_PREVIEW_JPEG_QUALITY)
            ),
            worker_heartbeat_interval_s=float(
                data.get("worker_heartbeat_interval_s", rd.DEFAULT_DISTRIBUTED_HEARTBEAT_INTERVAL_S)
            ),
            worker_timeout_s=float(data.get("worker_timeout_s", rd.DEFAULT_DISTRIBUTED_WORKER_TIMEOUT_S)),
            protocol_version=str(data.get("protocol_version", rd.DEFAULT_DISTRIBUTED_PROTOCOL_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "server_bind_host": self.server_bind_host,
            "server_port": self.server_port,
            "preview_fps": self.preview_fps,
            "preview_jpeg_quality": self.preview_jpeg_quality,
            "worker_heartbeat_interval_s": self.worker_heartbeat_interval_s,
            "worker_timeout_s": self.worker_timeout_s,
            "protocol_version": self.protocol_version,
        }


@dataclass(slots=True)
class ReIDConfig:
    enabled: bool = rd.DEFAULT_REID_ENABLED
    backend: str = rd.DEFAULT_REID_BACKEND
    model_name: str = rd.DEFAULT_REID_MODEL_NAME
    weights_path: str = rd.DEFAULT_REID_WEIGHTS_PATH
    weights_url: str = rd.DEFAULT_REID_WEIGHTS_URL
    download_if_missing: bool = rd.DEFAULT_REID_DOWNLOAD_IF_MISSING
    min_bbox_height_px: int = rd.DEFAULT_REID_MIN_BBOX_HEIGHT_PX
    min_confidence: float = rd.DEFAULT_REID_MIN_CONFIDENCE
    overlap_reid_min_bbox_height_px: int = rd.DEFAULT_REID_OVERLAP_MIN_BBOX_HEIGHT_PX
    overlap_reid_min_confidence: float = rd.DEFAULT_REID_OVERLAP_MIN_CONFIDENCE
    embedding_similarity_threshold: float = rd.DEFAULT_REID_EMBEDDING_SIMILARITY_THRESHOLD
    single_camera_restitch_threshold: float = rd.DEFAULT_REID_SINGLE_CAMERA_RESTITCH_THRESHOLD
    cross_camera_match_threshold: float = rd.DEFAULT_REID_CROSS_CAMERA_MATCH_THRESHOLD
    embedding_momentum: float = rd.DEFAULT_REID_EMBEDDING_MOMENTUM
    max_embedding_memory: int = rd.DEFAULT_REID_MAX_EMBEDDING_MEMORY
    gallery_size_per_camera: int = rd.DEFAULT_REID_GALLERY_SIZE_PER_CAMERA
    gallery_size_global: int = rd.DEFAULT_REID_GALLERY_SIZE_GLOBAL
    match_topk: int = rd.DEFAULT_REID_MATCH_TOPK
    match_reduce: str = rd.DEFAULT_REID_MATCH_REDUCE
    re_rank_enabled: bool = rd.DEFAULT_REID_RERANK_ENABLED
    input_width: int = rd.DEFAULT_REID_INPUT_WIDTH
    input_height: int = rd.DEFAULT_REID_INPUT_HEIGHT
    input_mean: list[float] = field(default_factory=lambda: list(rd.DEFAULT_REID_INPUT_MEAN))
    input_std: list[float] = field(default_factory=lambda: list(rd.DEFAULT_REID_INPUT_STD))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReIDConfig":
        backend = str(data.get("backend", rd.DEFAULT_REID_BACKEND))
        if backend == "osnet":
            backend = rd.DEFAULT_REID_BACKEND
        return cls(
            enabled=bool(data.get("enabled", rd.DEFAULT_REID_ENABLED)),
            backend=backend,
            model_name=str(data.get("model_name", rd.DEFAULT_REID_MODEL_NAME)),
            weights_path=str(data.get("weights_path", rd.DEFAULT_REID_WEIGHTS_PATH)),
            weights_url=str(data.get("weights_url", rd.DEFAULT_REID_WEIGHTS_URL)),
            download_if_missing=bool(data.get("download_if_missing", rd.DEFAULT_REID_DOWNLOAD_IF_MISSING)),
            min_bbox_height_px=int(data.get("min_bbox_height_px", rd.DEFAULT_REID_MIN_BBOX_HEIGHT_PX)),
            min_confidence=float(data.get("min_confidence", rd.DEFAULT_REID_MIN_CONFIDENCE)),
            overlap_reid_min_bbox_height_px=int(
                data.get("overlap_reid_min_bbox_height_px", rd.DEFAULT_REID_OVERLAP_MIN_BBOX_HEIGHT_PX)
            ),
            overlap_reid_min_confidence=float(
                data.get("overlap_reid_min_confidence", rd.DEFAULT_REID_OVERLAP_MIN_CONFIDENCE)
            ),
            embedding_similarity_threshold=float(
                data.get("embedding_similarity_threshold", rd.DEFAULT_REID_EMBEDDING_SIMILARITY_THRESHOLD)
            ),
            single_camera_restitch_threshold=float(
                data.get(
                    "single_camera_restitch_threshold",
                    rd.DEFAULT_REID_SINGLE_CAMERA_RESTITCH_THRESHOLD,
                )
            ),
            cross_camera_match_threshold=float(
                data.get("cross_camera_match_threshold", rd.DEFAULT_REID_CROSS_CAMERA_MATCH_THRESHOLD)
            ),
            embedding_momentum=float(data.get("embedding_momentum", rd.DEFAULT_REID_EMBEDDING_MOMENTUM)),
            max_embedding_memory=int(data.get("max_embedding_memory", rd.DEFAULT_REID_MAX_EMBEDDING_MEMORY)),
            gallery_size_per_camera=int(
                data.get("gallery_size_per_camera", rd.DEFAULT_REID_GALLERY_SIZE_PER_CAMERA)
            ),
            gallery_size_global=int(data.get("gallery_size_global", rd.DEFAULT_REID_GALLERY_SIZE_GLOBAL)),
            match_topk=int(data.get("match_topk", rd.DEFAULT_REID_MATCH_TOPK)),
            match_reduce=str(data.get("match_reduce", rd.DEFAULT_REID_MATCH_REDUCE)),
            re_rank_enabled=bool(data.get("re_rank_enabled", rd.DEFAULT_REID_RERANK_ENABLED)),
            input_width=int(data.get("input_width", rd.DEFAULT_REID_INPUT_WIDTH)),
            input_height=int(data.get("input_height", rd.DEFAULT_REID_INPUT_HEIGHT)),
            input_mean=[float(value) for value in data.get("input_mean", rd.DEFAULT_REID_INPUT_MEAN)],
            input_std=[float(value) for value in data.get("input_std", rd.DEFAULT_REID_INPUT_STD)],
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
    single_camera_restitch_enabled: bool = rd.DEFAULT_IDENTITY_RESTITCH_ENABLED
    single_camera_max_gap_s: float = rd.DEFAULT_IDENTITY_MAX_GAP_S
    single_camera_base_distance_m: float = rd.DEFAULT_IDENTITY_BASE_DISTANCE_M
    single_camera_speed_m_per_s: float = rd.DEFAULT_IDENTITY_SPEED_M_PER_S
    single_camera_restitch_threshold: float = rd.DEFAULT_IDENTITY_RESTITCH_THRESHOLD
    confirmation_frames: int = rd.DEFAULT_IDENTITY_CONFIRMATION_FRAMES
    lost_track_retention_s: float = rd.DEFAULT_IDENTITY_LOST_TRACK_RETENTION_S

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdentityConfig":
        return cls(
            single_camera_restitch_enabled=bool(
                data.get("single_camera_restitch_enabled", rd.DEFAULT_IDENTITY_RESTITCH_ENABLED)
            ),
            single_camera_max_gap_s=float(data.get("single_camera_max_gap_s", rd.DEFAULT_IDENTITY_MAX_GAP_S)),
            single_camera_base_distance_m=float(
                data.get("single_camera_base_distance_m", rd.DEFAULT_IDENTITY_BASE_DISTANCE_M)
            ),
            single_camera_speed_m_per_s=float(
                data.get("single_camera_speed_m_per_s", rd.DEFAULT_IDENTITY_SPEED_M_PER_S)
            ),
            single_camera_restitch_threshold=float(
                data.get("single_camera_restitch_threshold", rd.DEFAULT_IDENTITY_RESTITCH_THRESHOLD)
            ),
            confirmation_frames=int(data.get("confirmation_frames", rd.DEFAULT_IDENTITY_CONFIRMATION_FRAMES)),
            lost_track_retention_s=float(
                data.get("lost_track_retention_s", rd.DEFAULT_IDENTITY_LOST_TRACK_RETENTION_S)
            ),
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
    zone_entry_min_duration_s: float = rd.DEFAULT_ZONE_ENTRY_MIN_DURATION_S
    zone_exit_grace_s: float = rd.DEFAULT_ZONE_EXIT_GRACE_S

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalyticsConfig":
        return cls(
            zone_entry_min_duration_s=float(
                data.get("zone_entry_min_duration_s", rd.DEFAULT_ZONE_ENTRY_MIN_DURATION_S)
            ),
            zone_exit_grace_s=float(data.get("zone_exit_grace_s", rd.DEFAULT_ZONE_EXIT_GRACE_S)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_entry_min_duration_s": self.zone_entry_min_duration_s,
            "zone_exit_grace_s": self.zone_exit_grace_s,
        }


@dataclass(slots=True)
class OverlapDedupConfig:
    enabled: bool = rd.DEFAULT_OVERLAP_DEDUP_ENABLED
    overlap_area_min_m2: float = rd.DEFAULT_OVERLAP_AREA_MIN_M2
    boundary_gap_m: float = rd.DEFAULT_OVERLAP_BOUNDARY_GAP_M
    max_distance_m: float = rd.DEFAULT_OVERLAP_MAX_DISTANCE_M
    max_time_gap_s: float = rd.DEFAULT_OVERLAP_MAX_TIME_GAP_S
    appearance_weight: float = rd.DEFAULT_OVERLAP_APPEARANCE_WEIGHT
    world_weight: float = rd.DEFAULT_OVERLAP_WORLD_WEIGHT
    timing_weight: float = rd.DEFAULT_OVERLAP_TIMING_WEIGHT
    geometry_weight: float = rd.DEFAULT_OVERLAP_GEOMETRY_WEIGHT
    similarity_threshold: float = rd.DEFAULT_OVERLAP_SIMILARITY_THRESHOLD
    margin_min: float = rd.DEFAULT_OVERLAP_MARGIN_MIN
    confirmation_frames: int = rd.DEFAULT_OVERLAP_CONFIRMATION_FRAMES

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OverlapDedupConfig":
        return cls(
            enabled=bool(data.get("enabled", rd.DEFAULT_OVERLAP_DEDUP_ENABLED)),
            overlap_area_min_m2=float(data.get("overlap_area_min_m2", rd.DEFAULT_OVERLAP_AREA_MIN_M2)),
            boundary_gap_m=float(data.get("boundary_gap_m", rd.DEFAULT_OVERLAP_BOUNDARY_GAP_M)),
            max_distance_m=float(data.get("max_distance_m", rd.DEFAULT_OVERLAP_MAX_DISTANCE_M)),
            max_time_gap_s=float(data.get("max_time_gap_s", rd.DEFAULT_OVERLAP_MAX_TIME_GAP_S)),
            appearance_weight=float(data.get("appearance_weight", rd.DEFAULT_OVERLAP_APPEARANCE_WEIGHT)),
            world_weight=float(data.get("world_weight", rd.DEFAULT_OVERLAP_WORLD_WEIGHT)),
            timing_weight=float(data.get("timing_weight", rd.DEFAULT_OVERLAP_TIMING_WEIGHT)),
            geometry_weight=float(data.get("geometry_weight", rd.DEFAULT_OVERLAP_GEOMETRY_WEIGHT)),
            similarity_threshold=float(
                data.get("similarity_threshold", rd.DEFAULT_OVERLAP_SIMILARITY_THRESHOLD)
            ),
            margin_min=float(data.get("margin_min", rd.DEFAULT_OVERLAP_MARGIN_MIN)),
            confirmation_frames=int(data.get("confirmation_frames", rd.DEFAULT_OVERLAP_CONFIRMATION_FRAMES)),
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
    enabled: bool = rd.DEFAULT_MAP_DEDUP_ENABLED
    overlap_presence_boundary_buffer_m: float = rd.DEFAULT_MAP_OVERLAP_BOUNDARY_BUFFER_M
    max_distance_m: float = rd.DEFAULT_MAP_MAX_DISTANCE_M
    max_time_gap_s: float = rd.DEFAULT_MAP_MAX_TIME_GAP_S
    similarity_threshold: float = rd.DEFAULT_MAP_SIMILARITY_THRESHOLD
    margin_min: float = rd.DEFAULT_MAP_MARGIN_MIN
    confirmation_frames: int = rd.DEFAULT_MAP_CONFIRMATION_FRAMES
    world_alignment_weight: float = rd.DEFAULT_MAP_WORLD_ALIGNMENT_WEIGHT
    overlap_membership_weight: float = rd.DEFAULT_MAP_OVERLAP_MEMBERSHIP_WEIGHT
    timing_weight: float = rd.DEFAULT_MAP_TIMING_WEIGHT
    appearance_weight: float = rd.DEFAULT_MAP_APPEARANCE_WEIGHT
    motion_weight: float = rd.DEFAULT_MAP_MOTION_WEIGHT
    tracklet_window_s: float = rd.DEFAULT_MAP_TRACKLET_WINDOW_S
    tracklet_min_points: int = rd.DEFAULT_MAP_TRACKLET_MIN_POINTS
    tracklet_keepalive_s: float = rd.DEFAULT_MAP_TRACKLET_KEEPALIVE_S
    overlap_batch_window_s: float = rd.DEFAULT_MAP_BATCH_WINDOW_S
    overlap_batch_step_s: float = rd.DEFAULT_MAP_BATCH_STEP_S
    presence_hold_s: float = rd.DEFAULT_MAP_PRESENCE_HOLD_S
    presence_reassign_window_s: float = rd.DEFAULT_MAP_PRESENCE_REASSIGN_WINDOW_S
    presence_publish_ttl_s: float = rd.DEFAULT_MAP_PRESENCE_PUBLISH_TTL_S
    max_exact_assignment_size: int = rd.DEFAULT_MAP_MAX_EXACT_ASSIGNMENT_SIZE

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MapDedupConfig":
        return cls(
            enabled=bool(data.get("enabled", rd.DEFAULT_MAP_DEDUP_ENABLED)),
            overlap_presence_boundary_buffer_m=float(
                data.get("overlap_presence_boundary_buffer_m", rd.DEFAULT_MAP_OVERLAP_BOUNDARY_BUFFER_M)
            ),
            max_distance_m=float(data.get("max_distance_m", rd.DEFAULT_MAP_MAX_DISTANCE_M)),
            max_time_gap_s=float(data.get("max_time_gap_s", rd.DEFAULT_MAP_MAX_TIME_GAP_S)),
            similarity_threshold=float(data.get("similarity_threshold", rd.DEFAULT_MAP_SIMILARITY_THRESHOLD)),
            margin_min=float(data.get("margin_min", rd.DEFAULT_MAP_MARGIN_MIN)),
            confirmation_frames=int(data.get("confirmation_frames", rd.DEFAULT_MAP_CONFIRMATION_FRAMES)),
            world_alignment_weight=float(
                data.get("world_alignment_weight", rd.DEFAULT_MAP_WORLD_ALIGNMENT_WEIGHT)
            ),
            overlap_membership_weight=float(
                data.get("overlap_membership_weight", rd.DEFAULT_MAP_OVERLAP_MEMBERSHIP_WEIGHT)
            ),
            timing_weight=float(data.get("timing_weight", rd.DEFAULT_MAP_TIMING_WEIGHT)),
            appearance_weight=float(data.get("appearance_weight", rd.DEFAULT_MAP_APPEARANCE_WEIGHT)),
            motion_weight=float(data.get("motion_weight", rd.DEFAULT_MAP_MOTION_WEIGHT)),
            tracklet_window_s=float(data.get("tracklet_window_s", rd.DEFAULT_MAP_TRACKLET_WINDOW_S)),
            tracklet_min_points=int(data.get("tracklet_min_points", rd.DEFAULT_MAP_TRACKLET_MIN_POINTS)),
            tracklet_keepalive_s=float(data.get("tracklet_keepalive_s", rd.DEFAULT_MAP_TRACKLET_KEEPALIVE_S)),
            overlap_batch_window_s=float(
                data.get("overlap_batch_window_s", rd.DEFAULT_MAP_BATCH_WINDOW_S)
            ),
            overlap_batch_step_s=float(data.get("overlap_batch_step_s", rd.DEFAULT_MAP_BATCH_STEP_S)),
            presence_hold_s=float(data.get("presence_hold_s", rd.DEFAULT_MAP_PRESENCE_HOLD_S)),
            presence_reassign_window_s=float(
                data.get("presence_reassign_window_s", rd.DEFAULT_MAP_PRESENCE_REASSIGN_WINDOW_S)
            ),
            presence_publish_ttl_s=float(
                data.get("presence_publish_ttl_s", rd.DEFAULT_MAP_PRESENCE_PUBLISH_TTL_S)
            ),
            max_exact_assignment_size=int(
                data.get("max_exact_assignment_size", rd.DEFAULT_MAP_MAX_EXACT_ASSIGNMENT_SIZE)
            ),
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
            polygon_world=[(float(point[0]), float(point[1])) for point in data.get("polygon_world", [])],
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
    source_type: str = "udp"
    source_value: str = "udp://0.0.0.0:5000"
    loop_file: bool = False
    runtime_mode: str = "local"
    remote_worker_id: str = ""
    enabled: bool = True
    homography_image_to_world: list[list[float]] | None = None
    coverage_polygon_image: list[Point] | None = None
    calibration_rmse_px: float | None = None
    calibration_max_error_px: float | None = None
    calibration_valid: bool = False
    anchor_observations: list[CameraAnchorObservation] = field(default_factory=list)
    overlap_camera_ids: list[str] = field(default_factory=list)

    detector_model_path: str = rd.DEFAULT_DETECTOR_MODEL_PATH
    detector_use_augmentation: bool = rd.DEFAULT_DETECTOR_AUGMENTATION
    tracker_backend: str = rd.DEFAULT_TRACKER_BACKEND
    tracker_reid_enabled: bool = rd.DEFAULT_TRACKER_REID_ENABLED
    tracker_track_buffer: int = rd.DEFAULT_TRACKER_TRACK_BUFFER
    tracker_match_thresh: float = rd.DEFAULT_TRACKER_MATCH_THRESH
    tracker_new_track_thresh: float = rd.DEFAULT_TRACKER_NEW_TRACK_THRESH
    tracker_proximity_thresh: float = rd.DEFAULT_TRACKER_PROXIMITY_THRESH
    tracker_appearance_thresh: float = rd.DEFAULT_TRACKER_APPEARANCE_THRESH
    reid_enabled: bool = rd.DEFAULT_REID_ENABLED
    frame_width: int = 0
    frame_height: int = 0
    coverage_auto_generated: bool = False
    coverage_confidence: float | None = None
    coverage_warning_text: str = ""
    coverage_polygon_world: list[Point] | None = None
    coverage_polygon_world_raw: list[Point] | None = None
    calibration_warning_text: str = ""
    track_timeout_s: float = rd.DEFAULT_TRACK_TIMEOUT_S
    tracker_max_missed_frames: int = rd.DEFAULT_TRACKER_MAX_MISSED_FRAMES
    bbox_publish_ttl_s: float = rd.DEFAULT_BBOX_PUBLISH_TTL_S
    tracker_max_world_distance_m: float = rd.DEFAULT_TRACKER_MAX_WORLD_DISTANCE_M
    tracker_max_image_distance_px: float = rd.DEFAULT_TRACKER_MAX_IMAGE_DISTANCE_PX
    tracker_min_iou: float = rd.DEFAULT_TRACKER_MIN_IOU
    tracker_anchor_weight: float = rd.DEFAULT_TRACKER_ANCHOR_WEIGHT
    tracker_iou_weight: float = rd.DEFAULT_TRACKER_IOU_WEIGHT
    tracker_confidence_weight: float = rd.DEFAULT_TRACKER_CONFIDENCE_WEIGHT

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraConfig":
        homography = data.get("homography_image_to_world")
        coverage_polygon_image = [
            (float(point[0]), float(point[1]))
            for point in (data.get("coverage_polygon_image") or [])
        ] or None
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
            source_type=str(data.get("source_type", "udp")),
            source_value=str(data.get("source_value", data.get("udp_url", "udp://0.0.0.0:5000"))),
            loop_file=bool(data.get("loop_file", False)),
            runtime_mode=str(data.get("runtime_mode", "local")),
            remote_worker_id=str(data.get("remote_worker_id", "")),
            enabled=bool(data.get("enabled", True)),
            homography_image_to_world=homography if homography else None,
            coverage_polygon_image=coverage_polygon_image,
            calibration_rmse_px=(
                float(data["calibration_rmse_px"]) if data.get("calibration_rmse_px") is not None else None
            ),
            calibration_max_error_px=(
                float(data["calibration_max_error_px"])
                if data.get("calibration_max_error_px") is not None
                else None
            ),
            calibration_valid=calibration_valid,
            anchor_observations=[
                CameraAnchorObservation.from_dict(item) for item in data.get("anchor_observations", [])
            ],
            overlap_camera_ids=[str(value) for value in data.get("overlap_camera_ids", [])],
            detector_model_path=str(data.get("detector_model_path", rd.DEFAULT_DETECTOR_MODEL_PATH)),
            detector_use_augmentation=bool(
                data.get("detector_use_augmentation", rd.DEFAULT_DETECTOR_AUGMENTATION)
            ),
            tracker_backend=str(data.get("tracker_backend", rd.DEFAULT_TRACKER_BACKEND)),
            tracker_reid_enabled=bool(data.get("tracker_reid_enabled", rd.DEFAULT_TRACKER_REID_ENABLED)),
            tracker_track_buffer=int(data.get("tracker_track_buffer", rd.DEFAULT_TRACKER_TRACK_BUFFER)),
            tracker_match_thresh=float(data.get("tracker_match_thresh", rd.DEFAULT_TRACKER_MATCH_THRESH)),
            tracker_new_track_thresh=float(
                data.get("tracker_new_track_thresh", rd.DEFAULT_TRACKER_NEW_TRACK_THRESH)
            ),
            tracker_proximity_thresh=float(
                data.get("tracker_proximity_thresh", rd.DEFAULT_TRACKER_PROXIMITY_THRESH)
            ),
            tracker_appearance_thresh=float(
                data.get("tracker_appearance_thresh", rd.DEFAULT_TRACKER_APPEARANCE_THRESH)
            ),
            reid_enabled=bool(data.get("reid_enabled", rd.DEFAULT_REID_ENABLED)),
            frame_width=int(data.get("frame_width", 0)),
            frame_height=int(data.get("frame_height", 0)),
            coverage_auto_generated=bool(data.get("coverage_auto_generated", False)),
            coverage_confidence=(
                float(data["coverage_confidence"]) if data.get("coverage_confidence") is not None else None
            ),
            coverage_warning_text=str(data.get("coverage_warning_text", "")),
            coverage_polygon_world=[
                (float(point[0]), float(point[1]))
                for point in (data.get("coverage_polygon_world") or [])
            ] or None,
            coverage_polygon_world_raw=[
                (float(point[0]), float(point[1]))
                for point in (data.get("coverage_polygon_world_raw") or [])
            ] or None,
            calibration_warning_text=str(data.get("calibration_warning_text", "")),
            track_timeout_s=float(data.get("track_timeout_s", rd.DEFAULT_TRACK_TIMEOUT_S)),
            tracker_max_missed_frames=int(
                data.get("tracker_max_missed_frames", rd.DEFAULT_TRACKER_MAX_MISSED_FRAMES)
            ),
            bbox_publish_ttl_s=float(data.get("bbox_publish_ttl_s", rd.DEFAULT_BBOX_PUBLISH_TTL_S)),
            tracker_max_world_distance_m=float(
                data.get("tracker_max_world_distance_m", rd.DEFAULT_TRACKER_MAX_WORLD_DISTANCE_M)
            ),
            tracker_max_image_distance_px=float(
                data.get("tracker_max_image_distance_px", rd.DEFAULT_TRACKER_MAX_IMAGE_DISTANCE_PX)
            ),
            tracker_min_iou=float(data.get("tracker_min_iou", rd.DEFAULT_TRACKER_MIN_IOU)),
            tracker_anchor_weight=float(
                data.get("tracker_anchor_weight", rd.DEFAULT_TRACKER_ANCHOR_WEIGHT)
            ),
            tracker_iou_weight=float(data.get("tracker_iou_weight", rd.DEFAULT_TRACKER_IOU_WEIGHT)),
            tracker_confidence_weight=float(
                data.get("tracker_confidence_weight", rd.DEFAULT_TRACKER_CONFIDENCE_WEIGHT)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_persisted_dict()
        payload.update(
            {
                "detector_model_path": self.detector_model_path,
                "detector_use_augmentation": self.detector_use_augmentation,
                "tracker_backend": self.tracker_backend,
                "tracker_reid_enabled": self.tracker_reid_enabled,
                "tracker_track_buffer": self.tracker_track_buffer,
                "tracker_match_thresh": self.tracker_match_thresh,
                "tracker_new_track_thresh": self.tracker_new_track_thresh,
                "tracker_proximity_thresh": self.tracker_proximity_thresh,
                "tracker_appearance_thresh": self.tracker_appearance_thresh,
                "reid_enabled": self.reid_enabled,
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
                "coverage_auto_generated": self.coverage_auto_generated,
                "coverage_confidence": self.coverage_confidence,
                "coverage_warning_text": self.coverage_warning_text,
                "coverage_polygon_world": self.coverage_polygon_world,
                "coverage_polygon_world_raw": self.coverage_polygon_world_raw,
                "calibration_warning_text": self.calibration_warning_text,
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
        )
        return payload

    def to_persisted_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "camera_id": self.camera_id,
            "name": self.name,
            "display_order": self.display_order,
            "source_type": self.source_type,
            "source_value": self.source_value,
            "loop_file": self.loop_file,
            "runtime_mode": self.runtime_mode,
            "enabled": self.enabled,
        }
        if self.remote_worker_id:
            payload["remote_worker_id"] = self.remote_worker_id
        if self.homography_image_to_world is not None:
            payload["homography_image_to_world"] = self.homography_image_to_world
        if self.coverage_polygon_image:
            payload["coverage_polygon_image"] = self.coverage_polygon_image
        if self.calibration_rmse_px is not None:
            payload["calibration_rmse_px"] = self.calibration_rmse_px
        if self.calibration_max_error_px is not None:
            payload["calibration_max_error_px"] = self.calibration_max_error_px
        if self.calibration_valid:
            payload["calibration_valid"] = True
        if self.anchor_observations:
            payload["anchor_observations"] = [item.to_dict() for item in self.anchor_observations]
        if self.overlap_camera_ids:
            payload["overlap_camera_ids"] = self.overlap_camera_ids
        return payload


@dataclass(slots=True)
class ProjectConfig:
    venue_map: VenueMapConfig = field(default_factory=VenueMapConfig)
    cameras: list[CameraConfig] = field(default_factory=list)
    shared_anchors: list[SharedAnchor] = field(default_factory=list)
    overlap_dedup: OverlapDedupConfig = field(default_factory=OverlapDedupConfig)
    map_dedup: MapDedupConfig = field(default_factory=MapDedupConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    playback_sync: PlaybackSyncConfig = field(default_factory=PlaybackSyncConfig)
    distributed_runtime: DistributedRuntimeConfig = field(default_factory=DistributedRuntimeConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectConfig":
        return cls(
            venue_map=VenueMapConfig.from_dict(data.get("venue_map", {})),
            cameras=[CameraConfig.from_dict(item) for item in data.get("cameras", [])],
            shared_anchors=[SharedAnchor.from_dict(item) for item in data.get("shared_anchors", [])],
            overlap_dedup=OverlapDedupConfig.from_dict(data.get("overlap_dedup", {})),
            map_dedup=MapDedupConfig.from_dict(data.get("map_dedup", {})),
            analytics=AnalyticsConfig.from_dict(data.get("analytics", {})),
            playback_sync=PlaybackSyncConfig.from_dict(data.get("playback_sync", {})),
            distributed_runtime=DistributedRuntimeConfig.from_dict(data.get("distributed_runtime", {})),
            reid=ReIDConfig.from_dict(data.get("reid", {})),
            identity=IdentityConfig.from_dict(data.get("identity", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue_map": self.venue_map.to_dict(),
            "cameras": [camera.to_dict() for camera in self.cameras],
            "shared_anchors": [anchor.to_dict() for anchor in self.shared_anchors],
            "overlap_dedup": self.overlap_dedup.to_dict(),
            "map_dedup": self.map_dedup.to_dict(),
            "analytics": self.analytics.to_dict(),
            "playback_sync": self.playback_sync.to_dict(),
            "distributed_runtime": self.distributed_runtime.to_dict(),
            "reid": self.reid.to_dict(),
            "identity": self.identity.to_dict(),
        }

    def to_persisted_dict(self) -> dict[str, Any]:
        return {
            "shared_anchors": [anchor.to_dict() for anchor in self.shared_anchors],
            "analytics": self.analytics.to_dict(),
        }
