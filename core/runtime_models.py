from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.config_models import CameraOverlapGraph, Point


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
