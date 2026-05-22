from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from core.models import (
    CameraConfig,
    CameraIdentityTrack,
    CameraTrackingPacket,
    IdentityDebugRecord,
    LocalTrack,
    TrackletObservation,
)


def camera_config_sha256(camera_config: CameraConfig) -> str:
    payload = json.dumps(
        camera_config.to_persisted_dict(),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def camera_tracking_packet_to_network_dict(packet: CameraTrackingPacket) -> dict[str, Any]:
    return {
        "camera_id": packet.camera_id,
        "timestamp": packet.timestamp,
        "wall_time_s": packet.wall_time_s,
        "media_time_s": packet.media_time_s,
        "frame_index": packet.frame_index,
        "source_kind": packet.source_kind,
        "source_fps": packet.source_fps,
        "sync_ready": packet.sync_ready,
        "dropped_frame_count": packet.dropped_frame_count,
        "processing_latency_s": packet.processing_latency_s,
        "tracklet_observations": [asdict(item) for item in packet.tracklet_observations],
        "local_tracks": {track_id: asdict(track) for track_id, track in packet.local_tracks.items()},
        "expired_tracks": [asdict(track) for track in packet.expired_tracks],
        "camera_identity_tracks": {
            track_id: asdict(track) for track_id, track in packet.camera_identity_tracks.items()
        },
        "expired_camera_identity_tracks": [asdict(track) for track in packet.expired_camera_identity_tracks],
        "identity_debug_records": [asdict(item) for item in packet.identity_debug_records],
        "reid_backend_ready": packet.reid_backend_ready,
        "frame_size": list(packet.frame_size),
        "coverage_polygon_image": _point_list(packet.coverage_polygon_image),
        "coverage_polygon_world_raw": _point_list(packet.coverage_polygon_world_raw),
        "coverage_polygon_world": _point_list(packet.coverage_polygon_world),
        "coverage_auto_generated": packet.coverage_auto_generated,
        "coverage_confidence": packet.coverage_confidence,
        "coverage_warning_text": packet.coverage_warning_text,
        "fps": packet.fps,
        "status_text": packet.status_text,
    }


def camera_tracking_packet_from_network_dict(data: dict[str, Any]) -> CameraTrackingPacket:
    return CameraTrackingPacket(
        camera_id=str(data["camera_id"]),
        timestamp=float(data["timestamp"]),
        wall_time_s=float(data.get("wall_time_s", 0.0)),
        media_time_s=_optional_float(data.get("media_time_s")),
        frame_index=int(data.get("frame_index", 0)),
        source_kind=str(data.get("source_kind", "live")),
        source_fps=_optional_float(data.get("source_fps")),
        sync_ready=bool(data.get("sync_ready", True)),
        dropped_frame_count=int(data.get("dropped_frame_count", 0)),
        processing_latency_s=float(data.get("processing_latency_s", 0.0)),
        tracklet_observations=[
            _tracklet_observation_from_dict(item) for item in data.get("tracklet_observations", [])
        ],
        local_tracks={
            int(track_id): _local_track_from_dict(track_data)
            for track_id, track_data in data.get("local_tracks", {}).items()
        },
        expired_tracks=[_local_track_from_dict(item) for item in data.get("expired_tracks", [])],
        camera_identity_tracks={
            str(track_id): _camera_identity_track_from_dict(track_data)
            for track_id, track_data in data.get("camera_identity_tracks", {}).items()
        },
        expired_camera_identity_tracks=[
            _camera_identity_track_from_dict(item)
            for item in data.get("expired_camera_identity_tracks", [])
        ],
        identity_debug_records=[
            _identity_debug_record_from_dict(item) for item in data.get("identity_debug_records", [])
        ],
        reid_backend_ready=bool(data.get("reid_backend_ready", False)),
        frame_size=_frame_size_from_value(data.get("frame_size")),
        coverage_polygon_image=_points_from_value(data.get("coverage_polygon_image")),
        coverage_polygon_world_raw=_points_from_value(data.get("coverage_polygon_world_raw")),
        coverage_polygon_world=_points_from_value(data.get("coverage_polygon_world")),
        coverage_auto_generated=bool(data.get("coverage_auto_generated", False)),
        coverage_confidence=_optional_float(data.get("coverage_confidence")),
        coverage_warning_text=str(data.get("coverage_warning_text", "")),
        fps=float(data.get("fps", 0.0)),
        status_text=str(data.get("status_text", "")),
    )


def preview_frame_to_network_dict(
    *,
    camera_id: str,
    frame_index: int,
    timestamp: float,
    width: int,
    height: int,
    jpeg_bytes: bytes,
) -> dict[str, Any]:
    return {
        "camera_id": camera_id,
        "frame_index": frame_index,
        "timestamp": timestamp,
        "width": width,
        "height": height,
        "jpeg_bytes": jpeg_bytes,
    }


def preview_frame_from_network_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "camera_id": str(data["camera_id"]),
        "frame_index": int(data.get("frame_index", 0)),
        "timestamp": float(data.get("timestamp", 0.0)),
        "width": int(data.get("width", 0)),
        "height": int(data.get("height", 0)),
        "jpeg_bytes": bytes(data.get("jpeg_bytes", b"")),
    }


def _tracklet_observation_from_dict(data: dict[str, Any]) -> TrackletObservation:
    return TrackletObservation(
        camera_id=str(data["camera_id"]),
        tracker_track_id=int(data["tracker_track_id"]),
        timestamp=float(data["timestamp"]),
        bbox_xyxy=_bbox_from_value(data["bbox_xyxy"]) or (0, 0, 0, 0),
        ground_anchor_world=_point_from_value(data.get("ground_anchor_world")),
        ground_anchor_image=_point_from_value(data.get("ground_anchor_image")),
        confidence=float(data.get("confidence", 0.0)),
        appearance_embedding=[float(value) for value in data.get("appearance_embedding", [])],
        frame_index=int(data.get("frame_index", 0)),
        media_time_s=_optional_float(data.get("media_time_s")),
        entry_edge=_optional_str(data.get("entry_edge")),
        exit_edge=_optional_str(data.get("exit_edge")),
    )


def _local_track_from_dict(data: dict[str, Any]) -> LocalTrack:
    return LocalTrack(
        camera_id=str(data["camera_id"]),
        local_track_id=int(data["local_track_id"]),
        positions_world=_points_from_value(data.get("positions_world")) or [],
        last_seen_ts=float(data.get("last_seen_ts", 0.0)),
        velocity=_point_from_value(data.get("velocity")) or (0.0, 0.0),
        active_zone_id=_optional_str(data.get("active_zone_id")),
        visited_zone_ids=[str(value) for value in data.get("visited_zone_ids", [])],
        current_bbox_xyxy=_bbox_from_value(data.get("current_bbox_xyxy")),
        ground_anchor_image=_point_from_value(data.get("ground_anchor_image")),
        ground_anchor_world=_point_from_value(data.get("ground_anchor_world")),
        confidence=float(data.get("confidence", 0.0)),
        first_seen_ts=float(data.get("first_seen_ts", 0.0)),
        active=bool(data.get("active", True)),
        candidate_zone_id=_optional_str(data.get("candidate_zone_id")),
        candidate_zone_since=_optional_float(data.get("candidate_zone_since")),
        zone_entered_at=_optional_float(data.get("zone_entered_at")),
        missed_frames=int(data.get("missed_frames", 0)),
        last_bbox_xyxy_for_matching=_bbox_from_value(data.get("last_bbox_xyxy_for_matching")),
        smoothed_ground_anchor_world=_point_from_value(data.get("smoothed_ground_anchor_world")),
        match_score=_optional_float(data.get("match_score")),
        display_track_id=_optional_str(data.get("display_track_id")),
        appearance_descriptor=[float(value) for value in data.get("appearance_descriptor", [])],
        observed_frames=int(data.get("observed_frames", 0)),
        confirmed=bool(data.get("confirmed", False)),
        last_exit_edge=_optional_str(data.get("last_exit_edge")),
        last_entry_edge=_optional_str(data.get("last_entry_edge")),
        edge_proximity_score=float(data.get("edge_proximity_score", 0.0)),
        bbox_center_image=_point_from_value(data.get("bbox_center_image")),
    )


def _camera_identity_track_from_dict(data: dict[str, Any]) -> CameraIdentityTrack:
    return CameraIdentityTrack(
        camera_person_id=str(data["camera_person_id"]),
        camera_id=str(data["camera_id"]),
        member_tracklet_keys=[str(value) for value in data.get("member_tracklet_keys", [])],
        active_tracklet_keys=[str(value) for value in data.get("active_tracklet_keys", [])],
        active_tracker_track_ids=[int(value) for value in data.get("active_tracker_track_ids", [])],
        appearance_prototype=[float(value) for value in data.get("appearance_prototype", [])],
        appearance_memory=[
            [float(component) for component in item]
            for item in data.get("appearance_memory", [])
        ],
        first_seen_ts=float(data.get("first_seen_ts", 0.0)),
        last_seen_ts=float(data.get("last_seen_ts", 0.0)),
        active=bool(data.get("active", True)),
        current_bbox_xyxy=_bbox_from_value(data.get("current_bbox_xyxy")),
        ground_anchor_world=_point_from_value(data.get("ground_anchor_world")),
        ground_anchor_image=_point_from_value(data.get("ground_anchor_image")),
        smoothed_ground_anchor_world=_point_from_value(data.get("smoothed_ground_anchor_world")),
        confidence=float(data.get("confidence", 0.0)),
        state=str(data.get("state", "tentative")),
        world_trajectory=_points_from_value(data.get("world_trajectory")) or [],
        positions_world=_points_from_value(data.get("positions_world")) or [],
        velocity=_point_from_value(data.get("velocity")) or (0.0, 0.0),
        last_entry_edge=_optional_str(data.get("last_entry_edge")),
        last_exit_edge=_optional_str(data.get("last_exit_edge")),
        edge_proximity_score=float(data.get("edge_proximity_score", 0.0)),
        bbox_center_image=_point_from_value(data.get("bbox_center_image")),
        inactive_since_ts=_optional_float(data.get("inactive_since_ts")),
        support_count=int(data.get("support_count", 0)),
        current_local_track_id=_optional_int(data.get("current_local_track_id")),
        display_track_id=_optional_str(data.get("display_track_id")),
        observed_frames=int(data.get("observed_frames", 0)),
        last_match_score=_optional_float(data.get("last_match_score")),
        last_match_reason=_optional_str(data.get("last_match_reason")),
        raw_tracker_track_ids=[int(value) for value in data.get("raw_tracker_track_ids", [])],
        entered_overlap_ts=_optional_float(data.get("entered_overlap_ts")),
        left_overlap_ts=_optional_float(data.get("left_overlap_ts")),
        overlap_presence_count=int(data.get("overlap_presence_count", 0)),
        overlap_exit_side=_optional_str(data.get("overlap_exit_side")),
    )


def _identity_debug_record_from_dict(data: dict[str, Any]) -> IdentityDebugRecord:
    return IdentityDebugRecord(
        camera_id=str(data["camera_id"]),
        tracker_track_id=int(data.get("tracker_track_id", 0)),
        camera_person_id=str(data.get("camera_person_id", "")),
        reason=str(data.get("reason", "")),
        score=float(data.get("score", 0.0)),
        global_candidate_id=_optional_str(data.get("global_candidate_id")),
        stage=str(data.get("stage", "single_camera")),
        passed_threshold=bool(data.get("passed_threshold", False)),
        appearance_score=_optional_float(data.get("appearance_score")),
        world_score=_optional_float(data.get("world_score")),
        timing_score=_optional_float(data.get("timing_score")),
        motion_score=_optional_float(data.get("motion_score")),
        transition_score=_optional_float(data.get("transition_score")),
        second_best_margin=_optional_float(data.get("second_best_margin")),
        overlap_inside_left=_optional_bool(data.get("overlap_inside_left")),
        overlap_inside_right=_optional_bool(data.get("overlap_inside_right")),
        world_distance_m=_optional_float(data.get("world_distance_m")),
        appearance_available=_optional_bool(data.get("appearance_available")),
        normalized_total_score=_optional_float(data.get("normalized_total_score")),
        confirmation_progress=_optional_int(data.get("confirmation_progress")),
    )


def _point_list(points: list[tuple[float, float]] | None) -> list[list[float]] | None:
    if points is None:
        return None
    return [[float(x), float(y)] for x, y in points]


def _point_from_value(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    return float(value[0]), float(value[1])


def _points_from_value(value: Any) -> list[tuple[float, float]] | None:
    if value is None:
        return None
    return [(_point_from_value(item) or (0.0, 0.0)) for item in value]


def _bbox_from_value(value: Any) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    return int(value[0]), int(value[1]), int(value[2]), int(value[3])


def _frame_size_from_value(value: Any) -> tuple[int, int]:
    if not value:
        return 0, 0
    return int(value[0]), int(value[1])


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)
