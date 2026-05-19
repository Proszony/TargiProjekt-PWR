from __future__ import annotations

import math

from core.embedding_math import blend_embedding, cosine_similarity
from core.models import CameraIdentityTrack, GalleryEntry, IdentityConfig, IdentityDebugRecord, LocalTrack, ReIDConfig, TrackletObservation


class CameraIdentityEngine:
    def __init__(
        self,
        camera_id: str,
        identity_config: IdentityConfig,
        reid_config: ReIDConfig | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.identity_config = identity_config
        self.reid_config = reid_config or ReIDConfig()
        self._tracks: dict[str, CameraIdentityTrack] = {}
        self._local_to_person: dict[str, str] = {}
        self._next_person_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._local_to_person.clear()
        self._next_person_id = 1

    def update(
        self,
        timestamp: float,
        local_tracks: dict[int, LocalTrack],
        expired_tracks: list[LocalTrack],
        observations: list[TrackletObservation],
    ) -> tuple[dict[str, CameraIdentityTrack], list[CameraIdentityTrack], list[IdentityDebugRecord]]:
        debug_records: list[IdentityDebugRecord] = []
        observation_by_key = {observation.local_key: observation for observation in observations}
        emitted_expired: list[CameraIdentityTrack] = []

        for identity_track in self._tracks.values():
            identity_track.active_tracklet_keys = []
            identity_track.active_tracker_track_ids = []
            identity_track.support_count = 0
            identity_track.last_match_score = None
            identity_track.last_match_reason = None

        unmatched: list[tuple[str, LocalTrack]] = []
        for local_track_id, local_track in local_tracks.items():
            local_key = f"{self.camera_id}:{local_track_id}"
            camera_person_id = self._local_to_person.get(local_key)
            if camera_person_id is None or camera_person_id not in self._tracks:
                unmatched.append((local_key, local_track))
                continue
            identity_track = self._tracks[camera_person_id]
            self._attach(identity_track, local_key, local_track, observation_by_key.get(local_key), 1.0, "existing_tracklet")
            debug_records.append(
                IdentityDebugRecord(
                    camera_id=self.camera_id,
                    tracker_track_id=local_track.local_track_id,
                    camera_person_id=camera_person_id,
                    reason="existing_tracklet",
                    score=1.0,
                )
            )

        for local_key, local_track in unmatched:
            observation = observation_by_key.get(local_key)
            match = self._match_existing_identity(local_track, observation, timestamp)
            if match is None:
                camera_person_id = self._create_camera_person_id()
                identity_track = CameraIdentityTrack(
                    camera_person_id=camera_person_id,
                    camera_id=self.camera_id,
                    first_seen_ts=local_track.first_seen_ts,
                    last_seen_ts=local_track.last_seen_ts,
                    state="confirmed" if local_track.confirmed else "tentative",
                    display_track_id=camera_person_id.split(":")[-1],
                )
                self._tracks[camera_person_id] = identity_track
                score = 1.0
                reason = "new_identity"
            else:
                camera_person_id, score, reason = match
                identity_track = self._tracks[camera_person_id]
            self._local_to_person[local_key] = camera_person_id
            self._attach(identity_track, local_key, local_track, observation, score, reason)
            debug_records.append(
                IdentityDebugRecord(
                    camera_id=self.camera_id,
                    tracker_track_id=local_track.local_track_id,
                    camera_person_id=camera_person_id,
                    reason=reason,
                    score=score,
                )
            )

        for track in local_tracks.values():
            local_key = f"{self.camera_id}:{track.local_track_id}"
            camera_person_id = self._local_to_person.get(local_key)
            if camera_person_id:
                stable_id = camera_person_id.split(":")[-1]
                track.display_track_id = f"{stable_id}|L{track.local_track_id}"

        for expired_track in expired_tracks:
            local_key = f"{self.camera_id}:{expired_track.local_track_id}"
            camera_person_id = self._local_to_person.get(local_key)
            if camera_person_id is None:
                continue
            identity_track = self._tracks.get(camera_person_id)
            if identity_track is None:
                continue
            if local_key in identity_track.active_tracklet_keys:
                identity_track.active_tracklet_keys.remove(local_key)
            if expired_track.local_track_id in identity_track.active_tracker_track_ids:
                identity_track.active_tracker_track_ids.remove(expired_track.local_track_id)
            identity_track.support_count = len(identity_track.active_tracklet_keys)
            identity_track.last_seen_ts = max(identity_track.last_seen_ts, expired_track.last_seen_ts)
            identity_track.last_exit_edge = expired_track.last_exit_edge
            if not identity_track.active_tracklet_keys:
                identity_track.active = False
                identity_track.state = "lost"
                identity_track.inactive_since_ts = expired_track.last_seen_ts
                emitted_expired.append(self._copy_track(identity_track))

        for camera_person_id, identity_track in list(self._tracks.items()):
            if identity_track.active_tracklet_keys:
                identity_track.active = True
                if identity_track.observed_frames >= self.identity_config.confirmation_frames:
                    identity_track.state = "confirmed"
                elif identity_track.state == "inactive":
                    identity_track.state = "tentative"
                continue
            if identity_track.inactive_since_ts is None:
                identity_track.inactive_since_ts = identity_track.last_seen_ts
            if timestamp - identity_track.inactive_since_ts > self.identity_config.lost_track_retention_s:
                identity_track.state = "inactive"
                identity_track.active = False
                continue
            identity_track.active = False
            identity_track.state = "lost"

        active_tracks = {
            camera_person_id: self._copy_track(identity_track)
            for camera_person_id, identity_track in self._tracks.items()
            if identity_track.active
        }
        return active_tracks, emitted_expired, debug_records

    def _match_existing_identity(
        self,
        local_track: LocalTrack,
        observation: TrackletObservation | None,
        timestamp: float,
    ) -> tuple[str, float, str] | None:
        if not self.identity_config.single_camera_restitch_enabled:
            return None
        best: tuple[str, float, str] | None = None
        for camera_person_id, identity_track in self._tracks.items():
            if identity_track.state == "inactive":
                continue
            score = self._restitch_score(local_track, observation, identity_track, timestamp)
            if score is None or score < self.identity_config.single_camera_restitch_threshold:
                continue
            if best is None or score > best[1]:
                reason = "restitch_lost" if not identity_track.active else "merge_duplicate"
                best = (camera_person_id, score, reason)
        return best

    def _restitch_score(
        self,
        local_track: LocalTrack,
        observation: TrackletObservation | None,
        identity_track: CameraIdentityTrack,
        timestamp: float,
    ) -> float | None:
        delta_t = max(timestamp - identity_track.last_seen_ts, 0.0)
        if delta_t > self.identity_config.single_camera_max_gap_s:
            return None
        allowed_distance = (
            self.identity_config.single_camera_base_distance_m
            + self.identity_config.single_camera_speed_m_per_s * delta_t
        )
        world_distance = self._world_distance(local_track, identity_track)
        if world_distance is None or world_distance > allowed_distance:
            return None
        world_score = 1.0 - min(world_distance / max(allowed_distance, 1e-6), 1.0)
        appearance_score = self._appearance_score(local_track, observation, identity_track)
        timing_score = 1.0 - min(delta_t / max(self.identity_config.single_camera_max_gap_s, 1e-6), 1.0)
        motion_score = self._motion_similarity(local_track.velocity, identity_track.velocity)
        edge_score = self._edge_transition_score(local_track, identity_track)
        return (
            0.50 * appearance_score
            + 0.25 * max(world_score, motion_score)
            + 0.15 * timing_score
            + 0.10 * edge_score
        )

    @staticmethod
    def _world_distance(local_track: LocalTrack, identity_track: CameraIdentityTrack) -> float | None:
        point_a = local_track.smoothed_ground_anchor_world or local_track.ground_anchor_world
        point_b = identity_track.smoothed_ground_anchor_world or identity_track.ground_anchor_world
        if point_a is None or point_b is None:
            return None
        return math.dist(point_a, point_b)

    @staticmethod
    def _motion_similarity(left: tuple[float, float], right: tuple[float, float]) -> float:
        left_norm = math.hypot(*left)
        right_norm = math.hypot(*right)
        if left_norm <= 1e-6 or right_norm <= 1e-6:
            return 0.0
        cosine = (left[0] * right[0] + left[1] * right[1]) / max(left_norm * right_norm, 1e-6)
        return max(0.0, min((cosine + 1.0) / 2.0, 1.0))

    @staticmethod
    def _edge_transition_score(local_track: LocalTrack, identity_track: CameraIdentityTrack) -> float:
        if (
            local_track.last_entry_edge is not None
            and identity_track.last_exit_edge is not None
            and local_track.last_entry_edge == identity_track.last_exit_edge
        ):
            return 1.0
        return max(local_track.edge_proximity_score, identity_track.edge_proximity_score) * 0.5

    def _appearance_score(
        self,
        local_track: LocalTrack,
        observation: TrackletObservation | None,
        identity_track: CameraIdentityTrack,
    ) -> float:
        candidate = observation.appearance_embedding if observation is not None else local_track.appearance_descriptor
        if not candidate:
            candidate = local_track.appearance_descriptor
        if not candidate:
            return 0.0
        memory = list(identity_track.appearance_memory)
        if identity_track.appearance_prototype:
            memory.append(identity_track.appearance_prototype)
        if not memory:
            return 0.0
        topk = sorted(
            (_embedding_similarity(candidate, reference) for reference in memory),
            reverse=True,
        )[: max(1, self.reid_config.match_topk)]
        if self.reid_config.match_reduce == "max_topk":
            return topk[0]
        count = min(3, len(topk))
        return sum(topk[:count]) / count

    def _attach(
        self,
        identity_track: CameraIdentityTrack,
        local_key: str,
        local_track: LocalTrack,
        observation: TrackletObservation | None,
        score: float,
        reason: str,
    ) -> None:
        if local_key not in identity_track.member_tracklet_keys:
            identity_track.member_tracklet_keys.append(local_key)
        if local_key not in identity_track.active_tracklet_keys:
            identity_track.active_tracklet_keys.append(local_key)
        if local_track.local_track_id not in identity_track.active_tracker_track_ids:
            identity_track.active_tracker_track_ids.append(local_track.local_track_id)
        if local_track.local_track_id not in identity_track.raw_tracker_track_ids:
            identity_track.raw_tracker_track_ids.append(local_track.local_track_id)
        identity_track.support_count = len(identity_track.active_tracklet_keys)
        identity_track.current_local_track_id = local_track.local_track_id
        identity_track.current_bbox_xyxy = local_track.current_bbox_xyxy
        identity_track.ground_anchor_world = local_track.ground_anchor_world
        identity_track.ground_anchor_image = local_track.ground_anchor_image
        identity_track.smoothed_ground_anchor_world = local_track.smoothed_ground_anchor_world
        identity_track.confidence = local_track.confidence
        identity_track.velocity = local_track.velocity
        identity_track.last_seen_ts = max(identity_track.last_seen_ts, local_track.last_seen_ts)
        identity_track.first_seen_ts = (
            min(identity_track.first_seen_ts, local_track.first_seen_ts)
            if identity_track.first_seen_ts > 0.0
            else local_track.first_seen_ts
        )
        identity_track.active = True
        identity_track.inactive_since_ts = None
        identity_track.last_entry_edge = local_track.last_entry_edge
        identity_track.last_exit_edge = local_track.last_exit_edge
        identity_track.edge_proximity_score = local_track.edge_proximity_score
        identity_track.bbox_center_image = local_track.bbox_center_image
        identity_track.observed_frames += 1
        identity_track.last_match_score = score
        identity_track.last_match_reason = reason
        identity_track.display_track_id = identity_track.camera_person_id.split(":")[-1]
        if local_track.ground_anchor_world is not None:
            identity_track.positions_world.append(local_track.ground_anchor_world)
            identity_track.world_trajectory.append(local_track.ground_anchor_world)
            if len(identity_track.positions_world) > 180:
                identity_track.positions_world = identity_track.positions_world[-180:]
            if len(identity_track.world_trajectory) > 360:
                identity_track.world_trajectory = identity_track.world_trajectory[-360:]
        appearance = (
            observation.appearance_embedding
            if observation is not None and observation.appearance_embedding
            else local_track.appearance_descriptor
        )
        if appearance:
            identity_track.appearance_memory.append(list(appearance))
            if len(identity_track.appearance_memory) > self.reid_config.max_embedding_memory:
                identity_track.appearance_memory = identity_track.appearance_memory[-self.reid_config.max_embedding_memory :]
            identity_track.appearance_prototype = blend_embedding(
                identity_track.appearance_prototype,
                appearance,
                momentum=self.reid_config.embedding_momentum,
            )
            overlap_state = "overlap" if local_track.edge_proximity_score > 0.65 else "single_camera"
            identity_track.appearance_gallery.append(
                GalleryEntry(
                    embedding=list(appearance),
                    camera_id=self.camera_id,
                    camera_person_id=identity_track.camera_person_id,
                    timestamp=local_track.last_seen_ts,
                    quality_score=max(local_track.confidence, score),
                    overlap_state=overlap_state,
                )
            )
            if len(identity_track.appearance_gallery) > self.reid_config.gallery_size_per_camera:
                identity_track.appearance_gallery = identity_track.appearance_gallery[-self.reid_config.gallery_size_per_camera :]

    def _create_camera_person_id(self) -> str:
        camera_person_id = f"{self.camera_id}:P{self._next_person_id:05d}"
        self._next_person_id += 1
        return camera_person_id

    @staticmethod
    def _copy_track(identity_track: CameraIdentityTrack) -> CameraIdentityTrack:
        return CameraIdentityTrack(
            camera_person_id=identity_track.camera_person_id,
            camera_id=identity_track.camera_id,
            member_tracklet_keys=list(identity_track.member_tracklet_keys),
            active_tracklet_keys=list(identity_track.active_tracklet_keys),
            active_tracker_track_ids=list(identity_track.active_tracker_track_ids),
            appearance_prototype=list(identity_track.appearance_prototype),
            appearance_memory=[list(item) for item in identity_track.appearance_memory],
            first_seen_ts=identity_track.first_seen_ts,
            last_seen_ts=identity_track.last_seen_ts,
            active=identity_track.active,
            current_bbox_xyxy=identity_track.current_bbox_xyxy,
            ground_anchor_world=identity_track.ground_anchor_world,
            ground_anchor_image=identity_track.ground_anchor_image,
            smoothed_ground_anchor_world=identity_track.smoothed_ground_anchor_world,
            confidence=identity_track.confidence,
            state=identity_track.state,
            world_trajectory=list(identity_track.world_trajectory),
            positions_world=list(identity_track.positions_world),
            velocity=identity_track.velocity,
            last_entry_edge=identity_track.last_entry_edge,
            last_exit_edge=identity_track.last_exit_edge,
            edge_proximity_score=identity_track.edge_proximity_score,
            bbox_center_image=identity_track.bbox_center_image,
            inactive_since_ts=identity_track.inactive_since_ts,
            support_count=identity_track.support_count,
            current_local_track_id=identity_track.current_local_track_id,
            display_track_id=identity_track.display_track_id,
            observed_frames=identity_track.observed_frames,
            last_match_score=identity_track.last_match_score,
            last_match_reason=identity_track.last_match_reason,
            raw_tracker_track_ids=list(identity_track.raw_tracker_track_ids),
            entered_overlap_ts=identity_track.entered_overlap_ts,
            left_overlap_ts=identity_track.left_overlap_ts,
            overlap_presence_count=identity_track.overlap_presence_count,
            overlap_exit_side=identity_track.overlap_exit_side,
            appearance_gallery=[
                GalleryEntry(
                    embedding=list(entry.embedding),
                    camera_id=entry.camera_id,
                    camera_person_id=entry.camera_person_id,
                    timestamp=entry.timestamp,
                    quality_score=entry.quality_score,
                    overlap_state=entry.overlap_state,
                )
                for entry in identity_track.appearance_gallery
            ],
        )


def _embedding_similarity(left: list[float], right: list[float]) -> float:
    return cosine_similarity(left, right)
