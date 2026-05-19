from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.models import Detection, LocalTrack, Point


@dataclass(slots=True)
class _TrackState:
    track: LocalTrack
    missed_since: float = 0.0


class SimpleWorldTracker:
    def __init__(
        self,
        track_timeout_s: float = 2.0,
        max_world_distance_m: float = 0.45,
        max_image_distance_px: float = 70.0,
        max_missed_frames: int = 12,
        min_iou: float = 0.15,
        anchor_weight: float = 0.7,
        iou_weight: float = 0.3,
        confidence_weight: float = 0.05,
        smoothing_window: int = 3,
        confirmation_frames: int = 3,
    ) -> None:
        self.track_timeout_s = track_timeout_s
        self.max_world_distance_m = max_world_distance_m
        self.max_image_distance_px = max_image_distance_px
        self.max_missed_frames = max_missed_frames
        self.min_iou = min_iou
        self.anchor_weight = anchor_weight
        self.iou_weight = iou_weight
        self.confidence_weight = confidence_weight
        self.smoothing_window = smoothing_window
        self.confirmation_frames = confirmation_frames
        self._next_track_id = 1
        self._tracks: dict[int, _TrackState] = {}

    def reset(self) -> None:
        self._next_track_id = 1
        self._tracks.clear()

    def update(
        self,
        camera_id: str,
        timestamp: float,
        detections: list[Detection],
    ) -> tuple[dict[int, LocalTrack], list[LocalTrack]]:
        if detections and all(detection.track_id is not None for detection in detections):
            return self._update_from_tracked_detections(camera_id, timestamp, detections)

        active_tracks = {track_id: state for track_id, state in self._tracks.items() if state.track.active}
        matches, unmatched_tracks, unmatched_detections = self._match(active_tracks, detections)

        for track_id, detection_index in matches.items():
            detection = detections[detection_index]
            self._update_track(active_tracks[track_id].track, detection, timestamp)
            active_tracks[track_id].missed_since = 0.0

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            track = LocalTrack(
                camera_id=camera_id,
                local_track_id=self._next_track_id,
                positions_world=[detection.ground_anchor_world] if detection.ground_anchor_world else [],
                last_seen_ts=timestamp,
                current_bbox_xyxy=detection.person_bbox_xyxy,
                last_bbox_xyxy_for_matching=detection.person_bbox_xyxy,
                ground_anchor_image=detection.ground_anchor_image,
                ground_anchor_world=detection.ground_anchor_world,
                smoothed_ground_anchor_world=detection.ground_anchor_world,
                confidence=detection.confidence,
                first_seen_ts=timestamp,
                active=True,
                appearance_descriptor=detection.appearance_descriptor,
                observed_frames=1,
                confirmed=self.confirmation_frames <= 1,
            )
            self._tracks[self._next_track_id] = _TrackState(track=track)
            self._next_track_id += 1

        expired: list[LocalTrack] = []
        for track_id in list(unmatched_tracks):
            state = self._tracks[track_id]
            if state.missed_since == 0.0:
                state.missed_since = timestamp
            state.track.missed_frames += 1
            if (
                timestamp - state.track.last_seen_ts >= self.track_timeout_s
                or state.track.missed_frames > self.max_missed_frames
            ):
                state.track.active = False
                expired.append(state.track)
                del self._tracks[track_id]

        current_tracks = {
            track_id: state.track
            for track_id, state in self._tracks.items()
            if state.track.active and state.track.confirmed
        }
        return current_tracks, expired

    def _update_from_tracked_detections(
        self,
        camera_id: str,
        timestamp: float,
        detections: list[Detection],
    ) -> tuple[dict[int, LocalTrack], list[LocalTrack]]:
        seen_ids = {int(detection.track_id) for detection in detections if detection.track_id is not None}
        expired: list[LocalTrack] = []

        for detection in detections:
            if detection.track_id is None:
                continue
            track_id = int(detection.track_id)
            state = self._tracks.get(track_id)
            if state is None:
                track = LocalTrack(
                    camera_id=camera_id,
                    local_track_id=track_id,
                    positions_world=[detection.ground_anchor_world] if detection.ground_anchor_world else [],
                    last_seen_ts=timestamp,
                    current_bbox_xyxy=detection.person_bbox_xyxy,
                    last_bbox_xyxy_for_matching=detection.person_bbox_xyxy,
                    ground_anchor_image=detection.ground_anchor_image,
                    ground_anchor_world=detection.ground_anchor_world,
                    smoothed_ground_anchor_world=detection.ground_anchor_world,
                    confidence=detection.confidence,
                    first_seen_ts=timestamp,
                    active=True,
                    appearance_descriptor=detection.appearance_descriptor,
                    observed_frames=1,
                    confirmed=self.confirmation_frames <= 1,
                )
                self._tracks[track_id] = _TrackState(track=track)
                continue
            self._update_track(state.track, detection, timestamp)
            state.track.active = True
            state.missed_since = 0.0

        for track_id in list(self._tracks):
            if track_id in seen_ids:
                continue
            state = self._tracks[track_id]
            state.track.missed_frames += 1
            if state.track.missed_frames > self.max_missed_frames:
                state.track.active = False
                expired.append(state.track)
                del self._tracks[track_id]

        current_tracks = {
            track_id: state.track
            for track_id, state in self._tracks.items()
            if state.track.active and state.track.confirmed
        }
        return current_tracks, expired

    def _match(
        self,
        tracks: dict[int, _TrackState],
        detections: list[Detection],
    ) -> tuple[dict[int, int], set[int], set[int]]:
        unmatched_tracks = set(tracks.keys())
        unmatched_detections = set(range(len(detections)))
        candidates: list[tuple[float, int, int]] = []

        for track_id, state in tracks.items():
            for detection_index, detection in enumerate(detections):
                score = self._score_match(state.track, detection)
                if score is None:
                    continue
                candidates.append((score, track_id, detection_index))

        matches: dict[int, int] = {}
        for score, track_id, detection_index in sorted(candidates, key=lambda item: item[0]):
            if track_id not in unmatched_tracks or detection_index not in unmatched_detections:
                continue
            matches[track_id] = detection_index
            tracks[track_id].track.match_score = score
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(detection_index)

        return matches, unmatched_tracks, unmatched_detections

    def _score_match(self, track: LocalTrack, detection: Detection) -> float | None:
        anchor_score = self._anchor_score(track, detection)
        if anchor_score is None:
            return None
        iou = self._bbox_iou(track.last_bbox_xyxy_for_matching, detection.person_bbox_xyxy)
        appearance_similarity = self._appearance_similarity(
            track.appearance_descriptor,
            detection.appearance_descriptor,
        )
        if (
            track.last_bbox_xyxy_for_matching is not None
            and iou < self.min_iou
            and not self._allow_recent_reacquisition(track, anchor_score, appearance_similarity)
        ):
            return None
        return (
            self.anchor_weight * anchor_score
            + self.iou_weight * (1.0 - iou)
            - self.confidence_weight * detection.confidence
            - 0.08 * appearance_similarity
        )

    def _anchor_score(self, track: LocalTrack, detection: Detection) -> float | None:
        if track.smoothed_ground_anchor_world and detection.ground_anchor_world:
            distance = math.dist(track.smoothed_ground_anchor_world, detection.ground_anchor_world)
            if distance > self.max_world_distance_m:
                return None
            return distance / max(self.max_world_distance_m, 1e-6)
        if track.ground_anchor_image is not None:
            distance = math.dist(track.ground_anchor_image, detection.ground_anchor_image)
            if distance > self.max_image_distance_px:
                return None
            return distance / max(self.max_image_distance_px, 1e-6)
        return None

    def _update_track(self, track: LocalTrack, detection: Detection, timestamp: float) -> None:
        previous_point = track.ground_anchor_world
        previous_ts = track.last_seen_ts
        track.current_bbox_xyxy = detection.person_bbox_xyxy
        track.last_bbox_xyxy_for_matching = detection.person_bbox_xyxy
        track.ground_anchor_image = detection.ground_anchor_image
        track.ground_anchor_world = detection.ground_anchor_world
        track.confidence = detection.confidence
        track.last_seen_ts = timestamp
        track.missed_frames = 0
        track.appearance_descriptor = detection.appearance_descriptor
        track.observed_frames += 1
        if track.observed_frames >= self.confirmation_frames:
            track.confirmed = True
        if detection.ground_anchor_world is not None:
            track.positions_world.append(detection.ground_anchor_world)
            if len(track.positions_world) > 60:
                track.positions_world = track.positions_world[-60:]
            window = track.positions_world[-self.smoothing_window :]
            track.smoothed_ground_anchor_world = (
                sum(point[0] for point in window) / len(window),
                sum(point[1] for point in window) / len(window),
            )
        if previous_point and detection.ground_anchor_world and previous_ts > 0.0:
            dt = max(timestamp - previous_ts, 1e-6)
            dx = detection.ground_anchor_world[0] - previous_point[0]
            dy = detection.ground_anchor_world[1] - previous_point[1]
            track.velocity = (dx / dt, dy / dt)

    def _allow_recent_reacquisition(
        self,
        track: LocalTrack,
        anchor_score: float,
        appearance_similarity: float,
    ) -> bool:
        if track.missed_frames <= 0:
            return anchor_score <= 0.28 and appearance_similarity >= 0.45
        if track.missed_frames <= 2 and anchor_score <= 0.35:
            return True
        if anchor_score <= 0.2:
            return True
        if anchor_score <= 0.45 and appearance_similarity >= 0.35:
            return True
        return anchor_score <= 0.60 and appearance_similarity >= 0.50

    @staticmethod
    def _appearance_similarity(descriptor_a: list[float], descriptor_b: list[float]) -> float:
        if not descriptor_a or not descriptor_b or len(descriptor_a) != len(descriptor_b):
            return 0.0
        return sum(min(left, right) for left, right in zip(descriptor_a, descriptor_b))

    @staticmethod
    def _bbox_iou(
        bbox_a: tuple[int, int, int, int] | None,
        bbox_b: tuple[int, int, int, int] | None,
    ) -> float:
        if bbox_a is None or bbox_b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
        area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
        union = area_a + area_b - intersection
        if union <= 0:
            return 0.0
        return intersection / union
