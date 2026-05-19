from __future__ import annotations

import math
from itertools import combinations

from core.camera_overlap import point_distance_to_polygon, point_in_polygon, point_inside_or_near_overlap
from core.embedding_math import cosine_similarity
from core.models import (
    AnalyticsTrack,
    CameraIdentityTrack,
    CameraOverlapGraph,
    IdentityDebugRecord,
    MapDedupConfig,
    MapDedupDebugStats,
    MapPresence,
    OverlapTracklet,
    OverlapDedupConfig,
    ReIDConfig,
    StableMapPresence,
)


class OverlapDedupEngine:
    def __init__(
        self,
        dedup_config: OverlapDedupConfig | None = None,
        reid_config: ReIDConfig | None = None,
    ) -> None:
        self.dedup_config = dedup_config or OverlapDedupConfig()
        self.reid_config = reid_config or ReIDConfig()
        self._tracks: dict[str, AnalyticsTrack] = {}
        self._local_to_analytics: dict[str, str] = {}
        self._pair_confirmations: dict[tuple[str, str], int] = {}
        self._map_local_to_presence: dict[str, str] = {}
        self._map_pair_confirmations: dict[tuple[str, str], int] = {}
        self._overlap_tracklets: dict[str, OverlapTracklet] = {}
        self._stable_map_presences: dict[str, StableMapPresence] = {}
        self._next_analytics_id = 1
        self._debug_records: list[IdentityDebugRecord] = []
        self._last_map_pairs_considered = 0
        self._last_map_pairs_merged = 0
        self._map_debug_stats = MapDedupDebugStats()
        self._map_publish_ttl_s = 0.20

    def reset(self) -> None:
        self._tracks.clear()
        self._local_to_analytics.clear()
        self._pair_confirmations.clear()
        self._map_local_to_presence.clear()
        self._map_pair_confirmations.clear()
        self._overlap_tracklets.clear()
        self._stable_map_presences.clear()
        self._next_analytics_id = 1
        self._debug_records.clear()
        self._last_map_pairs_considered = 0
        self._last_map_pairs_merged = 0
        self._map_debug_stats = MapDedupDebugStats()
        self._map_publish_ttl_s = 0.20

    def consume_debug_records(self) -> list[IdentityDebugRecord]:
        records = list(self._debug_records)
        self._debug_records.clear()
        return records

    def last_map_presence_stats(self) -> tuple[int, int]:
        return self._last_map_pairs_considered, self._last_map_pairs_merged

    def map_debug_stats(self) -> MapDedupDebugStats:
        return MapDedupDebugStats(
            overlap_tracklets_active=self._map_debug_stats.overlap_tracklets_active,
            candidates_considered=self._map_debug_stats.candidates_considered,
            matches_committed=self._map_debug_stats.matches_committed,
            matches_rejected_geometry=self._map_debug_stats.matches_rejected_geometry,
            matches_rejected_margin=self._map_debug_stats.matches_rejected_margin,
            matches_rejected_time=self._map_debug_stats.matches_rejected_time,
            matches_without_appearance=self._map_debug_stats.matches_without_appearance,
        )

    def update(
        self,
        timestamp: float,
        active_local_tracks_by_camera: dict[str, dict[str, CameraIdentityTrack]],
        expired_local_tracks: list[CameraIdentityTrack],
        overlap_graph: CameraOverlapGraph | None = None,
    ) -> dict[str, AnalyticsTrack]:
        graph = overlap_graph or CameraOverlapGraph()
        flattened = self._flatten(active_local_tracks_by_camera)
        current_keys = {local_key for local_key, _track in flattened}
        self._debug_records = []

        for track in self._tracks.values():
            track.active = False
            track.active_camera_person_ids = {}
            track.source_camera_ids = []

        confirmed_pairs = (
            self._resolve_confirmed_pairs(timestamp, flattened, graph)
            if self.dedup_config.enabled
            else []
        )
        resolved_analytics_ids = self._resolve_group_assignments(confirmed_pairs)

        for local_key, track in flattened:
            analytics_id = resolved_analytics_ids.get(local_key)
            if analytics_id is None:
                analytics_id = self._local_to_analytics.get(local_key)
            if analytics_id is None:
                analytics_id = self._create_analytics_id()
            self._local_to_analytics[local_key] = analytics_id
            self._attach(analytics_id, local_key, track, timestamp)

        expired_keys = {track.camera_person_id for track in expired_local_tracks}
        for local_key in list(self._local_to_analytics):
            if local_key in current_keys:
                continue
            if local_key in expired_keys:
                self._local_to_analytics.pop(local_key, None)

        self._expire_stale_tracks(timestamp)
        return {track_id: self._copy_track(track) for track_id, track in self._tracks.items()}

    def resolve_map_presences(
        self,
        timestamp: float,
        active_local_tracks_by_camera: dict[str, dict[str, CameraIdentityTrack]],
        overlap_graph: CameraOverlapGraph,
        map_dedup_config: MapDedupConfig,
    ) -> list[MapPresence]:
        flattened = self._flatten(active_local_tracks_by_camera)
        self._map_publish_ttl_s = map_dedup_config.presence_publish_ttl_s
        current_keys = {local_key for local_key, _track in flattened}
        self._debug_records = []
        active_tracks = {
            local_key: track
            for local_key, track in flattened
            if (track.smoothed_ground_anchor_world or track.ground_anchor_world) is not None
        }
        tracklets = self._refresh_overlap_tracklets(
            timestamp,
            flattened,
            overlap_graph,
            map_dedup_config,
        )
        confirmed_pairs, stats = self._resolve_map_tracklet_pairs(
            timestamp,
            tracklets,
            overlap_graph,
            map_dedup_config,
        )
        self._last_map_pairs_considered = stats.candidates_considered
        self._last_map_pairs_merged = len(confirmed_pairs)
        self._map_debug_stats = stats

        groups = self._build_presence_groups(current_keys, active_tracks, confirmed_pairs)
        self._update_stable_map_presences(timestamp, groups, map_dedup_config)
        self._cleanup_presence_bindings(current_keys)
        presences = self._emit_map_presences(timestamp)
        presences.sort(key=lambda item: item.presence_id)
        return presences

    def _resolve_group_assignments(
        self,
        confirmed_pairs: list[tuple[str, str]],
    ) -> dict[str, str]:
        if not confirmed_pairs:
            return {}
        adjacency: dict[str, set[str]] = {}
        for left_key, right_key in confirmed_pairs:
            adjacency.setdefault(left_key, set()).add(right_key)
            adjacency.setdefault(right_key, set()).add(left_key)

        resolved_ids: dict[str, str] = {}
        visited: set[str] = set()
        for start_key in sorted(adjacency):
            if start_key in visited:
                continue
            stack = [start_key]
            component: set[str] = set()
            while stack:
                local_key = stack.pop()
                if local_key in visited:
                    continue
                visited.add(local_key)
                component.add(local_key)
                stack.extend(adjacency.get(local_key, ()))

            existing_ids = sorted(
                {
                    analytics_id
                    for local_key in component
                    if (analytics_id := self._local_to_analytics.get(local_key)) is not None
                }
            )
            analytics_id = existing_ids[0] if existing_ids else self._create_analytics_id()
            for other_id in existing_ids[1:]:
                self._merge_tracks(analytics_id, other_id)
            for local_key in sorted(component):
                resolved_ids[local_key] = analytics_id
                self._local_to_analytics[local_key] = analytics_id
        return resolved_ids

    @staticmethod
    def _flatten(
        active_local_tracks_by_camera: dict[str, dict[str, CameraIdentityTrack]],
    ) -> list[tuple[str, CameraIdentityTrack]]:
        flattened: list[tuple[str, CameraIdentityTrack]] = []
        for tracks in active_local_tracks_by_camera.values():
            for camera_person_id, track in tracks.items():
                flattened.append((camera_person_id, track))
        flattened.sort(key=lambda item: item[0])
        return flattened

    def _resolve_confirmed_pairs(
        self,
        timestamp: float,
        flattened: list[tuple[str, CameraIdentityTrack]],
        overlap_graph: CameraOverlapGraph,
    ) -> list[tuple[str, str]]:
        candidates: list[tuple[float, str, str, float, float, float, float]] = []
        scores_by_key: dict[str, list[float]] = {}
        for index, (left_key, left_track) in enumerate(flattened):
            for right_key, right_track in flattened[index + 1 :]:
                if left_track.camera_id == right_track.camera_id:
                    continue
                score_tuple = self._score_pair(timestamp, left_track, right_track, overlap_graph)
                if score_tuple is None:
                    continue
                total, appearance, world, timing, geometry = score_tuple
                if total < self.dedup_config.similarity_threshold:
                    continue
                candidates.append((total, left_key, right_key, appearance, world, timing, geometry))
                scores_by_key.setdefault(left_key, []).append(total)
                scores_by_key.setdefault(right_key, []).append(total)

        confirmed: list[tuple[str, str]] = []
        claimed: set[str] = set()
        for total, left_key, right_key, appearance, world, timing, geometry in sorted(
            candidates,
            key=lambda item: item[0],
            reverse=True,
        ):
            if left_key in claimed or right_key in claimed:
                continue
            left_margin = self._score_margin(scores_by_key.get(left_key, []), total)
            right_margin = self._score_margin(scores_by_key.get(right_key, []), total)
            margin = min(left_margin, right_margin)
            pair_key = tuple(sorted((left_key, right_key)))
            passed = margin >= self.dedup_config.margin_min
            self._record_pair_debug(left_key, right_key, total, appearance, world, timing, geometry, margin, passed)
            if not passed:
                self._pair_confirmations.pop(pair_key, None)
                continue
            current = self._pair_confirmations.get(pair_key, 0) + 1
            self._pair_confirmations[pair_key] = current
            if current < self.dedup_config.confirmation_frames:
                continue
            confirmed.append(pair_key)
            claimed.add(left_key)
            claimed.add(right_key)

        active_pairs = {tuple(sorted((left_key, right_key))) for _score, left_key, right_key, *_ in candidates}
        for pair_key in list(self._pair_confirmations):
            if pair_key not in active_pairs:
                self._pair_confirmations.pop(pair_key, None)
        return confirmed

    def _resolve_map_tracklet_pairs(
        self,
        timestamp: float,
        tracklets: dict[str, OverlapTracklet],
        overlap_graph: CameraOverlapGraph,
        map_dedup_config: MapDedupConfig,
    ) -> tuple[list[tuple[str, str]], MapDedupDebugStats]:
        stats = MapDedupDebugStats(overlap_tracklets_active=len(tracklets))
        candidates_by_pair: dict[tuple[str, str], list[dict[str, object]]] = {}
        scores_by_key: dict[str, list[float]] = {}
        tracklet_items = sorted(tracklets.items(), key=lambda item: item[0])
        for index, (left_key, left_tracklet) in enumerate(tracklet_items):
            for right_key, right_tracklet in tracklet_items[index + 1 :]:
                if left_tracklet.camera_id == right_tracklet.camera_id:
                    continue
                relation = overlap_graph.relation_for(left_tracklet.camera_id, right_tracklet.camera_id)
                if relation is None or not relation.is_adjacent or len(relation.intersection_polygon_world) < 3:
                    continue
                score_tuple = self._score_map_tracklet_pair(
                    timestamp,
                    left_tracklet,
                    right_tracklet,
                    relation.intersection_polygon_world,
                    map_dedup_config,
                )
                if score_tuple is None:
                    continue
                rejection_reason = str(score_tuple["rejection_reason"])
                if rejection_reason == "geometry":
                    stats.matches_rejected_geometry += 1
                    continue
                if rejection_reason == "time":
                    stats.matches_rejected_time += 1
                    continue
                stats.candidates_considered += 1
                if bool(score_tuple["appearance_available"]):
                    pass
                else:
                    stats.matches_without_appearance += 1
                total = float(score_tuple["score"])
                if total < map_dedup_config.similarity_threshold:
                    self._record_map_pair_debug(
                        left_key,
                        right_key,
                        total,
                        float(score_tuple["appearance"]),
                        float(score_tuple["world"]),
                        float(score_tuple["timing"]),
                        float(score_tuple["overlap"]),
                        bool(score_tuple["left_inside"]),
                        bool(score_tuple["right_inside"]),
                        float(score_tuple["world_distance"]),
                        bool(score_tuple["appearance_available"]),
                        0.0,
                        0,
                        False,
                    )
                    continue
                camera_pair = tuple(sorted((left_tracklet.camera_id, right_tracklet.camera_id)))
                candidates_by_pair.setdefault(camera_pair, []).append(
                    {
                        "left_key": left_key,
                        "right_key": right_key,
                        **score_tuple,
                    }
                )
                scores_by_key.setdefault(left_key, []).append(total)
                scores_by_key.setdefault(right_key, []).append(total)

        confirmed: list[tuple[str, str]] = []
        active_pairs: set[tuple[str, str]] = set()
        for candidates in candidates_by_pair.values():
            matched_candidates = self._select_candidate_matches(candidates, map_dedup_config)
            for candidate in matched_candidates:
                left_key = str(candidate["left_key"])
                right_key = str(candidate["right_key"])
                pair_key = tuple(sorted((left_key, right_key)))
                active_pairs.add(pair_key)
                left_margin = self._score_margin(scores_by_key.get(left_key, []), float(candidate["score"]))
                right_margin = self._score_margin(scores_by_key.get(right_key, []), float(candidate["score"]))
                margin = min(left_margin, right_margin)
                if margin < map_dedup_config.margin_min:
                    stats.matches_rejected_margin += 1
                    self._map_pair_confirmations.pop(pair_key, None)
                    self._record_map_pair_debug(
                        left_key,
                        right_key,
                        float(candidate["score"]),
                        float(candidate["appearance"]),
                        float(candidate["world"]),
                        float(candidate["timing"]),
                        float(candidate["overlap"]),
                        bool(candidate["left_inside"]),
                        bool(candidate["right_inside"]),
                        float(candidate["world_distance"]),
                        bool(candidate["appearance_available"]),
                        margin,
                        0,
                        False,
                    )
                    continue
                current_confirmation = self._map_pair_confirmations.get(pair_key, 0) + 1
                self._map_pair_confirmations[pair_key] = current_confirmation
                passed = current_confirmation >= map_dedup_config.confirmation_frames
                self._record_map_pair_debug(
                    left_key,
                    right_key,
                    float(candidate["score"]),
                    float(candidate["appearance"]),
                    float(candidate["world"]),
                    float(candidate["timing"]),
                    float(candidate["overlap"]),
                    bool(candidate["left_inside"]),
                    bool(candidate["right_inside"]),
                    float(candidate["world_distance"]),
                    bool(candidate["appearance_available"]),
                    margin,
                    current_confirmation,
                    passed,
                )
                if not passed:
                    continue
                confirmed.append(pair_key)

        for pair_key in list(self._map_pair_confirmations):
            if pair_key not in active_pairs:
                self._map_pair_confirmations.pop(pair_key, None)
        stats.matches_committed = len(confirmed)
        return confirmed, stats

    def _score_pair(
        self,
        timestamp: float,
        left: CameraIdentityTrack,
        right: CameraIdentityTrack,
        overlap_graph: CameraOverlapGraph,
    ) -> tuple[float, float, float, float, float] | None:
        relation = overlap_graph.relation_for(left.camera_id, right.camera_id)
        if relation is None or not relation.is_adjacent:
            return None
        left_point = left.smoothed_ground_anchor_world or left.ground_anchor_world
        right_point = right.smoothed_ground_anchor_world or right.ground_anchor_world
        if left_point is None or right_point is None:
            return None
        world_distance = math.dist(left_point, right_point)
        if world_distance > self.dedup_config.max_distance_m:
            return None
        time_gap = abs(left.last_seen_ts - right.last_seen_ts)
        if time_gap > self.dedup_config.max_time_gap_s:
            return None
        appearance = self._appearance_similarity(left, right)
        world = 1.0 - min(world_distance / max(self.dedup_config.max_distance_m, 1e-6), 1.0)
        timing = 1.0 - min(time_gap / max(self.dedup_config.max_time_gap_s, 1e-6), 1.0)
        geometry = 1.0 if relation.overlap_area_m2 >= self.dedup_config.overlap_area_min_m2 else max(
            0.0,
            1.0 - relation.min_boundary_distance_m / max(self.dedup_config.boundary_gap_m, 1e-6),
        )
        total = (
            self.dedup_config.appearance_weight * appearance
            + self.dedup_config.world_weight * world
            + self.dedup_config.timing_weight * timing
            + self.dedup_config.geometry_weight * geometry
        )
        return total, appearance, world, timing, geometry

    def _refresh_overlap_tracklets(
        self,
        timestamp: float,
        flattened: list[tuple[str, CameraIdentityTrack]],
        overlap_graph: CameraOverlapGraph,
        map_dedup_config: MapDedupConfig,
    ) -> dict[str, OverlapTracklet]:
        overlap_polygons_by_camera: dict[str, list[list[tuple[float, float]]]] = {}
        for relation in overlap_graph.relations.values():
            if not relation.is_adjacent or len(relation.intersection_polygon_world) < 3:
                continue
            overlap_polygons_by_camera.setdefault(relation.camera_a_id, []).append(relation.intersection_polygon_world)
            overlap_polygons_by_camera.setdefault(relation.camera_b_id, []).append(relation.intersection_polygon_world)

        active_tracklets: dict[str, OverlapTracklet] = {}
        for local_key, track in flattened:
            point = track.smoothed_ground_anchor_world or track.ground_anchor_world
            if point is None:
                continue
            overlap_polygons = overlap_polygons_by_camera.get(track.camera_id, [])
            if overlap_polygons:
                near_any_overlap = any(
                    point_inside_or_near_overlap(point, polygon, map_dedup_config.overlap_presence_boundary_buffer_m)
                    for polygon in overlap_polygons
                )
            else:
                near_any_overlap = False
            if not near_any_overlap:
                continue
            tracklet = self._overlap_tracklets.get(local_key)
            if tracklet is None or timestamp - tracklet.end_ts > map_dedup_config.tracklet_keepalive_s:
                tracklet = OverlapTracklet(
                    tracklet_id=local_key,
                    camera_id=track.camera_id,
                    camera_person_id=local_key,
                    start_ts=timestamp,
                    end_ts=timestamp,
                    last_world_point=point,
                )
                self._overlap_tracklets[local_key] = tracklet

            tracklet.end_ts = timestamp
            tracklet.last_world_point = point
            tracklet.smoothed_world_points.append(point)
            tracklet.sample_timestamps.append(timestamp)
            inside_flags = [
                point_in_polygon(point, polygon) for polygon in overlap_polygons
            ]
            inside_value = 1.0 if any(inside_flags) else 0.0
            sample_count = max(len(tracklet.smoothed_world_points), 1)
            previous_inside_total = tracklet.overlap_inside_ratio * max(sample_count - 1, 0)
            tracklet.overlap_inside_ratio = (previous_inside_total + inside_value) / sample_count
            if self._track_has_appearance(track):
                if track.appearance_prototype:
                    tracklet.appearance_samples.append(list(track.appearance_prototype))
                else:
                    tracklet.appearance_samples.extend(
                        [list(embedding) for embedding in track.appearance_memory[-1:]]
                    )
            if len(tracklet.smoothed_world_points) >= 2:
                start_point = tracklet.smoothed_world_points[0]
                dt = max(tracklet.end_ts - tracklet.start_ts, 1e-6)
                tracklet.mean_velocity = (
                    (point[0] - start_point[0]) / dt,
                    (point[1] - start_point[1]) / dt,
                )
            trimmed_samples = [
                (point_item, ts_item)
                for point_item, ts_item in zip(tracklet.smoothed_world_points, tracklet.sample_timestamps)
                if timestamp - ts_item <= map_dedup_config.tracklet_window_s
            ][-64:]
            tracklet.smoothed_world_points = [point_item for point_item, _ts_item in trimmed_samples]
            tracklet.sample_timestamps = [ts_item for _point_item, ts_item in trimmed_samples]
            tracklet.appearance_samples = tracklet.appearance_samples[-self.reid_config.match_topk :]
            if tracklet.sample_timestamps:
                tracklet.start_ts = tracklet.sample_timestamps[0]
            if len(tracklet.smoothed_world_points) >= map_dedup_config.tracklet_min_points:
                active_tracklets[local_key] = tracklet

        for local_key in list(self._overlap_tracklets):
            tracklet = self._overlap_tracklets[local_key]
            if local_key not in active_tracklets and timestamp - tracklet.end_ts > map_dedup_config.tracklet_keepalive_s:
                self._overlap_tracklets.pop(local_key, None)
        return active_tracklets

    def _score_map_tracklet_pair(
        self,
        timestamp: float,
        left: OverlapTracklet,
        right: OverlapTracklet,
        overlap_polygon_world: list[tuple[float, float]],
        map_dedup_config: MapDedupConfig,
    ) -> dict[str, object] | None:
        del timestamp
        left_point = left.last_world_point
        right_point = right.last_world_point
        left_inside = point_in_polygon(left_point, overlap_polygon_world)
        right_inside = point_in_polygon(right_point, overlap_polygon_world)
        left_near = point_inside_or_near_overlap(
            left_point,
            overlap_polygon_world,
            map_dedup_config.overlap_presence_boundary_buffer_m,
        )
        right_near = point_inside_or_near_overlap(
            right_point,
            overlap_polygon_world,
            map_dedup_config.overlap_presence_boundary_buffer_m,
        )
        if not ((left_inside and right_inside) or (left_near and right_near)):
            return {"rejection_reason": "geometry"}

        world_distance = math.dist(left_point, right_point)
        if world_distance > map_dedup_config.max_distance_m:
            return {"rejection_reason": "geometry"}
        time_gap = abs(left.end_ts - right.end_ts)
        if time_gap > map_dedup_config.max_time_gap_s:
            return {"rejection_reason": "time"}

        overlap_membership = 1.0 if left_inside and right_inside else 0.6
        world_alignment = self._tracklet_world_alignment(left, right, map_dedup_config.max_distance_m)
        timing = self._tracklet_timing_score(left, right, map_dedup_config.max_time_gap_s)
        motion = self._tracklet_motion_consistency(left, right)
        appearance, appearance_available = self._tracklet_appearance_similarity(left, right)

        weighted_components = [
            (map_dedup_config.world_alignment_weight, world_alignment),
            (map_dedup_config.overlap_membership_weight, overlap_membership),
            (map_dedup_config.timing_weight, timing),
            (map_dedup_config.motion_weight, motion),
        ]
        if appearance_available:
            weighted_components.append((map_dedup_config.appearance_weight, appearance))
        total_weight = sum(weight for weight, _value in weighted_components)
        if total_weight <= 0.0:
            return None
        total = sum(weight * value for weight, value in weighted_components) / total_weight
        return {
            "rejection_reason": "",
            "score": total,
            "appearance": appearance,
            "world": world_alignment,
            "timing": timing,
            "motion": motion,
            "overlap": overlap_membership,
            "left_inside": left_inside,
            "right_inside": right_inside,
            "world_distance": world_distance,
            "appearance_available": appearance_available,
        }

    def _select_candidate_matches(
        self,
        candidates: list[dict[str, object]],
        map_dedup_config: MapDedupConfig,
    ) -> list[dict[str, object]]:
        left_keys = {str(candidate["left_key"]) for candidate in candidates}
        right_keys = {str(candidate["right_key"]) for candidate in candidates}
        if len(left_keys) <= map_dedup_config.max_exact_assignment_size and len(right_keys) <= map_dedup_config.max_exact_assignment_size:
            return self._exact_match_candidates(candidates)
        claimed_left: set[str] = set()
        claimed_right: set[str] = set()
        selected: list[dict[str, object]] = []
        for candidate in sorted(candidates, key=lambda item: float(item["score"]), reverse=True):
            left_key = str(candidate["left_key"])
            right_key = str(candidate["right_key"])
            if left_key in claimed_left or right_key in claimed_right:
                continue
            selected.append(candidate)
            claimed_left.add(left_key)
            claimed_right.add(right_key)
        return selected

    def _exact_match_candidates(self, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        by_left: dict[str, list[dict[str, object]]] = {}
        for candidate in candidates:
            by_left.setdefault(str(candidate["left_key"]), []).append(candidate)
        ordered_left = sorted(by_left)

        def recurse(index: int, used_right: set[str]) -> tuple[float, list[dict[str, object]]]:
            if index >= len(ordered_left):
                return 0.0, []
            best_score, best_selection = recurse(index + 1, used_right)
            for candidate in by_left[ordered_left[index]]:
                right_key = str(candidate["right_key"])
                if right_key in used_right:
                    continue
                score, selection = recurse(index + 1, used_right | {right_key})
                score += float(candidate["score"])
                if score > best_score:
                    best_score = score
                    best_selection = [candidate] + selection
            return best_score, best_selection

        return recurse(0, set())[1]

    def _build_presence_groups(
        self,
        current_keys: set[str],
        active_tracks: dict[str, CameraIdentityTrack],
        confirmed_pairs: list[tuple[str, str]],
    ) -> list[dict[str, object]]:
        adjacency: dict[str, set[str]] = {}
        for left_key, right_key in confirmed_pairs:
            adjacency.setdefault(left_key, set()).add(right_key)
            adjacency.setdefault(right_key, set()).add(left_key)
        groups: list[dict[str, object]] = []
        visited: set[str] = set()
        for start_key in sorted(adjacency):
            if start_key in visited:
                continue
            stack = [start_key]
            component: set[str] = set()
            while stack:
                local_key = stack.pop()
                if local_key in visited:
                    continue
                visited.add(local_key)
                component.add(local_key)
                stack.extend(adjacency.get(local_key, ()))
            groups.append(self._build_group_payload(component, active_tracks, merged=True))
        for local_key in sorted(current_keys - visited):
            if local_key not in active_tracks:
                continue
            groups.append(self._build_group_payload({local_key}, active_tracks, merged=False))
        return groups

    def _build_group_payload(
        self,
        member_keys: set[str],
        active_tracks: dict[str, CameraIdentityTrack],
        *,
        merged: bool,
    ) -> dict[str, object]:
        members = [(local_key, active_tracks[local_key]) for local_key in sorted(member_keys) if local_key in active_tracks]
        world_points = [
            track.smoothed_ground_anchor_world or track.ground_anchor_world
            for _local_key, track in members
            if (track.smoothed_ground_anchor_world or track.ground_anchor_world) is not None
        ]
        world_point = (
            sum(point[0] for point in world_points) / len(world_points),
            sum(point[1] for point in world_points) / len(world_points),
        )
        confidence = self._presence_confidence(members, merged)
        return {
            "member_keys": sorted(member_keys),
            "members": members,
            "world_point": world_point,
            "confidence": confidence,
            "merged": merged,
        }

    def _update_stable_map_presences(
        self,
        timestamp: float,
        groups: list[dict[str, object]],
        map_dedup_config: MapDedupConfig,
    ) -> None:
        touched_presence_ids: set[str] = set()
        current_local_bindings: dict[str, str] = {}
        for group in groups:
            member_keys = [str(value) for value in group["member_keys"]]
            existing_presence_ids = [
                presence_id
                for presence_id in {self._map_local_to_presence.get(local_key) for local_key in member_keys}
                if presence_id and presence_id in self._stable_map_presences and presence_id not in touched_presence_ids
            ]
            if not existing_presence_ids:
                reassignable_presence_id = self._find_reassignable_presence_id(
                    member_keys,
                    tuple(group["world_point"]),  # type: ignore[arg-type]
                    timestamp,
                    map_dedup_config,
                    touched_presence_ids,
                )
                if reassignable_presence_id is not None:
                    existing_presence_ids = [reassignable_presence_id]
            if existing_presence_ids:
                existing_presence_ids.sort(key=lambda presence_id: self._stable_map_presences[presence_id].first_seen_ts)
                presence_id = existing_presence_ids[0]
                for other_presence_id in existing_presence_ids[1:]:
                    self._merge_presence_states(presence_id, other_presence_id)
            else:
                presence_id = self._create_analytics_id()
            presence = self._stable_map_presences.setdefault(
                presence_id,
                StableMapPresence(
                    presence_id=presence_id,
                    first_seen_ts=timestamp,
                    last_seen_ts=timestamp,
                    hold_until_ts=timestamp + map_dedup_config.presence_hold_s,
                ),
            )
            presence.member_tracklets = list(member_keys)
            presence.world_point = tuple(group["world_point"])  # type: ignore[arg-type]
            presence.last_seen_ts = timestamp
            presence.hold_until_ts = timestamp + map_dedup_config.presence_hold_s
            presence.confirmed = bool(group["merged"]) or presence.confirmed or len(member_keys) == 1
            presence.confidence = float(group["confidence"])
            presence.source_camera_ids = sorted({track.camera_id for _local_key, track in group["members"]})
            presence.source_camera_person_ids = {
                track.camera_id: local_key for local_key, track in group["members"]
            }
            presence.merged_for_counting = bool(group["merged"])
            presence.dedup_mode = "overlap_merged" if presence.merged_for_counting else "local_only"
            touched_presence_ids.add(presence_id)
            for local_key in member_keys:
                current_local_bindings[local_key] = presence_id

        self._map_local_to_presence = current_local_bindings
        for presence_id, presence in list(self._stable_map_presences.items()):
            if presence_id in touched_presence_ids:
                continue
            if timestamp > presence.hold_until_ts:
                self._stable_map_presences.pop(presence_id, None)

    def _find_reassignable_presence_id(
        self,
        member_keys: list[str],
        world_point: tuple[float, float],
        timestamp: float,
        map_dedup_config: MapDedupConfig,
        excluded_presence_ids: set[str],
    ) -> str | None:
        candidates: list[tuple[float, str]] = []
        for presence_id, presence in self._stable_map_presences.items():
            if presence_id in excluded_presence_ids:
                continue
            if timestamp - presence.last_seen_ts > map_dedup_config.presence_reassign_window_s:
                continue
            member_overlap = len(set(member_keys) & set(presence.member_tracklets))
            distance = math.dist(world_point, presence.world_point)
            if member_overlap <= 0 and distance > map_dedup_config.max_distance_m:
                continue
            score = member_overlap * 10.0 - distance
            candidates.append((score, presence_id))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _emit_map_presences(self, timestamp: float) -> list[MapPresence]:
        presences: list[MapPresence] = []
        for presence in self._stable_map_presences.values():
            if timestamp > presence.hold_until_ts:
                continue
            if timestamp - presence.last_seen_ts > self._map_publish_ttl_s:
                continue
            presences.append(
                MapPresence(
                    presence_id=presence.presence_id,
                    world_point=presence.world_point,
                    source_camera_ids=list(presence.source_camera_ids),
                    source_camera_person_ids=dict(presence.source_camera_person_ids),
                    merged_for_counting=presence.merged_for_counting,
                    confidence=presence.confidence,
                    dedup_mode=presence.dedup_mode,
                    first_seen_ts=presence.first_seen_ts,
                    last_seen_ts=presence.last_seen_ts,
                    confirmed=presence.confirmed,
                    hold_until_ts=presence.hold_until_ts,
                    member_tracklets=list(presence.member_tracklets),
                )
            )
        return presences

    def _cleanup_presence_bindings(self, current_keys: set[str]) -> None:
        for local_key in list(self._map_local_to_presence):
            if local_key not in current_keys:
                self._map_local_to_presence.pop(local_key, None)

    def _merge_presence_states(self, keep_id: str, drop_id: str) -> None:
        if keep_id == drop_id:
            return
        keep = self._stable_map_presences.get(keep_id)
        drop = self._stable_map_presences.get(drop_id)
        if keep is None or drop is None:
            return
        keep.first_seen_ts = min(keep.first_seen_ts, drop.first_seen_ts)
        keep.last_seen_ts = max(keep.last_seen_ts, drop.last_seen_ts)
        keep.hold_until_ts = max(keep.hold_until_ts, drop.hold_until_ts)
        keep.member_tracklets = sorted(set(keep.member_tracklets) | set(drop.member_tracklets))
        keep.source_camera_ids = sorted(set(keep.source_camera_ids) | set(drop.source_camera_ids))
        keep.source_camera_person_ids.update(drop.source_camera_person_ids)
        keep.merged_for_counting = keep.merged_for_counting or drop.merged_for_counting
        keep.dedup_mode = "overlap_merged" if keep.merged_for_counting else "local_only"
        self._stable_map_presences.pop(drop_id, None)
        for local_key, presence_id in list(self._map_local_to_presence.items()):
            if presence_id == drop_id:
                self._map_local_to_presence[local_key] = keep_id

    def _tracklet_world_alignment(
        self,
        left: OverlapTracklet,
        right: OverlapTracklet,
        max_distance_m: float,
    ) -> float:
        centroid_left = (
            sum(point[0] for point in left.smoothed_world_points) / len(left.smoothed_world_points),
            sum(point[1] for point in left.smoothed_world_points) / len(left.smoothed_world_points),
        )
        centroid_right = (
            sum(point[0] for point in right.smoothed_world_points) / len(right.smoothed_world_points),
            sum(point[1] for point in right.smoothed_world_points) / len(right.smoothed_world_points),
        )
        centroid_distance = math.dist(centroid_left, centroid_right)
        last_distance = math.dist(left.last_world_point, right.last_world_point)
        blended_distance = 0.6 * centroid_distance + 0.4 * last_distance
        return max(0.0, 1.0 - min(blended_distance / max(max_distance_m, 1e-6), 1.0))

    def _tracklet_timing_score(
        self,
        left: OverlapTracklet,
        right: OverlapTracklet,
        max_time_gap_s: float,
    ) -> float:
        overlap_start = max(left.start_ts, right.start_ts)
        overlap_end = min(left.end_ts, right.end_ts)
        if overlap_end >= overlap_start:
            window = max(left.end_ts - left.start_ts, right.end_ts - right.start_ts, 1e-6)
            overlap_duration = overlap_end - overlap_start
            return min(overlap_duration / window, 1.0) * 0.5 + 0.5
        gap = min(abs(left.end_ts - right.start_ts), abs(right.end_ts - left.start_ts))
        return max(0.0, 1.0 - min(gap / max(max_time_gap_s, 1e-6), 1.0))

    def _tracklet_motion_consistency(self, left: OverlapTracklet, right: OverlapTracklet) -> float:
        left_norm = math.hypot(*left.mean_velocity)
        right_norm = math.hypot(*right.mean_velocity)
        if left_norm <= 1e-6 or right_norm <= 1e-6:
            return 0.5
        cosine = (
            left.mean_velocity[0] * right.mean_velocity[0] + left.mean_velocity[1] * right.mean_velocity[1]
        ) / (left_norm * right_norm)
        cosine = max(-1.0, min(1.0, cosine))
        return (cosine + 1.0) / 2.0

    def _tracklet_appearance_similarity(self, left: OverlapTracklet, right: OverlapTracklet) -> tuple[float, bool]:
        if not left.appearance_samples or not right.appearance_samples:
            return 0.0, False
        scores = sorted(
            (
                cosine_similarity(left_embedding, right_embedding)
                for left_embedding in left.appearance_samples[-self.reid_config.match_topk :]
                for right_embedding in right.appearance_samples[-self.reid_config.match_topk :]
            ),
            reverse=True,
        )
        if not scores:
            return 0.0, False
        count = min(3, len(scores))
        return sum(scores[:count]) / count, True

    def _appearance_similarity(self, left: CameraIdentityTrack, right: CameraIdentityTrack) -> float:
        left_embeddings = list(left.appearance_memory[-self.reid_config.match_topk :])
        right_embeddings = list(right.appearance_memory[-self.reid_config.match_topk :])
        if left.appearance_prototype:
            left_embeddings.append(left.appearance_prototype)
        if right.appearance_prototype:
            right_embeddings.append(right.appearance_prototype)
        if not left_embeddings or not right_embeddings:
            return 0.0
        scores = sorted(
            (
                cosine_similarity(left_embedding, right_embedding)
                for left_embedding in left_embeddings
                for right_embedding in right_embeddings
            ),
            reverse=True,
        )[: max(1, self.reid_config.match_topk)]
        if self.reid_config.match_reduce == "max_topk":
            return scores[0]
        count = min(3, len(scores))
        return sum(scores[:count]) / count

    @staticmethod
    def _score_margin(scores: list[float], best_score: float) -> float:
        ordered = sorted(scores, reverse=True)
        if len(ordered) < 2:
            return 1.0
        if ordered[0] != best_score:
            return max(best_score - ordered[0], 0.0)
        return max(best_score - ordered[1], 0.0)

    def _record_pair_debug(
        self,
        left_key: str,
        right_key: str,
        score: float,
        appearance: float,
        world: float,
        timing: float,
        geometry: float,
        margin: float,
        passed: bool,
    ) -> None:
        for local_key in (left_key, right_key):
            camera_id, person_token = local_key.split(":", 1)
            tracker_track_id = int(person_token.removeprefix("P")) if person_token.startswith("P") else -1
            self._debug_records.append(
                IdentityDebugRecord(
                    camera_id=camera_id,
                    tracker_track_id=tracker_track_id,
                    camera_person_id=local_key,
                    reason="overlap_dedup_candidate",
                    score=score,
                    stage="overlap_dedup",
                    passed_threshold=passed,
                    appearance_score=appearance,
                    world_score=world,
                    timing_score=timing,
                    transition_score=geometry,
                    second_best_margin=margin,
                )
            )

    def _record_map_pair_debug(
        self,
        left_key: str,
        right_key: str,
        score: float,
        appearance: float,
        world: float,
        timing: float,
        overlap_membership: float,
        left_inside: bool,
        right_inside: bool,
        world_distance_m: float,
        appearance_available: bool,
        margin: float,
        confirmation_progress: int,
        passed: bool,
    ) -> None:
        for local_key in (left_key, right_key):
            camera_id, person_token = local_key.split(":", 1)
            tracker_track_id = int(person_token.removeprefix("P")) if person_token.startswith("P") else -1
            self._debug_records.append(
                IdentityDebugRecord(
                    camera_id=camera_id,
                    tracker_track_id=tracker_track_id,
                    camera_person_id=local_key,
                    reason="map_presence_candidate",
                    score=score,
                    stage="map_overlap_dedup",
                    passed_threshold=passed,
                    appearance_score=appearance if appearance_available else None,
                    world_score=world,
                    timing_score=timing,
                    transition_score=overlap_membership,
                    second_best_margin=margin,
                    overlap_inside_left=left_inside,
                    overlap_inside_right=right_inside,
                    world_distance_m=world_distance_m,
                    appearance_available=appearance_available,
                    normalized_total_score=score,
                    confirmation_progress=confirmation_progress,
                )
            )

    def _attach(
        self,
        analytics_id: str,
        local_key: str,
        track: CameraIdentityTrack,
        timestamp: float,
    ) -> None:
        analytics_track = self._tracks.setdefault(
            analytics_id,
            AnalyticsTrack(
                analytics_track_id=analytics_id,
                first_seen_ts=timestamp,
                last_seen_ts=timestamp,
            ),
        )
        analytics_track.active = True
        analytics_track.last_seen_ts = timestamp
        analytics_track.primary_camera_id = track.camera_id
        analytics_track.active_camera_person_ids[track.camera_id] = local_key
        if track.camera_id not in analytics_track.source_camera_ids:
            analytics_track.source_camera_ids.append(track.camera_id)
        analytics_track.ground_anchor_world = track.ground_anchor_world
        analytics_track.smoothed_ground_anchor_world = track.smoothed_ground_anchor_world or track.ground_anchor_world
        analytics_track.appearance_descriptor = list(track.appearance_prototype)
        analytics_track.support_count = len(analytics_track.active_camera_person_ids)
        analytics_track.dedup_mode = (
            "overlap_merged" if len(analytics_track.active_camera_person_ids) > 1 else "local_only"
        )

    def _resolve_map_group_assignments(
        self,
        confirmed_pairs: list[tuple[str, str]],
    ) -> dict[str, str]:
        if not confirmed_pairs:
            return {}
        adjacency: dict[str, set[str]] = {}
        for left_key, right_key in confirmed_pairs:
            adjacency.setdefault(left_key, set()).add(right_key)
            adjacency.setdefault(right_key, set()).add(left_key)

        resolved_ids: dict[str, str] = {}
        visited: set[str] = set()
        for start_key in sorted(adjacency):
            if start_key in visited:
                continue
            stack = [start_key]
            component: set[str] = set()
            while stack:
                local_key = stack.pop()
                if local_key in visited:
                    continue
                visited.add(local_key)
                component.add(local_key)
                stack.extend(adjacency.get(local_key, ()))

            existing_ids = sorted(
                {
                    presence_id
                    for local_key in component
                    if (presence_id := self._map_local_to_presence.get(local_key)) is not None
                }
            )
            presence_id = existing_ids[0] if existing_ids else self._create_analytics_id()
            for local_key in sorted(component):
                resolved_ids[local_key] = presence_id
                self._map_local_to_presence[local_key] = presence_id
        return resolved_ids

    def _presence_confidence(
        self,
        members: list[tuple[str, CameraIdentityTrack]],
        merged_for_counting: bool,
    ) -> float:
        if not members:
            return 0.0
        confidence_values = [track.confidence for _local_key, track in members if track.confidence > 0.0]
        base_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.5
        if merged_for_counting:
            return min(base_confidence + 0.15, 1.0)
        return base_confidence

    @staticmethod
    def _track_has_appearance(track: CameraIdentityTrack) -> bool:
        return bool(track.appearance_prototype or track.appearance_memory)

    def _merge_tracks(self, keep_id: str, drop_id: str) -> None:
        keep = self._tracks.get(keep_id)
        drop = self._tracks.get(drop_id)
        if keep is None or drop is None or keep_id == drop_id:
            return
        for camera_id, local_key in drop.active_camera_person_ids.items():
            keep.active_camera_person_ids[camera_id] = local_key
            self._local_to_analytics[local_key] = keep_id
        for camera_id in drop.source_camera_ids:
            if camera_id not in keep.source_camera_ids:
                keep.source_camera_ids.append(camera_id)
        keep.first_seen_ts = min(keep.first_seen_ts, drop.first_seen_ts)
        keep.last_seen_ts = max(keep.last_seen_ts, drop.last_seen_ts)
        keep.support_count = len(keep.active_camera_person_ids)
        keep.dedup_mode = "overlap_merged"
        self._tracks.pop(drop_id, None)

    def _expire_stale_tracks(self, timestamp: float) -> None:
        retention_s = max(self.dedup_config.max_time_gap_s, 1.0) * 4.0
        for analytics_id, track in list(self._tracks.items()):
            if track.active:
                continue
            if timestamp - track.last_seen_ts > retention_s:
                self._tracks.pop(analytics_id, None)

    @staticmethod
    def _copy_track(track: AnalyticsTrack) -> AnalyticsTrack:
        return AnalyticsTrack(
            analytics_track_id=track.analytics_track_id,
            source_camera_ids=list(track.source_camera_ids),
            active_camera_person_ids=dict(track.active_camera_person_ids),
            first_seen_ts=track.first_seen_ts,
            last_seen_ts=track.last_seen_ts,
            active=track.active,
            dedup_mode=track.dedup_mode,
            ground_anchor_world=track.ground_anchor_world,
            smoothed_ground_anchor_world=track.smoothed_ground_anchor_world,
            primary_camera_id=track.primary_camera_id,
            appearance_descriptor=list(track.appearance_descriptor),
            support_count=track.support_count,
        )

    def _create_analytics_id(self) -> str:
        analytics_id = f"A{self._next_analytics_id:06d}"
        self._next_analytics_id += 1
        return analytics_id
