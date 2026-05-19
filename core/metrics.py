from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from core.models import (
    AnalyticsEvent,
    AnalyticsSnapshot,
    AnalyticsTrack,
    BoothVisitSession,
    MapPresence,
    VenueMapConfig,
    ZoneMetrics,
)
from core.zones import find_zone


@dataclass(slots=True)
class _TrackAnalyticsState:
    active_zone_id: str | None = None
    zone_entered_at: float | None = None
    candidate_zone_id: str | None = None
    candidate_zone_since: float | None = None
    zone_exit_candidate_since: float | None = None
    active_visit_id: str | None = None


class AnalyticsEngine:
    def __init__(
        self,
        venue_map: VenueMapConfig,
        zone_entry_min_duration_s: float = 0.35,
        zone_exit_grace_s: float = 0.75,
    ) -> None:
        self.venue_map = venue_map
        self.zone_entry_min_duration_s = zone_entry_min_duration_s
        self.zone_exit_grace_s = zone_exit_grace_s
        self._states: dict[str, _TrackAnalyticsState] = {}
        self._next_visit_id = 1
        self._finalized_sessions: list[BoothVisitSession] = []
        self._peak_occupancy_by_zone: dict[str, int] = {}

    def reset(self) -> None:
        self._states.clear()
        self._next_visit_id = 1
        self._finalized_sessions.clear()
        self._peak_occupancy_by_zone.clear()

    def update(
        self,
        timestamp: float,
        analytics_tracks: dict[str, AnalyticsTrack] | list[MapPresence],
    ) -> AnalyticsSnapshot:
        active_presences = self._normalize_active_presences(analytics_tracks)
        recent_events: list[AnalyticsEvent] = []
        finalized_recent: list[BoothVisitSession] = []
        active_ids = {presence.presence_id for presence in active_presences}

        for presence in active_presences:
            track_id = presence.presence_id
            state = self._states.setdefault(track_id, _TrackAnalyticsState())
            zone = find_zone(presence.world_point, self.venue_map.zones)
            zone_id = zone.zone_id if zone else None

            if state.active_zone_id == zone_id:
                state.candidate_zone_id = None
                state.candidate_zone_since = None
                state.zone_exit_candidate_since = None
                continue

            if state.active_zone_id is not None and zone_id is None:
                if state.zone_exit_candidate_since is None:
                    state.zone_exit_candidate_since = timestamp
                elif timestamp - state.zone_exit_candidate_since >= self.zone_exit_grace_s:
                    finalized = self._close_active_visit(track_id, state, presence, timestamp)
                    if finalized is not None:
                        finalized_recent.append(finalized)
                        recent_events.append(
                            AnalyticsEvent(
                                event_type="booth_visit_finished",
                                analytics_track_id=track_id,
                                zone_id=finalized.zone_id,
                                timestamp=timestamp,
                                payload={
                                    "visit_id": finalized.visit_id,
                                    "dwell_s": finalized.dwell_s,
                                    "source_camera_ids": finalized.source_camera_ids,
                                },
                            )
                        )
                continue

            if state.candidate_zone_id == zone_id and state.candidate_zone_since is not None:
                if timestamp - state.candidate_zone_since >= self.zone_entry_min_duration_s:
                    if state.active_zone_id is not None and state.active_zone_id != zone_id:
                        finalized = self._close_active_visit(track_id, state, presence, timestamp)
                        if finalized is not None:
                            finalized_recent.append(finalized)
                    self._open_visit(track_id, state, presence, zone_id, timestamp)
                    recent_events.append(
                        AnalyticsEvent(
                            event_type="booth_visit_started",
                            analytics_track_id=track_id,
                            zone_id=zone_id,
                            timestamp=timestamp,
                            payload={"visit_id": state.active_visit_id},
                        )
                    )
                continue

            state.candidate_zone_id = zone_id
            state.candidate_zone_since = timestamp
            state.zone_exit_candidate_since = None

        for track_id, state in list(self._states.items()):
            if track_id in active_ids:
                continue
            presence = next(
                (item for item in active_presences if item.presence_id == track_id),
                None,
            )
            if presence is None:
                presence = MapPresence(
                    presence_id=track_id,
                    world_point=(0.0, 0.0),
                )
            finalized = self._close_active_visit(track_id, state, presence, timestamp)
            if finalized is not None:
                finalized_recent.append(finalized)
                recent_events.append(
                    AnalyticsEvent(
                        event_type="booth_visit_finished",
                        analytics_track_id=track_id,
                        zone_id=finalized.zone_id,
                        timestamp=finalized.left_at or timestamp,
                        payload={
                            "visit_id": finalized.visit_id,
                            "dwell_s": finalized.dwell_s,
                            "source_camera_ids": finalized.source_camera_ids,
                        },
                    )
                )

        self._finalized_sessions.extend(finalized_recent)
        return self._build_snapshot(timestamp, active_presences, recent_events, finalized_recent)

    def _open_visit(
        self,
        track_id: str,
        state: _TrackAnalyticsState,
        track: MapPresence,
        zone_id: str | None,
        timestamp: float,
    ) -> None:
        if zone_id is None:
            return
        state.active_zone_id = zone_id
        state.zone_entered_at = timestamp
        state.candidate_zone_id = None
        state.candidate_zone_since = None
        state.zone_exit_candidate_since = None
        state.active_visit_id = f"V{self._next_visit_id:07d}"
        self._next_visit_id += 1

    def _close_active_visit(
        self,
        track_id: str,
        state: _TrackAnalyticsState,
        track: MapPresence,
        timestamp: float,
    ) -> BoothVisitSession | None:
        if state.active_zone_id is None or state.zone_entered_at is None or state.active_visit_id is None:
            state.active_zone_id = None
            state.zone_entered_at = None
            state.active_visit_id = None
            state.zone_exit_candidate_since = None
            return None
        dwell_s = max(timestamp - state.zone_entered_at, 0.0)
        session = BoothVisitSession(
            visit_id=state.active_visit_id,
            zone_id=state.active_zone_id,
            analytics_track_id=track_id,
            entered_at=state.zone_entered_at,
            left_at=timestamp,
            dwell_s=dwell_s,
            source_camera_ids=list(track.source_camera_ids),
            dedup_mode=track.dedup_mode,
        )
        state.active_zone_id = None
        state.zone_entered_at = None
        state.active_visit_id = None
        state.zone_exit_candidate_since = None
        state.candidate_zone_id = None
        state.candidate_zone_since = None
        return session

    def _build_snapshot(
        self,
        timestamp: float,
        active_presences: list[MapPresence],
        recent_events: list[AnalyticsEvent],
        finalized_recent: list[BoothVisitSession],
    ) -> AnalyticsSnapshot:
        zone_metrics: dict[str, ZoneMetrics] = {
            zone.zone_id: ZoneMetrics(
                zone_id=zone.zone_id,
                zone_name=zone.name,
                zone_kind=zone.kind,
            )
            for zone in self.venue_map.zones
        }
        active_zone_counts: dict[str, int] = {}
        unique_zone_entries: dict[str, int] = {}
        dwell_times: dict[str, float] = {}
        avg_dwell_times: dict[str, float] = {}
        median_dwell_times: dict[str, float] = {}
        peak_occupancy_by_zone = dict(self._peak_occupancy_by_zone)
        dwell_values_by_zone: dict[str, list[float]] = {}

        for state in self._states.values():
            if state.active_zone_id is None:
                continue
            active_zone_counts[state.active_zone_id] = active_zone_counts.get(state.active_zone_id, 0) + 1

        for zone_id, occupancy in active_zone_counts.items():
            peak_occupancy_by_zone[zone_id] = max(peak_occupancy_by_zone.get(zone_id, 0), occupancy)
        self._peak_occupancy_by_zone = peak_occupancy_by_zone

        for session in self._finalized_sessions:
            unique_zone_entries[session.zone_id] = unique_zone_entries.get(session.zone_id, 0) + 1
            dwell_times[session.zone_id] = dwell_times.get(session.zone_id, 0.0) + session.dwell_s
            dwell_values_by_zone.setdefault(session.zone_id, []).append(session.dwell_s)

        for track_id, state in self._states.items():
            if state.active_zone_id is None or state.zone_entered_at is None:
                continue
            dwell_times[state.active_zone_id] = dwell_times.get(state.active_zone_id, 0.0) + max(
                timestamp - state.zone_entered_at,
                0.0,
            )

        for zone_id, metrics in zone_metrics.items():
            metrics.current_occupancy = active_zone_counts.get(zone_id, 0)
            metrics.unique_visits = unique_zone_entries.get(zone_id, 0)
            metrics.total_dwell_s = dwell_times.get(zone_id, 0.0)
            if metrics.unique_visits > 0:
                metrics.avg_dwell_s = metrics.total_dwell_s / metrics.unique_visits
                values = dwell_values_by_zone.get(zone_id, [])
                if values:
                    metrics.median_dwell_s = float(median(values))
            metrics.peak_occupancy = peak_occupancy_by_zone.get(zone_id, metrics.current_occupancy)
            if timestamp > 0.0:
                metrics.booth_utilization_ratio = min(metrics.total_dwell_s / max(timestamp, 1e-6), 1.0)
            avg_dwell_times[zone_id] = metrics.avg_dwell_s
            median_dwell_times[zone_id] = metrics.median_dwell_s

        return AnalyticsSnapshot(
            timestamp=timestamp,
            active_zone_counts=active_zone_counts,
            unique_zone_entries=unique_zone_entries,
            dwell_times=dwell_times,
            active_analytics_tracks={},
            active_map_presences=list(active_presences),
            recent_events=recent_events,
            avg_dwell_times=avg_dwell_times,
            median_dwell_times=median_dwell_times,
            peak_occupancy_by_zone=peak_occupancy_by_zone,
            finalized_visit_sessions_recent=finalized_recent,
            total_entries=sum(unique_zone_entries.values()),
            total_current_occupancy=sum(active_zone_counts.values()),
            zone_metrics=zone_metrics,
        )

    @staticmethod
    def _normalize_active_presences(
        analytics_tracks: dict[str, AnalyticsTrack] | list[MapPresence],
    ) -> list[MapPresence]:
        if isinstance(analytics_tracks, list):
            return list(analytics_tracks)
        presences: list[MapPresence] = []
        for track_id, track in analytics_tracks.items():
            if not track.active:
                continue
            point = track.smoothed_ground_anchor_world or track.ground_anchor_world
            if point is None:
                continue
            presences.append(
                MapPresence(
                    presence_id=track_id,
                    world_point=point,
                    source_camera_ids=list(track.source_camera_ids),
                    source_camera_person_ids=dict(track.active_camera_person_ids),
                    merged_for_counting=len(track.active_camera_person_ids) > 1,
                    confidence=1.0,
                    dedup_mode=track.dedup_mode,
                )
            )
        return presences
