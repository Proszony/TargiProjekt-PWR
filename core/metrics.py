from __future__ import annotations

from dataclasses import dataclass, field

from core.models import AnalyticsEvent, AnalyticsSnapshot, GlobalTrack, VenueMapConfig, ZoneMetrics
from core.zones import find_zone


@dataclass(slots=True)
class _TrackAnalyticsState:
    active_zone_id: str | None = None
    zone_entered_at: float | None = None
    candidate_zone_id: str | None = None
    candidate_zone_since: float | None = None
    visited_zone_ids: list[str] = field(default_factory=list)
    last_left_at: dict[str, float] = field(default_factory=dict)
    last_seen_ts: float = 0.0


class AnalyticsEngine:
    def __init__(
        self,
        venue_map: VenueMapConfig,
        zone_entry_min_duration_s: float = 1.0,
        return_threshold_s: float = 30.0,
    ) -> None:
        self.venue_map = venue_map
        self.zone_entry_min_duration_s = zone_entry_min_duration_s
        self.return_threshold_s = return_threshold_s
        self._states: dict[str, _TrackAnalyticsState] = {}

    def reset(self) -> None:
        self._states.clear()

    def update(self, timestamp: float, global_tracks: dict[str, GlobalTrack]) -> AnalyticsSnapshot:
        events: list[AnalyticsEvent] = []
        active_ids = {track_id for track_id, track in global_tracks.items() if track.active}

        for global_id, track in global_tracks.items():
            if not track.active:
                continue
            state = self._states.setdefault(global_id, _TrackAnalyticsState())
            current_point = track.smoothed_ground_anchor_world or track.ground_anchor_world
            current_zone = find_zone(current_point, self.venue_map.zones)
            current_zone_id = current_zone.zone_id if current_zone else None
            state.last_seen_ts = timestamp

            if current_zone_id == state.active_zone_id:
                state.candidate_zone_id = None
                state.candidate_zone_since = None
            elif state.candidate_zone_since is not None and current_zone_id == state.candidate_zone_id:
                if state.candidate_zone_since is not None and (
                    timestamp - state.candidate_zone_since >= self.zone_entry_min_duration_s
                ):
                    events.extend(self._switch_zone(track, state, current_zone_id, timestamp))
            else:
                state.candidate_zone_id = current_zone_id
                state.candidate_zone_since = timestamp

            track.current_zone_id = state.active_zone_id

        for global_id in list(self._states):
            if global_id in active_ids:
                continue
            track = global_tracks.get(global_id)
            if track is None:
                continue
            state = self._states[global_id]
            if state.active_zone_id is not None and state.zone_entered_at is not None:
                dwell = max(track.last_seen_ts - state.zone_entered_at, 0.0)
                track.dwell_times[state.active_zone_id] = track.dwell_times.get(state.active_zone_id, 0.0) + dwell
                events.append(
                    AnalyticsEvent(
                        event_type="person_left_zone",
                        global_track_id=global_id,
                        zone_id=state.active_zone_id,
                        timestamp=track.last_seen_ts,
                    )
                )
                state.last_left_at[state.active_zone_id] = track.last_seen_ts
                state.active_zone_id = None
                state.zone_entered_at = None

        return self._build_snapshot(timestamp, global_tracks, events)

    def _switch_zone(
        self,
        track: GlobalTrack,
        state: _TrackAnalyticsState,
        new_zone_id: str | None,
        timestamp: float,
    ) -> list[AnalyticsEvent]:
        events: list[AnalyticsEvent] = []
        old_zone_id = state.active_zone_id

        if old_zone_id is not None and state.zone_entered_at is not None:
            dwell = max(timestamp - state.zone_entered_at, 0.0)
            track.dwell_times[old_zone_id] = track.dwell_times.get(old_zone_id, 0.0) + dwell
            state.last_left_at[old_zone_id] = timestamp
            events.append(
                AnalyticsEvent(
                    event_type="person_left_zone",
                    global_track_id=track.global_track_id,
                    zone_id=old_zone_id,
                    timestamp=timestamp,
                )
            )

        state.active_zone_id = new_zone_id
        state.zone_entered_at = timestamp if new_zone_id is not None else None
        state.candidate_zone_id = None
        state.candidate_zone_since = None

        if new_zone_id is None:
            return events

        track.zone_visits[new_zone_id] = track.zone_visits.get(new_zone_id, 0) + 1
        if new_zone_id in state.visited_zone_ids:
            last_left_at = state.last_left_at.get(new_zone_id)
            if last_left_at is not None and timestamp - last_left_at >= self.return_threshold_s:
                track.return_count += 1
                track.return_counts_by_zone[new_zone_id] = track.return_counts_by_zone.get(new_zone_id, 0) + 1
                events.append(
                    AnalyticsEvent(
                        event_type="person_returned_to_zone",
                        global_track_id=track.global_track_id,
                        zone_id=new_zone_id,
                        timestamp=timestamp,
                    )
                )
        else:
            state.visited_zone_ids.append(new_zone_id)

        events.append(
            AnalyticsEvent(
                event_type="person_entered_zone",
                global_track_id=track.global_track_id,
                zone_id=new_zone_id,
                timestamp=timestamp,
            )
        )
        return events

    def _build_snapshot(
        self,
        timestamp: float,
        global_tracks: dict[str, GlobalTrack],
        events: list[AnalyticsEvent],
    ) -> AnalyticsSnapshot:
        active_zone_counts: dict[str, int] = {}
        unique_zone_entries: dict[str, int] = {}
        dwell_times: dict[str, float] = {}
        return_counts: dict[str, int] = {}
        avg_dwell_times: dict[str, float] = {}
        zone_metrics: dict[str, ZoneMetrics] = {
            zone.zone_id: ZoneMetrics(
                zone_id=zone.zone_id,
                zone_name=zone.name,
                zone_kind=zone.kind,
            )
            for zone in self.venue_map.zones
        }

        for track in global_tracks.values():
            if not track.active:
                continue
            if track.current_zone_id is not None:
                active_zone_counts[track.current_zone_id] = active_zone_counts.get(track.current_zone_id, 0) + 1

        for track in global_tracks.values():
            for zone_id, visits in track.zone_visits.items():
                unique_zone_entries[zone_id] = unique_zone_entries.get(zone_id, 0) + min(visits, 1)
            for zone_id, dwell in track.dwell_times.items():
                dwell_times[zone_id] = dwell_times.get(zone_id, 0.0) + dwell
            for zone_id, count in track.return_counts_by_zone.items():
                return_counts[zone_id] = return_counts.get(zone_id, 0) + count

        for zone_id, metrics in zone_metrics.items():
            metrics.current_occupancy = active_zone_counts.get(zone_id, 0)
            metrics.unique_entries = unique_zone_entries.get(zone_id, 0)
            metrics.total_dwell_s = dwell_times.get(zone_id, 0.0)
            metrics.return_count = return_counts.get(zone_id, 0)
            if metrics.unique_entries > 0:
                metrics.avg_dwell_s = metrics.total_dwell_s / metrics.unique_entries
            avg_dwell_times[zone_id] = metrics.avg_dwell_s

        total_entries = sum(unique_zone_entries.values())
        session_total_returns = sum(return_counts.values())

        return AnalyticsSnapshot(
            timestamp=timestamp,
            active_zone_counts=active_zone_counts,
            unique_zone_entries=unique_zone_entries,
            dwell_times=dwell_times,
            return_counts=return_counts,
            active_global_tracks={track_id: track for track_id, track in global_tracks.items() if track.active},
            recent_events=events,
            avg_dwell_times=avg_dwell_times,
            total_entries=total_entries,
            session_total_returns=session_total_returns,
            zone_metrics=zone_metrics,
        )
