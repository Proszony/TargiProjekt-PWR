from __future__ import annotations

from pathlib import Path

from core.models import AnalyticsSnapshot, GlobalTrack, VenueMapConfig
from core.statistics_repository import StatisticsRepository


class StatisticsService:
    def __init__(self, project_root: Path) -> None:
        self.repository = StatisticsRepository(project_root / "data" / "fair_monitor_stats.sqlite")
        self.current_session_id: int | None = None

    def start_session(
        self,
        started_at: float,
        source_type: str,
        source_label: str,
        camera_id: str,
    ) -> int:
        self.current_session_id = self.repository.start_session(
            started_at=started_at,
            source_type=source_type,
            source_label=source_label,
            camera_id=camera_id,
        )
        return self.current_session_id

    def finish_session(self, ended_at: float) -> None:
        if self.current_session_id is None:
            return
        self.repository.finish_session(self.current_session_id, ended_at)
        self.current_session_id = None

    def record_snapshot(
        self,
        snapshot: AnalyticsSnapshot,
        venue_map: VenueMapConfig,
        global_tracks: dict[str, GlobalTrack] | None = None,
    ) -> None:
        if self.current_session_id is None:
            return
        self.repository.record_events(self.current_session_id, snapshot.recent_events)
        self.repository.record_snapshot(self.current_session_id, snapshot, venue_map)
        self.repository.upsert_track_lifecycle(
            self.current_session_id,
            global_tracks or snapshot.active_global_tracks,
        )
