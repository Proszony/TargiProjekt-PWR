from __future__ import annotations

from pathlib import Path

from core.models import AnalyticsSnapshot, HeatmapSnapshot, VenueMapConfig
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

    def finish_session_with_heatmap(self, ended_at: float, heatmap_snapshot: HeatmapSnapshot | None) -> None:
        if self.current_session_id is None:
            return
        if heatmap_snapshot is not None:
            self.repository.record_session_heatmap(self.current_session_id, heatmap_snapshot)
        self.repository.finish_session(self.current_session_id, ended_at)
        self.current_session_id = None

    def record_snapshot(
        self,
        snapshot: AnalyticsSnapshot,
        venue_map: VenueMapConfig,
    ) -> None:
        if self.current_session_id is None:
            return
        self.repository.record_visit_sessions(
            self.current_session_id,
            snapshot.finalized_visit_sessions_recent,
        )
        self.repository.record_snapshot(self.current_session_id, snapshot, venue_map)
