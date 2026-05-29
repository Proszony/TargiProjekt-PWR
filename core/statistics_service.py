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
        self.repository.begin_active_write_session()
        try:
            self.current_session_id = self.repository.start_session(
                started_at=started_at,
                source_type=source_type,
                source_label=source_label,
                camera_id=camera_id,
            )
        except Exception:
            self.repository.end_active_write_session()
            raise
        return self.current_session_id

    def finish_session(self, ended_at: float) -> None:
        if self.current_session_id is None:
            return
        try:
            self.repository.finish_session(self.current_session_id, ended_at)
        finally:
            self.current_session_id = None
            self.repository.end_active_write_session()

    def finish_session_with_heatmap(self, ended_at: float, heatmap_snapshot: HeatmapSnapshot | None) -> None:
        if self.current_session_id is None:
            return
        try:
            if heatmap_snapshot is not None:
                self.repository.record_session_heatmap(self.current_session_id, heatmap_snapshot)
            self.repository.finish_session(self.current_session_id, ended_at)
        finally:
            self.current_session_id = None
            self.repository.end_active_write_session()

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
