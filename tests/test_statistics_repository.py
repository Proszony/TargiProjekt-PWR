import tempfile
import unittest
from pathlib import Path
import sqlite3

from core.models import (
    AnalyticsSnapshot,
    BoothVisitSession,
    HeatmapCell,
    HeatmapSnapshot,
    VenueMapConfig,
    WorldViewport,
    ZoneDefinition,
    ZoneMetrics,
)
from core.statistics_repository import StatisticsRepository


class StatisticsRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "stats.sqlite"
        self.repository = StatisticsRepository(self.database_path)
        self.venue = VenueMapConfig(
            zones=[
                ZoneDefinition(
                    zone_id="booth-a",
                    name="Booth A",
                    kind="booth",
                    polygon_world=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
                )
            ],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_session_snapshot_and_visit_sessions_persist(self) -> None:
        session_id = self.repository.start_session(10.0, "file", "demo.mp4", "camera-1,camera-2")
        snapshot = AnalyticsSnapshot(
            timestamp=11.0,
            finalized_visit_sessions_recent=[
                BoothVisitSession(
                    visit_id="V0000001",
                    zone_id="booth-a",
                    analytics_track_id="A000001",
                    entered_at=10.0,
                    left_at=11.0,
                    dwell_s=1.0,
                    source_camera_ids=["camera-1"],
                    dedup_mode="local_only",
                )
            ],
            zone_metrics={
                "booth-a": ZoneMetrics(
                    zone_id="booth-a",
                    zone_name="Booth A",
                    zone_kind="booth",
                    current_occupancy=2,
                    unique_visits=3,
                    total_dwell_s=12.5,
                    avg_dwell_s=4.2,
                    median_dwell_s=3.8,
                    peak_occupancy=4,
                )
            },
        )
        self.repository.record_visit_sessions(session_id, snapshot.finalized_visit_sessions_recent)
        self.repository.record_snapshot(session_id, snapshot, self.venue)
        self.repository.finish_session(session_id, 12.0)

        metrics = self.repository.load_session_zone_metrics(session_id)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["zone_name"], "Booth A")
        self.assertEqual(metrics[0]["current_occupancy"], 2)
        self.assertEqual(metrics[0]["peak_occupancy"], 4)

        visits = self.repository.load_booth_visit_sessions(session_id)
        self.assertEqual(len(visits), 1)
        self.assertEqual(visits[0]["visit_id"], "V0000001")
        self.assertEqual(visits[0]["dedup_mode"], "local_only")

    def test_active_write_session_reuses_one_connection_for_writes(self) -> None:
        connect_calls = 0
        original_connect = self.repository._connect

        def counted_connect() -> sqlite3.Connection:
            nonlocal connect_calls
            connect_calls += 1
            return original_connect()

        self.repository._connect = counted_connect  # type: ignore[method-assign]
        self.repository.begin_active_write_session()
        try:
            session_id = self.repository.start_session(10.0, "file", "demo.mp4", "camera-1")
            snapshot = AnalyticsSnapshot(
                timestamp=11.0,
                zone_metrics={
                    "booth-a": ZoneMetrics(
                        zone_id="booth-a",
                        zone_name="Booth A",
                        zone_kind="booth",
                        current_occupancy=1,
                        unique_visits=1,
                        total_dwell_s=2.0,
                        avg_dwell_s=2.0,
                        median_dwell_s=2.0,
                        peak_occupancy=1,
                    )
                },
            )
            self.repository.record_snapshot(session_id, snapshot, self.venue)
            self.repository.finish_session(session_id, 12.0)
        finally:
            self.repository.end_active_write_session()

        self.assertEqual(connect_calls, 1)

    def test_legacy_zone_snapshot_schema_is_repaired(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.executescript(
                """
                DROP TABLE IF EXISTS zone_snapshots;
                CREATE TABLE zone_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    ts REAL NOT NULL,
                    zone_id TEXT NOT NULL,
                    zone_name TEXT NOT NULL,
                    zone_kind TEXT NOT NULL,
                    occupancy INTEGER NOT NULL,
                    unique_entries INTEGER NOT NULL,
                    total_dwell_s REAL NOT NULL,
                    avg_dwell_s REAL NOT NULL
                );
                """
            )
        repaired = StatisticsRepository(self.database_path)
        with sqlite3.connect(self.database_path) as connection:
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(zone_snapshots)").fetchall()
            }
        self.assertIn("unique_visits", columns)
        self.assertIn("median_dwell_s", columns)
        self.assertIn("peak_occupancy", columns)
        self.assertNotIn("unique_entries", columns)
        self.assertIsInstance(repaired, StatisticsRepository)

    def test_session_heatmap_persists_and_loads(self) -> None:
        session_id = self.repository.start_session(10.0, "file", "demo.mp4", "camera-1")
        heatmap = HeatmapSnapshot(
            timestamp=12.0,
            viewport=WorldViewport(min_x=0.0, min_y=0.0, max_x=4.0, max_y=2.0),
            columns=8,
            rows=4,
            max_dwell_s=3.0,
            total_dwell_s=5.0,
            cells=[
                HeatmapCell(x_index=2, y_index=1, dwell_s=2.0),
                HeatmapCell(x_index=3, y_index=1, dwell_s=3.0),
            ],
        )

        self.repository.record_session_heatmap(session_id, heatmap)
        loaded = self.repository.load_session_heatmap(session_id)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.columns, 8)
        self.assertEqual(loaded.rows, 4)
        self.assertAlmostEqual(loaded.total_dwell_s, 5.0)
        self.assertEqual(len(loaded.cells), 2)
        self.assertEqual(loaded.cells[1].x_index, 3)

    def test_legacy_database_adds_session_heatmap_table(self) -> None:
        repaired = StatisticsRepository(self.database_path)
        with sqlite3.connect(self.database_path) as connection:
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(session_heatmaps)").fetchall()
            }
        self.assertIn("session_id", columns)
        self.assertIn("cells_json", columns)
        self.assertIsInstance(repaired, StatisticsRepository)


if __name__ == "__main__":
    unittest.main()
