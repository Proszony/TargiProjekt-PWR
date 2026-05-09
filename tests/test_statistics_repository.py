import tempfile
import unittest
from pathlib import Path

from core.models import AnalyticsEvent, AnalyticsSnapshot, VenueMapConfig, ZoneDefinition, ZoneMetrics
from core.statistics_repository import StatisticsRepository


class StatisticsRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "stats.sqlite"
        self.repository = StatisticsRepository(self.database_path)
        self.venue = VenueMapConfig(
            world_width_m=10.0,
            world_height_m=10.0,
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

    def test_session_and_snapshot_persist(self) -> None:
        session_id = self.repository.start_session(10.0, "file", "demo.mp4", "camera-1")
        snapshot = AnalyticsSnapshot(
            timestamp=11.0,
            recent_events=[
                AnalyticsEvent(
                    event_type="person_entered_zone",
                    global_track_id="G0001",
                    zone_id="booth-a",
                    timestamp=11.0,
                )
            ],
            zone_metrics={
                "booth-a": ZoneMetrics(
                    zone_id="booth-a",
                    zone_name="Booth A",
                    zone_kind="booth",
                    current_occupancy=2,
                    unique_entries=3,
                    total_dwell_s=12.5,
                    avg_dwell_s=4.2,
                    return_count=1,
                )
            },
        )
        self.repository.record_events(session_id, snapshot.recent_events)
        self.repository.record_snapshot(session_id, snapshot, self.venue)
        self.repository.finish_session(session_id, 12.0)

        sessions = self.repository.list_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(int(sessions[0]["id"]), session_id)

        metrics = self.repository.load_session_zone_metrics(session_id)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["zone_name"], "Booth A")
        self.assertEqual(metrics[0]["current_occupancy"], 2)

        events = self.repository.load_events(session_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "person_entered_zone")


if __name__ == "__main__":
    unittest.main()
