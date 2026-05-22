import unittest

from core.models import AnalyticsSnapshot, MultiCameraRuntimeSnapshot
from ui.runtime_presenter import RuntimePresenter


class RuntimePresenterTests(unittest.TestCase):
    def test_rate_limits_presentation_updates(self) -> None:
        presenter = RuntimePresenter(refresh_interval_s=0.25)
        analytics = AnalyticsSnapshot(timestamp=1.0, total_current_occupancy=3, total_entries=5)
        runtime = MultiCameraRuntimeSnapshot(
            timestamp=1.0,
            analytics_snapshot=analytics,
            session_sync_mode="all_file_strict",
            session_media_time_s=1.0,
            active_analytics_track_count=2,
            active_map_presence_count=2,
        )

        first = presenter.submit(analytics, runtime, calibration_suffix="", now_s=1.0)
        second = presenter.submit(analytics, runtime, calibration_suffix="", now_s=1.1)
        flushed = presenter.flush(now_s=1.3)

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertIsNotNone(flushed)
        self.assertNotIn("Identity", first.status_bar_text)
        self.assertIn("Current occupancy", first.tracks_stats_text)
        self.assertIn("Map presences", first.status_bar_text)
        self.assertNotIn("Overlap merges", first.status_bar_text)

    def test_same_payload_does_not_force_fast_repaint(self) -> None:
        presenter = RuntimePresenter(refresh_interval_s=0.25)
        analytics = AnalyticsSnapshot(timestamp=1.0, total_current_occupancy=1)
        runtime = MultiCameraRuntimeSnapshot(timestamp=1.0)

        first = presenter.submit(analytics, runtime, calibration_suffix="", now_s=1.0)
        second = presenter.submit(analytics, runtime, calibration_suffix="", now_s=1.05)

        self.assertIsNotNone(first)
        self.assertIsNone(second)


if __name__ == "__main__":
    unittest.main()
