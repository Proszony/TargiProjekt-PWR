import unittest

from core.metrics import AnalyticsEngine
from core.models import AnalyticsTrack, MapPresence, VenueMapConfig, ZoneDefinition


class MetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.zone = ZoneDefinition(
            zone_id="booth-a",
            name="Booth A",
            kind="booth",
            polygon_world=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        )
        self.venue = VenueMapConfig(zones=[self.zone])
        self.engine = AnalyticsEngine(self.venue, zone_entry_min_duration_s=0.5, zone_exit_grace_s=0.5)

    def test_booth_visit_session_and_average_dwell(self) -> None:
        track = AnalyticsTrack(
            analytics_track_id="A000001",
            active=True,
            ground_anchor_world=(1.0, 1.0),
            source_camera_ids=["camera-1"],
        )
        snapshot = self.engine.update(0.0, {"A000001": track})
        self.assertEqual(snapshot.active_zone_counts, {})

        track.last_seen_ts = 0.6
        snapshot = self.engine.update(0.6, {"A000001": track})
        self.assertEqual(snapshot.active_zone_counts.get("booth-a"), 1)

        track.active = False
        track.last_seen_ts = 2.0
        snapshot = self.engine.update(2.0, {"A000001": track})

        self.assertEqual(len(snapshot.finalized_visit_sessions_recent), 1)
        visit = snapshot.finalized_visit_sessions_recent[0]
        self.assertAlmostEqual(visit.dwell_s, 1.4, places=2)
        self.assertAlmostEqual(snapshot.zone_metrics["booth-a"].avg_dwell_s, 1.4, places=2)
        self.assertAlmostEqual(snapshot.zone_metrics["booth-a"].median_dwell_s, 1.4, places=2)

    def test_peak_occupancy_tracks_multiple_people(self) -> None:
        track_a = AnalyticsTrack(
            analytics_track_id="A000001",
            active=True,
            ground_anchor_world=(1.0, 1.0),
            source_camera_ids=["camera-1"],
        )
        track_b = AnalyticsTrack(
            analytics_track_id="A000002",
            active=True,
            ground_anchor_world=(2.0, 2.0),
            source_camera_ids=["camera-1"],
        )
        self.engine.update(0.0, {"A000001": track_a, "A000002": track_b})
        snapshot = self.engine.update(0.6, {"A000001": track_a, "A000002": track_b})

        self.assertEqual(snapshot.zone_metrics["booth-a"].current_occupancy, 2)
        self.assertEqual(snapshot.zone_metrics["booth-a"].peak_occupancy, 2)

    def test_single_local_presence_counts_once(self) -> None:
        presence = MapPresence(
            presence_id="camera-1:L1",
            world_point=(1.0, 1.0),
            source_camera_ids=["camera-1"],
            source_camera_person_ids={"camera-1": "camera-1:L1"},
            merged_for_counting=False,
            dedup_mode="local_only",
        )
        self.engine.update(0.0, [presence])
        snapshot = self.engine.update(0.6, [presence])
        self.assertEqual(snapshot.active_zone_counts.get("booth-a"), 1)

    def test_two_camera_local_presences_count_as_two_booth_occupants(self) -> None:
        presence_a = MapPresence(
            presence_id="camera-1:L1",
            world_point=(1.0, 1.0),
            source_camera_ids=["camera-1"],
            source_camera_person_ids={"camera-1": "camera-1:L1"},
            merged_for_counting=False,
            dedup_mode="local_only",
        )
        presence_b = MapPresence(
            presence_id="camera-2:L1",
            world_point=(2.0, 2.0),
            source_camera_ids=["camera-2"],
            source_camera_person_ids={"camera-2": "camera-2:L1"},
            merged_for_counting=False,
            dedup_mode="local_only",
        )
        self.engine.update(0.0, [presence_a, presence_b])
        snapshot = self.engine.update(0.6, [presence_a, presence_b])
        self.assertEqual(snapshot.zone_metrics["booth-a"].current_occupancy, 2)


if __name__ == "__main__":
    unittest.main()
