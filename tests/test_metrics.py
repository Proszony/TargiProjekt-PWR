import unittest

from core.metrics import AnalyticsEngine
from core.models import GlobalTrack, VenueMapConfig, ZoneDefinition


class MetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.zone = ZoneDefinition(
            zone_id="booth-a",
            name="Booth A",
            kind="booth",
            polygon_world=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        )
        self.venue = VenueMapConfig(world_width_m=10.0, world_height_m=10.0, zones=[self.zone])
        self.engine = AnalyticsEngine(self.venue, zone_entry_min_duration_s=1.0, return_threshold_s=5.0)

    def test_return_count_after_reentry(self) -> None:
        track = GlobalTrack(global_track_id="G0001", first_seen_ts=0.0, last_seen_ts=0.0, active=True)

        track.ground_anchor_world = (1.0, 1.0)
        snapshot = self.engine.update(0.0, {"G0001": track})
        self.assertEqual(snapshot.active_zone_counts, {})

        track.last_seen_ts = 1.1
        snapshot = self.engine.update(1.1, {"G0001": track})
        self.assertEqual(snapshot.active_zone_counts.get("booth-a"), 1)

        track.ground_anchor_world = (6.0, 6.0)
        track.last_seen_ts = 2.5
        self.engine.update(2.5, {"G0001": track})
        track.last_seen_ts = 3.7
        self.engine.update(3.7, {"G0001": track})

        track.ground_anchor_world = (1.5, 1.5)
        track.last_seen_ts = 9.0
        self.engine.update(9.0, {"G0001": track})
        track.last_seen_ts = 10.2
        snapshot = self.engine.update(10.2, {"G0001": track})

        self.assertEqual(snapshot.active_zone_counts.get("booth-a"), 1)
        self.assertEqual(track.return_count, 1)
        self.assertEqual(snapshot.return_counts.get("booth-a"), 1)

    def test_zone_change_requires_stable_confirmation(self) -> None:
        zone_b = ZoneDefinition(
            zone_id="booth-b",
            name="Booth B",
            kind="booth",
            polygon_world=[(5.0, 0.0), (9.0, 0.0), (9.0, 4.0), (5.0, 4.0)],
        )
        self.engine = AnalyticsEngine(
            VenueMapConfig(world_width_m=10.0, world_height_m=10.0, zones=[self.zone, zone_b]),
            zone_entry_min_duration_s=0.35,
            return_threshold_s=5.0,
        )
        track = GlobalTrack(global_track_id="G0001", first_seen_ts=0.0, last_seen_ts=0.0, active=True)

        track.smoothed_ground_anchor_world = (1.0, 1.0)
        self.engine.update(0.0, {"G0001": track})
        track.last_seen_ts = 0.4
        snapshot = self.engine.update(0.4, {"G0001": track})
        self.assertEqual(snapshot.active_zone_counts.get("booth-a"), 1)

        track.smoothed_ground_anchor_world = (6.0, 1.0)
        track.last_seen_ts = 0.5
        snapshot = self.engine.update(0.5, {"G0001": track})
        self.assertEqual(snapshot.active_zone_counts.get("booth-a"), 1)

        track.last_seen_ts = 0.9
        snapshot = self.engine.update(0.9, {"G0001": track})
        self.assertEqual(snapshot.active_zone_counts.get("booth-b"), 1)


if __name__ == "__main__":
    unittest.main()
