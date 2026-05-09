import unittest

from core.fusion import GlobalFusionEngine
from core.models import LocalTrack


def _track(local_track_id: int, timestamp: float, world: tuple[float, float]) -> LocalTrack:
    return LocalTrack(
        camera_id="camera-1",
        local_track_id=local_track_id,
        first_seen_ts=timestamp,
        last_seen_ts=timestamp,
        current_bbox_xyxy=(100, 100, 180, 260),
        last_bbox_xyxy_for_matching=(100, 100, 180, 260),
        ground_anchor_world=world,
        smoothed_ground_anchor_world=world,
        ground_anchor_image=(120.0, 260.0),
        active=True,
        confirmed=True,
        observed_frames=3,
    )


class FusionTests(unittest.TestCase):
    def test_single_camera_uses_deterministic_global_ids(self) -> None:
        fusion = GlobalFusionEngine()
        track = _track(42, 0.0, (1.0, 1.0))

        tracks = fusion.update(0.0, {42: track}, [])

        self.assertEqual(sorted(tracks), ["camera-1:T0042"])
        self.assertEqual(tracks["camera-1:T0042"].member_local_tracks, ["camera-1:42"])

    def test_expired_track_is_marked_inactive(self) -> None:
        fusion = GlobalFusionEngine()
        track = _track(7, 0.0, (2.0, 3.0))
        fusion.update(0.0, {7: track}, [])

        expired = _track(7, 5.0, (2.0, 3.0))
        tracks = fusion.update(5.0, {}, [expired])

        self.assertIn("camera-1:T0007", tracks)
        self.assertFalse(tracks["camera-1:T0007"].active)
        self.assertEqual(tracks["camera-1:T0007"].inactive_since_ts, 5.0)


if __name__ == "__main__":
    unittest.main()
