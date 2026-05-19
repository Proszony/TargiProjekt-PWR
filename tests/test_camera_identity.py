import unittest

from core.camera_identity import CameraIdentityEngine
from core.models import IdentityConfig, LocalTrack, TrackletObservation


def _local_track(
    local_track_id: int,
    timestamp: float,
    world: tuple[float, float],
    *,
    appearance: list[float] | None = None,
) -> LocalTrack:
    return LocalTrack(
        camera_id="camera-1",
        local_track_id=local_track_id,
        positions_world=[world],
        last_seen_ts=timestamp,
        current_bbox_xyxy=(100, 100, 180, 260),
        last_bbox_xyxy_for_matching=(100, 100, 180, 260),
        ground_anchor_image=(120.0, 260.0),
        ground_anchor_world=world,
        smoothed_ground_anchor_world=world,
        confidence=0.95,
        first_seen_ts=timestamp,
        active=True,
        appearance_descriptor=appearance or [1.0, 0.0, 0.0],
        observed_frames=3,
        confirmed=True,
        last_entry_edge="left",
        last_exit_edge="right",
        edge_proximity_score=0.1,
    )


def _observation(local_track: LocalTrack, *, frame_index: int = 0) -> TrackletObservation:
    return TrackletObservation(
        camera_id=local_track.camera_id,
        tracker_track_id=local_track.local_track_id,
        timestamp=local_track.last_seen_ts,
        bbox_xyxy=local_track.current_bbox_xyxy or (0, 0, 1, 1),
        ground_anchor_world=local_track.ground_anchor_world,
        ground_anchor_image=local_track.ground_anchor_image,
        confidence=local_track.confidence,
        appearance_embedding=list(local_track.appearance_descriptor),
        frame_index=frame_index,
        entry_edge=local_track.last_entry_edge,
        exit_edge=local_track.last_exit_edge,
    )


class CameraIdentityTests(unittest.TestCase):
    def test_new_tracker_track_can_restitch_to_existing_camera_person(self) -> None:
        engine = CameraIdentityEngine("camera-1", IdentityConfig(single_camera_restitch_threshold=0.60))
        first = _local_track(15, 0.0, (2.0, 2.0))
        active, _expired, _debug = engine.update(0.0, {15: first}, [], [_observation(first)])
        original_person_id = next(iter(active))

        expired = _local_track(15, 0.6, (2.1, 2.0))
        expired.active = False
        engine.update(0.6, {}, [expired], [])

        second = _local_track(18, 1.0, (2.15, 2.05))
        active, _expired, debug = engine.update(1.0, {18: second}, [], [_observation(second, frame_index=1)])

        self.assertEqual(sorted(active), [original_person_id])
        self.assertEqual(active[original_person_id].current_local_track_id, 18)
        self.assertTrue(second.display_track_id.startswith(original_person_id.split(":")[-1]))
        self.assertTrue(any(record.reason == "restitch_lost" for record in debug))

    def test_long_gap_creates_new_camera_person(self) -> None:
        engine = CameraIdentityEngine("camera-1", IdentityConfig(single_camera_max_gap_s=0.5, lost_track_retention_s=0.5))
        first = _local_track(4, 0.0, (1.0, 1.0))
        active, _expired, _debug = engine.update(0.0, {4: first}, [], [_observation(first)])
        first_person_id = next(iter(active))

        expired = _local_track(4, 0.6, (1.0, 1.0))
        expired.active = False
        engine.update(0.6, {}, [expired], [])

        second = _local_track(8, 2.0, (1.1, 1.1))
        active, _expired, _debug = engine.update(2.0, {8: second}, [], [_observation(second)])

        self.assertEqual(len(active), 1)
        second_person_id = next(iter(active))
        self.assertNotEqual(first_person_id, second_person_id)


if __name__ == "__main__":
    unittest.main()
