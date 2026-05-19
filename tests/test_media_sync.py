import unittest

from core.media_sync import MultiCameraMediaSynchronizer
from core.models import CameraTrackingPacket, PlaybackSyncConfig


def _packet(camera_id: str, media_time_s: float, frame_index: int) -> CameraTrackingPacket:
    return CameraTrackingPacket(
        camera_id=camera_id,
        timestamp=media_time_s,
        wall_time_s=100.0 + media_time_s,
        media_time_s=media_time_s,
        frame_index=frame_index,
        source_kind="file",
        source_fps=30.0,
    )


class MediaSynchronizerTests(unittest.TestCase):
    def test_aligns_cameras_by_shared_media_time(self) -> None:
        synchronizer = MultiCameraMediaSynchronizer(
            ["camera-a", "camera-b"],
            PlaybackSyncConfig(target_fps=30.0, sync_tolerance_s=0.04, stale_packet_threshold_s=0.2),
        )
        synchronizer.add_packet(_packet("camera-a", 0.00, 0))
        synchronizer.add_packet(_packet("camera-b", 0.00, 0))

        frame_set = synchronizer.next_frame_set(now_wall_time=10.0)
        self.assertIsNotNone(frame_set)
        assert frame_set is not None
        self.assertAlmostEqual(frame_set.media_time_s, 0.0, places=3)

        synchronizer.add_packet(_packet("camera-a", 0.10, 3))
        synchronizer.add_packet(_packet("camera-b", 0.11, 3))
        frame_set = synchronizer.next_frame_set(now_wall_time=10.11)

        self.assertIsNotNone(frame_set)
        assert frame_set is not None
        self.assertLess(abs(frame_set.drift_by_camera_s["camera-a"]), 0.05)
        self.assertLess(abs(frame_set.drift_by_camera_s["camera-b"]), 0.05)

    def test_drops_older_packets_and_keeps_latest_candidate(self) -> None:
        synchronizer = MultiCameraMediaSynchronizer(
            ["camera-a", "camera-b"],
            PlaybackSyncConfig(target_fps=30.0, sync_tolerance_s=0.04, stale_packet_threshold_s=0.2),
        )
        synchronizer.add_packet(_packet("camera-b", 0.0, 0))
        synchronizer.add_packet(_packet("camera-a", 0.0, 0))
        frame_set = synchronizer.next_frame_set(now_wall_time=5.0)
        self.assertIsNotNone(frame_set)

        synchronizer.add_packet(_packet("camera-a", 0.03, 1))
        synchronizer.add_packet(_packet("camera-a", 0.06, 2))
        frame_set = synchronizer.next_frame_set(now_wall_time=5.08)
        self.assertIsNotNone(frame_set)
        assert frame_set is not None
        self.assertGreaterEqual(frame_set.dropped_packets_by_camera["camera-a"], 1)

    def test_marks_camera_missing_when_packet_is_too_stale(self) -> None:
        synchronizer = MultiCameraMediaSynchronizer(
            ["camera-a", "camera-b"],
            PlaybackSyncConfig(
                target_fps=30.0,
                sync_tolerance_s=0.04,
                stale_packet_threshold_s=0.05,
                camera_missing_timeout_s=0.1,
            ),
        )
        synchronizer.add_packet(_packet("camera-a", 0.0, 0))
        synchronizer.add_packet(_packet("camera-b", 0.0, 0))
        initial = synchronizer.next_frame_set(now_wall_time=2.0)
        self.assertIsNotNone(initial)

        synchronizer.add_packet(_packet("camera-a", 0.03, 1))
        synchronizer.add_packet(_packet("camera-a", 0.06, 2))
        synchronizer.add_packet(_packet("camera-a", 0.09, 3))
        synchronizer.add_packet(_packet("camera-a", 0.12, 4))
        frame_set = synchronizer.next_frame_set(now_wall_time=2.12)
        self.assertIsNotNone(frame_set)
        assert frame_set is not None
        self.assertIn("camera-b", frame_set.missing_cameras)

    def test_startup_anchors_to_first_common_buffered_media_time(self) -> None:
        synchronizer = MultiCameraMediaSynchronizer(
            ["camera-a", "camera-b"],
            PlaybackSyncConfig(
                target_fps=30.0,
                sync_tolerance_s=0.04,
                stale_packet_threshold_s=0.5,
                max_buffered_packets_per_camera=3,
            ),
        )
        synchronizer.add_packet(_packet("camera-a", 1.20, 36))
        synchronizer.add_packet(_packet("camera-a", 1.23, 37))
        synchronizer.add_packet(_packet("camera-b", 1.10, 33))
        synchronizer.add_packet(_packet("camera-b", 1.20, 36))

        frame_set = synchronizer.next_frame_set(now_wall_time=15.0)

        self.assertIsNotNone(frame_set)
        assert frame_set is not None
        self.assertAlmostEqual(frame_set.media_time_s, 1.20, places=2)
        self.assertIn("camera-a", frame_set.camera_packets)
        self.assertIn("camera-b", frame_set.camera_packets)


if __name__ == "__main__":
    unittest.main()
