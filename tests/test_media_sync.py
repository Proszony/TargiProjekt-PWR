import unittest

from core.media_sync import MultiCameraMediaSynchronizer
from core.models import CameraTrackingPacket, PlaybackSyncConfig


def _packet(camera_id: str, media_time_s: float, frame_index: int) -> CameraTrackingPacket:
    return CameraTrackingPacket(
        camera_id=camera_id,
        timestamp=media_time_s,
        media_time_s=media_time_s,
        frame_index=frame_index,
    )


class MediaSyncTests(unittest.TestCase):
    def test_strict_sync_advances_with_slowest_camera_availability(self) -> None:
        sync = MultiCameraMediaSynchronizer(
            ["camera-1", "camera-2"],
            PlaybackSyncConfig(target_fps=10.0),
        )

        sync.add_packet(_packet("camera-1", 0.00, 0))
        sync.add_packet(_packet("camera-1", 0.10, 1))
        sync.add_packet(_packet("camera-1", 0.20, 2))
        sync.add_packet(_packet("camera-2", 0.00, 0))

        first = sync.next_frame_set(now_wall_time=10.0)
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first.missing_cameras, [])
        self.assertAlmostEqual(first.media_time_s, 0.00)

        sync.add_packet(_packet("camera-2", 0.10, 1))
        second = sync.next_frame_set(now_wall_time=10.1)
        self.assertIsNotNone(second)
        assert second is not None
        self.assertEqual(second.missing_cameras, [])
        self.assertAlmostEqual(second.media_time_s, 0.10)
        self.assertEqual(second.dropped_packets_by_camera["camera-1"], 0)
        self.assertEqual(second.dropped_packets_by_camera["camera-2"], 0)


if __name__ == "__main__":
    unittest.main()
