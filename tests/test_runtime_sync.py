import unittest

from core.multi_camera_runtime import MultiCameraPipelineManager
from core.models import AnalyticsSnapshot, MultiCameraRuntimeSnapshot


class RuntimeSyncSnapshotTests(unittest.TestCase):
    def test_runtime_snapshot_accepts_sync_drift_by_camera_s(self) -> None:
        snapshot = MultiCameraRuntimeSnapshot(
            timestamp=1.0,
            analytics_snapshot=AnalyticsSnapshot(timestamp=1.0),
            session_sync_mode="all_file_strict",
            session_media_time_s=1.0,
            sync_drift_by_camera_s={"camera-1": 0.01},
            dropped_frames_by_camera={"camera-1": 2},
            missing_cameras=["camera-2"],
            active_analytics_track_count=2,
            deduped_overlap_track_count=1,
            active_map_presence_count=1,
            merged_map_presence_count=1,
            map_presence_debug_pairs_considered=2,
            map_presence_debug_pairs_merged=1,
            overlap_tracklets_active=3,
            map_presence_matches_committed=1,
            map_presence_matches_rejected_geometry=2,
            map_presence_matches_rejected_margin=1,
            map_presence_matches_rejected_time=0,
            map_presence_matches_without_appearance=1,
        )

        self.assertEqual(snapshot.sync_drift_by_camera_s["camera-1"], 0.01)
        self.assertEqual(snapshot.dropped_frames_by_camera["camera-1"], 2)
        self.assertEqual(snapshot.missing_cameras, ["camera-2"])
        self.assertEqual(snapshot.active_analytics_track_count, 2)
        self.assertEqual(snapshot.deduped_overlap_track_count, 1)
        self.assertEqual(snapshot.active_map_presence_count, 1)
        self.assertEqual(snapshot.merged_map_presence_count, 1)
        self.assertEqual(snapshot.map_presence_debug_pairs_considered, 2)
        self.assertEqual(snapshot.map_presence_debug_pairs_merged, 1)
        self.assertEqual(snapshot.overlap_tracklets_active, 3)
        self.assertEqual(snapshot.map_presence_matches_committed, 1)
        self.assertEqual(snapshot.map_presence_matches_rejected_geometry, 2)
        self.assertEqual(snapshot.map_presence_matches_rejected_margin, 1)
        self.assertEqual(snapshot.map_presence_matches_rejected_time, 0)
        self.assertEqual(snapshot.map_presence_matches_without_appearance, 1)

    def test_file_sync_modes_get_worker_playback_clock(self) -> None:
        self.assertIsNotNone(
            MultiCameraPipelineManager.worker_file_playback_started_wall_time("all_file_strict")
        )
        self.assertIsNotNone(
            MultiCameraPipelineManager.worker_file_playback_started_wall_time("all_file_unsynced")
        )
        self.assertIsNone(
            MultiCameraPipelineManager.worker_file_playback_started_wall_time("all_live_unsynced")
        )


if __name__ == "__main__":
    unittest.main()
