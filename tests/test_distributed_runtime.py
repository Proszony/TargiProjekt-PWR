import tempfile
import unittest
from pathlib import Path

from core.models import (
    CameraConfig,
    CameraIdentityTrack,
    CameraTrackingPacket,
    DistributedRuntimeConfig,
    MultiCameraRuntimeSnapshot,
    OverlapDedupConfig,
    ProjectConfig,
    VenueMapConfig,
    ZoneDefinition,
)
from core.statistics_service import StatisticsService

try:
    from core.multi_camera_runtime import MultiCameraPipelineManager
except ModuleNotFoundError as exc:  # pragma: no cover - optional UI dependency
    MultiCameraPipelineManager = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _packet(camera_id: str, person_id: str, point: tuple[float, float], appearance: list[float]) -> CameraTrackingPacket:
    return CameraTrackingPacket(
        camera_id=camera_id,
        timestamp=1.0,
        frame_index=1,
        camera_identity_tracks={
            person_id: CameraIdentityTrack(
                camera_person_id=person_id,
                camera_id=camera_id,
                current_bbox_xyxy=(10, 10, 30, 60),
                ground_anchor_world=point,
                smoothed_ground_anchor_world=point,
                appearance_prototype=list(appearance),
                appearance_memory=[list(appearance)],
            )
        },
        reid_backend_ready=True,
        frame_size=(640, 480),
        coverage_polygon_world=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
    )


@unittest.skipIf(MultiCameraPipelineManager is None, f"UI dependencies unavailable: {_IMPORT_ERROR}")
class DistributedRuntimeTests(unittest.TestCase):
    def test_remote_packet_ingestion_updates_runtime_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = ProjectConfig(
                venue_map=VenueMapConfig(
                    zones=[
                        ZoneDefinition(
                            zone_id="booth-1",
                            name="Booth 1",
                            kind="booth",
                            polygon_world=[(1.0, 1.0), (3.5, 1.0), (3.5, 3.5), (1.0, 3.5)],
                        )
                    ]
                ),
                cameras=[
                    CameraConfig(
                        camera_id="camera-local",
                        runtime_mode="local",
                        calibration_valid=True,
                        coverage_polygon_world=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
                    ),
                    CameraConfig(
                        camera_id="camera-remote",
                        runtime_mode="remote",
                        remote_worker_id="edge-1",
                        calibration_valid=True,
                        coverage_polygon_world=[(0.5, 0.0), (4.5, 0.0), (4.5, 4.0), (0.5, 4.0)],
                    ),
                ],
                overlap_dedup=OverlapDedupConfig(confirmation_frames=1),
                distributed_runtime=DistributedRuntimeConfig(enabled=True),
            )
            manager = MultiCameraPipelineManager(project, StatisticsService(root), root)
            snapshots: list[MultiCameraRuntimeSnapshot] = []
            manager.runtime_snapshot_ready.connect(lambda snapshot: snapshots.append(snapshot))

            manager._handle_camera_packet(_packet("camera-local", "camera-local:P1", (2.0, 2.0), [1.0, 0.0, 0.0]))
            manager._handle_remote_packet(
                _packet("camera-remote", "camera-remote:P1", (2.1, 2.0), [0.99, 0.01, 0.0])
            )

            self.assertTrue(snapshots)
            latest = snapshots[-1]
            self.assertEqual(set(latest.camera_packets), {"camera-local", "camera-remote"})
            self.assertGreaterEqual(latest.active_map_presence_count, 1)
            self.assertEqual(latest.analytics_snapshot.total_current_occupancy, 1)


if __name__ == "__main__":
    unittest.main()
