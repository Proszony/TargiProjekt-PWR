import unittest

from core.distributed_protocol import pack_message, unpack_from_buffer
from core.distributed_serialization import (
    camera_tracking_packet_from_network_dict,
    camera_tracking_packet_to_network_dict,
    preview_frame_from_network_dict,
    preview_frame_to_network_dict,
)
from core.models import CameraIdentityTrack, CameraTrackingPacket, IdentityDebugRecord, LocalTrack, TrackletObservation


class DistributedProtocolTests(unittest.TestCase):
    def test_messagepack_framing_round_trip(self) -> None:
        message = {
            "type": "hello",
            "camera_id": "camera-1",
            "worker_id": "edge-1",
            "config_hash": "abc123",
        }
        encoded = pack_message(message)
        buffer = bytearray(encoded)
        decoded = unpack_from_buffer(buffer)

        self.assertEqual(decoded, [message])
        self.assertEqual(buffer, bytearray())

    def test_camera_packet_serialization_round_trip(self) -> None:
        packet = CameraTrackingPacket(
            camera_id="camera-1",
            timestamp=1.5,
            wall_time_s=10.0,
            media_time_s=1.4,
            frame_index=12,
            source_kind="file",
            source_fps=25.0,
            sync_ready=True,
            dropped_frame_count=2,
            processing_latency_s=0.03,
            tracklet_observations=[
                TrackletObservation(
                    camera_id="camera-1",
                    tracker_track_id=3,
                    timestamp=1.5,
                    bbox_xyxy=(10, 20, 30, 40),
                    ground_anchor_world=(1.0, 2.0),
                    ground_anchor_image=(15.0, 35.0),
                    confidence=0.8,
                    appearance_embedding=[0.1, 0.9],
                    frame_index=12,
                    media_time_s=1.4,
                    entry_edge="left",
                    exit_edge="right",
                )
            ],
            local_tracks={
                3: LocalTrack(
                    camera_id="camera-1",
                    local_track_id=3,
                    positions_world=[(1.0, 2.0)],
                    current_bbox_xyxy=(10, 20, 30, 40),
                    ground_anchor_world=(1.0, 2.0),
                    smoothed_ground_anchor_world=(1.0, 2.0),
                    bbox_center_image=(20.0, 30.0),
                )
            },
            camera_identity_tracks={
                "camera-1:P00001": CameraIdentityTrack(
                    camera_person_id="camera-1:P00001",
                    camera_id="camera-1",
                    appearance_prototype=[0.2, 0.8],
                    appearance_memory=[[0.2, 0.8]],
                    current_bbox_xyxy=(10, 20, 30, 40),
                    ground_anchor_world=(1.0, 2.0),
                    smoothed_ground_anchor_world=(1.0, 2.0),
                )
            },
            identity_debug_records=[
                IdentityDebugRecord(
                    camera_id="camera-1",
                    tracker_track_id=3,
                    camera_person_id="camera-1:P00001",
                    reason="track-continued",
                    score=0.9,
                )
            ],
            reid_backend_ready=True,
            frame_size=(1920, 1080),
            coverage_polygon_image=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
            coverage_polygon_world=[(1.0, 1.0), (2.0, 1.0), (2.0, 2.0)],
            coverage_auto_generated=True,
            coverage_confidence=0.77,
            coverage_warning_text="review",
            fps=24.5,
            status_text="Connected",
        )

        encoded = camera_tracking_packet_to_network_dict(packet)
        restored = camera_tracking_packet_from_network_dict(encoded)

        self.assertEqual(restored.camera_id, packet.camera_id)
        self.assertEqual(restored.frame_index, packet.frame_index)
        self.assertEqual(restored.frame_size, packet.frame_size)
        self.assertEqual(restored.tracklet_observations[0].bbox_xyxy, (10, 20, 30, 40))
        self.assertEqual(restored.local_tracks[3].smoothed_ground_anchor_world, (1.0, 2.0))
        self.assertEqual(
            restored.camera_identity_tracks["camera-1:P00001"].appearance_prototype,
            [0.2, 0.8],
        )
        self.assertTrue(restored.reid_backend_ready)

    def test_preview_frame_serialization_round_trip(self) -> None:
        payload = preview_frame_to_network_dict(
            camera_id="camera-1",
            frame_index=7,
            timestamp=4.2,
            width=640,
            height=360,
            jpeg_bytes=b"jpeg",
        )
        restored = preview_frame_from_network_dict(payload)

        self.assertEqual(restored["camera_id"], "camera-1")
        self.assertEqual(restored["frame_index"], 7)
        self.assertEqual(restored["jpeg_bytes"], b"jpeg")


if __name__ == "__main__":
    unittest.main()
