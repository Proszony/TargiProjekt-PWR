from __future__ import annotations

import time
from collections import deque

from core.models import CameraTrackingPacket, PlaybackSyncConfig, SynchronizedCameraFrameSet


class MultiCameraMediaSynchronizer:
    def __init__(
        self,
        camera_ids: list[str],
        playback_sync: PlaybackSyncConfig,
    ) -> None:
        self.camera_ids = list(camera_ids)
        self.playback_sync = playback_sync
        self._buffers: dict[str, deque[CameraTrackingPacket]] = {
            camera_id: deque() for camera_id in self.camera_ids
        }
        self._selected_packets: dict[str, CameraTrackingPacket] = {}
        self._dropped_counts: dict[str, int] = {camera_id: 0 for camera_id in self.camera_ids}
        self._playback_started_wall_time: float | None = None
        self._last_target_media_time_s = -1.0
        self._startup_media_time_s: float | None = None

    def reset(self) -> None:
        for buffer in self._buffers.values():
            buffer.clear()
        self._selected_packets.clear()
        self._dropped_counts = {camera_id: 0 for camera_id in self.camera_ids}
        self._playback_started_wall_time = None
        self._last_target_media_time_s = -1.0
        self._startup_media_time_s = None

    def add_packet(self, packet: CameraTrackingPacket) -> None:
        if packet.camera_id not in self._buffers or packet.media_time_s is None:
            return
        buffer = self._buffers[packet.camera_id]
        buffer.append(packet)
        max_buffered_packets = self.playback_sync.max_buffered_packets_per_camera
        if self._playback_started_wall_time is None:
            max_buffered_packets = max(
                max_buffered_packets,
                int(round(max(self.playback_sync.target_fps, 1.0) * 3.0)),
            )
        while len(buffer) > max_buffered_packets:
            buffer.popleft()
            self._dropped_counts[packet.camera_id] += 1

    def next_frame_set(self, now_wall_time: float | None = None) -> SynchronizedCameraFrameSet | None:
        if not self.camera_ids:
            return None
        current_time = now_wall_time if now_wall_time is not None else time.perf_counter()
        if self._playback_started_wall_time is None:
            if not self._all_cameras_buffered():
                return None
            self._startup_media_time_s = self._resolve_startup_media_time_s()
            self._playback_started_wall_time = current_time - self._startup_media_time_s
            self._last_target_media_time_s = -1.0

        target_media_time_s = self._resolve_target_media_time_s(current_time)
        if target_media_time_s is None:
            return None
        min_interval = 1.0 / max(self.playback_sync.target_fps, 1e-6)
        if self._last_target_media_time_s >= 0.0 and (
            target_media_time_s - self._last_target_media_time_s < min_interval * 0.8
        ):
            return None

        selected_packets: dict[str, CameraTrackingPacket] = {}
        missing_cameras: list[str] = []
        drift_by_camera_s: dict[str, float] = {}

        for camera_id in self.camera_ids:
            packet = self._select_packet(camera_id, target_media_time_s)
            if packet is None:
                missing_cameras.append(camera_id)
                continue
            selected_packets[camera_id] = packet
            if packet.media_time_s is not None:
                drift_by_camera_s[camera_id] = target_media_time_s - packet.media_time_s

        if not selected_packets:
            return None

        self._last_target_media_time_s = target_media_time_s
        return SynchronizedCameraFrameSet(
            media_time_s=target_media_time_s,
            camera_packets=selected_packets,
            dropped_packets_by_camera=dict(self._dropped_counts),
            missing_cameras=missing_cameras,
            drift_by_camera_s=drift_by_camera_s,
        )

    def _all_cameras_buffered(self) -> bool:
        return all(self._buffers[camera_id] for camera_id in self.camera_ids)

    def _resolve_startup_media_time_s(self) -> float:
        earliest_media_times: list[float] = []
        for camera_id in self.camera_ids:
            buffer = self._buffers[camera_id]
            packet = next((item for item in buffer if item.media_time_s is not None), None)
            if packet is None or packet.media_time_s is None:
                return 0.0
            earliest_media_times.append(packet.media_time_s)
        if not earliest_media_times:
            return 0.0
        return max(earliest_media_times)

    def _resolve_target_media_time_s(self, current_time: float) -> float | None:
        available_media_times: list[float] = []
        for camera_id in self.camera_ids:
            latest_available = self._latest_available_media_time(camera_id)
            if latest_available is None:
                return None
            available_media_times.append(latest_available)
        if not available_media_times:
            return None
        if self._last_target_media_time_s < 0.0:
            return min(available_media_times)
        return max(self._last_target_media_time_s, min(available_media_times))

    def _latest_available_media_time(self, camera_id: str) -> float | None:
        latest_selected = self._selected_packets.get(camera_id)
        if self._buffers[camera_id]:
            buffered_packet = self._buffers[camera_id][-1]
            if buffered_packet.media_time_s is not None:
                if latest_selected is None or latest_selected.media_time_s is None:
                    return buffered_packet.media_time_s
                return max(buffered_packet.media_time_s, latest_selected.media_time_s)
        if latest_selected is None or latest_selected.media_time_s is None:
            return None
        return latest_selected.media_time_s

    def _select_packet(self, camera_id: str, target_media_time_s: float) -> CameraTrackingPacket | None:
        buffer = self._buffers[camera_id]
        chosen_new: CameraTrackingPacket | None = None
        chosen_new_count = 0

        while buffer and buffer[0].media_time_s is not None:
            candidate = buffer[0]
            if candidate.media_time_s > target_media_time_s + self.playback_sync.sync_tolerance_s:
                break
            chosen_new = buffer.popleft()
            chosen_new_count += 1

        if chosen_new_count > 1:
            self._dropped_counts[camera_id] += chosen_new_count - 1

        if chosen_new is not None:
            self._selected_packets[camera_id] = chosen_new

        packet = self._selected_packets.get(camera_id)
        if packet is None or packet.media_time_s is None:
            return None

        packet_age_s = target_media_time_s - packet.media_time_s
        if packet_age_s > self.playback_sync.camera_missing_timeout_s:
            return None
        if packet_age_s > self.playback_sync.stale_packet_threshold_s:
            return None
        return packet
