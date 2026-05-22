from __future__ import annotations

from dataclasses import dataclass

from core.models import AnalyticsSnapshot, MultiCameraRuntimeSnapshot


@dataclass(slots=True)
class RuntimePresentation:
    status_bar_text: str
    tracks_stats_text: str
    fps_text: str


class RuntimePresenter:
    def __init__(self, refresh_interval_s: float = 0.25) -> None:
        self.refresh_interval_s = refresh_interval_s
        self._last_emit_at = 0.0
        self._pending: RuntimePresentation | None = None
        self._pending_hash: tuple[str, str, str] | None = None
        self._last_emitted_hash: tuple[str, str, str] | None = None

    def submit(
        self,
        analytics_snapshot: AnalyticsSnapshot,
        runtime_snapshot: MultiCameraRuntimeSnapshot,
        *,
        calibration_suffix: str = "",
        now_s: float,
    ) -> RuntimePresentation | None:
        presentation = self._build_presentation(analytics_snapshot, runtime_snapshot, calibration_suffix)
        payload_hash = (
            presentation.status_bar_text,
            presentation.tracks_stats_text,
            presentation.fps_text,
        )
        self._pending = presentation
        self._pending_hash = payload_hash
        if self._last_emitted_hash == payload_hash and now_s - self._last_emit_at < self.refresh_interval_s:
            return None
        if now_s - self._last_emit_at < self.refresh_interval_s:
            return None
        return self.flush(now_s=now_s)

    def flush(self, *, now_s: float) -> RuntimePresentation | None:
        if self._pending is None or self._pending_hash is None:
            return None
        self._last_emit_at = now_s
        self._last_emitted_hash = self._pending_hash
        presentation = self._pending
        self._pending = None
        self._pending_hash = None
        return presentation

    @staticmethod
    def _build_presentation(
        analytics_snapshot: AnalyticsSnapshot,
        runtime_snapshot: MultiCameraRuntimeSnapshot,
        calibration_suffix: str,
    ) -> RuntimePresentation:
        total_drops = sum(runtime_snapshot.dropped_frames_by_camera.values())
        aggregate = sum(packet.fps for packet in runtime_snapshot.camera_packets.values())
        max_drift = max((abs(value) for value in runtime_snapshot.sync_drift_by_camera_s.values()), default=0.0)
        active_booths = sum(1 for count in analytics_snapshot.active_zone_counts.values() if count > 0)
        mean_avg_dwell = 0.0
        non_zero_avg = [value for value in analytics_snapshot.avg_dwell_times.values() if value > 0.0]
        if non_zero_avg:
            mean_avg_dwell = sum(non_zero_avg) / len(non_zero_avg)
        status_bar_text = (
            f"Sync: {runtime_snapshot.session_sync_mode} | "
            f"Media t: {runtime_snapshot.session_media_time_s or 0.0:.2f}s | "
            f"Booths active: {active_booths} | "
            f"Current occupancy: {analytics_snapshot.total_current_occupancy} | "
            f"Avg dwell: {mean_avg_dwell:.1f}s | "
            f"Map presences: {runtime_snapshot.active_map_presence_count} | "
            f"Drift max: {max_drift:.3f}s | "
            f"Missing: {len(runtime_snapshot.missing_cameras)}"
            f"{calibration_suffix}"
        )
        tracks_stats_text = (
            f"Current occupancy: {analytics_snapshot.total_current_occupancy} | "
            f"Visits: {analytics_snapshot.total_entries} | "
            f"Booths active: {active_booths} | "
            f"Map presences: {runtime_snapshot.active_map_presence_count}"
        )
        fps_text = (
            f"Aggregate FPS: {aggregate:.1f} | "
            f"Media t: {(runtime_snapshot.session_media_time_s or 0.0):.2f}s | "
            f"Drops: {total_drops}"
        )
        return RuntimePresentation(
            status_bar_text=status_bar_text,
            tracks_stats_text=tracks_stats_text,
            fps_text=fps_text,
        )
