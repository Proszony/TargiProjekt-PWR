from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

from PySide6.QtCore import QByteArray, QBuffer, QIODevice, Qt
from PySide6.QtGui import QImage

from core import runtime_defaults as rd
from core.config import ConfigRepository
from core.distributed_protocol import (
    MESSAGE_CAMERA_PACKET,
    MESSAGE_ERROR,
    MESSAGE_HEARTBEAT,
    MESSAGE_HELLO,
    MESSAGE_PREVIEW_FRAME,
    MESSAGE_START_SESSION,
    MESSAGE_STATUS,
    MESSAGE_STOP_SESSION,
    pack_message,
    unpack_from_buffer,
)
from core.distributed_serialization import (
    camera_config_sha256,
    camera_tracking_packet_to_network_dict,
    preview_frame_to_network_dict,
)
from core.models import CameraConfig, CameraTrackingPacket, PlaybackSyncConfig, ProjectConfig
from core.statistics_service import StatisticsService
from core.streaming import CameraPipelineWorker


class DistributedCameraWorker:
    def __init__(
        self,
        project_root: Path,
        camera_id: str,
        server_host: str | None = None,
        server_port: int | None = None,
    ) -> None:
        self.project_root = project_root
        self.camera_id = camera_id
        self.config_repo = ConfigRepository(project_root)
        self.project_config = self.config_repo.ensure_defaults()
        self.camera_config = self._camera_config(self.project_config)
        self.server_host = server_host or self.project_config.distributed_runtime.server_bind_host
        self.server_port = server_port or self.project_config.distributed_runtime.server_port
        self.statistics_service = StatisticsService(project_root)
        self._running = True
        self._socket: socket.socket | None = None
        self._socket_lock = threading.Lock()
        self._connected = False
        self._session_sync_mode = "all_live_unsynced"
        self._session_started_at_unix_s: float | None = None
        self._playback_sync = PlaybackSyncConfig.from_dict(self.project_config.playback_sync.to_dict())
        self._pipeline_worker: CameraPipelineWorker | None = None
        self._pipeline_thread: threading.Thread | None = None
        self._preview_interval_s = 1.0 / max(self.project_config.distributed_runtime.preview_fps, 0.5)
        self._last_preview_sent_at = 0.0
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name="distributed-heartbeat", daemon=True)

    def run_forever(self) -> None:
        self._heartbeat_thread.start()
        backoff_s = 1.0
        while self._running:
            try:
                self._reload_project()
                self._connect_and_stream()
                backoff_s = 1.0
            except OSError:
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2.0, 8.0)
            except KeyboardInterrupt:
                break
        self.stop()

    def stop(self) -> None:
        self._running = False
        self._connected = False
        self._stop_pipeline()
        with self._socket_lock:
            sock = self._socket
            self._socket = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def _connect_and_stream(self) -> None:
        sock = socket.create_connection((self.server_host, self.server_port), timeout=5.0)
        sock.settimeout(0.5)
        with self._socket_lock:
            self._socket = sock
        self._connected = True
        self._send_message(
            {
                "type": MESSAGE_HELLO,
                "worker_id": self.camera_config.remote_worker_id or socket.gethostname(),
                "camera_id": self.camera_id,
                "app_version": "distributed-runtime-test",
                "hostname": socket.gethostname(),
                "protocol_version": self.project_config.distributed_runtime.protocol_version,
                "config_hash": camera_config_sha256(self.camera_config),
            }
        )
        buffer = bytearray()
        while self._running and self._connected:
            try:
                chunk = sock.recv(65536)
            except TimeoutError:
                continue
            except OSError:
                break
            if not chunk:
                break
            buffer.extend(chunk)
            for message in unpack_from_buffer(buffer):
                self._handle_message(message)
        self._connected = False
        with self._socket_lock:
            if self._socket is sock:
                self._socket = None
        try:
            sock.close()
        except OSError:
            pass

    def _handle_message(self, message: dict[str, object]) -> None:
        message_type = str(message.get("type", ""))
        if message_type == MESSAGE_START_SESSION:
            self._session_sync_mode = str(message.get("session_sync_mode", "all_live_unsynced"))
            session_started_at = message.get("session_started_at_unix_s")
            self._session_started_at_unix_s = (
                float(session_started_at) if isinstance(session_started_at, (int, float)) else None
            )
            playback_sync = message.get("playback_sync", {})
            if isinstance(playback_sync, dict):
                self._playback_sync = PlaybackSyncConfig.from_dict(playback_sync)
            self._start_pipeline()
            return
        if message_type == MESSAGE_STOP_SESSION:
            self._session_started_at_unix_s = None
            self._stop_pipeline()
            return
        if message_type == MESSAGE_ERROR:
            print(f"Server error for {self.camera_id}: {message.get('message', '')}")

    def _start_pipeline(self) -> None:
        if self._pipeline_thread is not None and self._pipeline_thread.is_alive():
            return
        file_playback_started_wall_time = self._file_playback_started_wall_time()
        worker = CameraPipelineWorker(
            camera_config=CameraConfig.from_dict(self.camera_config.to_dict()),
            venue_map=self.project_config.venue_map,
            statistics_service=self.statistics_service,
            project_root=self.project_root,
            playback_sync_config=self._playback_sync,
            session_sync_mode=self._session_sync_mode,
            file_playback_started_wall_time=file_playback_started_wall_time,
            confidence=0.18,
            inference_size=736,
            preview_fps=self.project_config.distributed_runtime.preview_fps,
            on_frame_ready=self._handle_frame_ready,
            on_camera_packet_ready=self._handle_camera_packet,
            on_status_changed=self._handle_status_changed,
            on_error_occurred=self._handle_error,
        )
        self._pipeline_worker = worker
        self._pipeline_thread = threading.Thread(target=worker.run, name=f"worker-{self.camera_id}", daemon=True)
        self._pipeline_thread.start()

    def _stop_pipeline(self) -> None:
        worker = self._pipeline_worker
        thread = self._pipeline_thread
        if worker is not None:
            worker.stop()
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self._pipeline_worker = None
        self._pipeline_thread = None

    def _file_playback_started_wall_time(self) -> float | None:
        if not (self._session_sync_mode.startswith("all_file") or self._session_sync_mode == "single_file_realtime"):
            return None
        if self._session_sync_mode == "single_file_realtime":
            return time.perf_counter()
        if self._session_started_at_unix_s is None:
            return time.perf_counter()
        return time.perf_counter() + (self._session_started_at_unix_s - time.time())

    def _handle_frame_ready(self, camera_id: str, frame_index: int, image: object) -> None:
        if not self._connected or not isinstance(image, QImage):
            return
        now = time.monotonic()
        if now - self._last_preview_sent_at < self._preview_interval_s:
            return
        jpeg_bytes = self._encode_jpeg(image)
        if not jpeg_bytes:
            return
        self._last_preview_sent_at = now
        self._send_message(
            {
                "type": MESSAGE_PREVIEW_FRAME,
                "payload": preview_frame_to_network_dict(
                    camera_id=camera_id,
                    frame_index=frame_index,
                    timestamp=time.time(),
                    width=image.width(),
                    height=image.height(),
                    jpeg_bytes=jpeg_bytes,
                ),
            }
        )

    def _handle_camera_packet(self, packet: object) -> None:
        if not self._connected or not isinstance(packet, CameraTrackingPacket):
            return
        self._send_message(
            {
                "type": MESSAGE_CAMERA_PACKET,
                "payload": camera_tracking_packet_to_network_dict(packet),
            }
        )

    def _handle_status_changed(self, status_text: str) -> None:
        print(f"[{self.camera_id}] {status_text}")
        self._send_message(
            {
                "type": MESSAGE_STATUS,
                "camera_id": self.camera_id,
                "status_text": status_text,
            }
        )

    def _handle_error(self, message: str) -> None:
        print(f"[{self.camera_id}] ERROR: {message}")
        self._send_message(
            {
                "type": MESSAGE_ERROR,
                "camera_id": self.camera_id,
                "message": message,
            }
        )

    def _heartbeat_loop(self) -> None:
        while self._running:
            if self._connected:
                self._send_message(
                    {
                        "type": MESSAGE_HEARTBEAT,
                        "camera_id": self.camera_id,
                        "timestamp": time.time(),
                    }
                )
            time.sleep(self.project_config.distributed_runtime.worker_heartbeat_interval_s)

    def _send_message(self, message: dict[str, object]) -> None:
        if not self._connected:
            return
        with self._socket_lock:
            sock = self._socket
            if sock is None:
                return
            try:
                sock.sendall(pack_message(message))
            except OSError:
                self._connected = False

    def _reload_project(self) -> None:
        self.project_config = self.config_repo.load_project()
        self.camera_config = self._camera_config(self.project_config)
        self._preview_interval_s = 1.0 / max(self.project_config.distributed_runtime.preview_fps, 0.5)

    def _camera_config(self, project_config: ProjectConfig) -> CameraConfig:
        for camera in project_config.cameras:
            if camera.camera_id == self.camera_id:
                return CameraConfig.from_dict(camera.to_dict())
        raise ValueError(f"Camera '{self.camera_id}' not found in project config.")

    def _encode_jpeg(self, image: QImage) -> bytes:
        encoded_image = image
        if image.width() > rd.DEFAULT_DISTRIBUTED_PREVIEW_MAX_WIDTH:
            encoded_image = image.scaledToWidth(
                rd.DEFAULT_DISTRIBUTED_PREVIEW_MAX_WIDTH,
                Qt.SmoothTransformation,
            )
        buffer = QByteArray()
        handle = QBuffer(buffer)
        if not handle.open(QIODevice.WriteOnly):
            return b""
        quality = self.project_config.distributed_runtime.preview_jpeg_quality
        encoded_image.save(handle, "JPEG", quality)
        handle.close()
        return bytes(buffer)


def run_worker_process(
    *,
    project_root: Path,
    camera_id: str,
    server_host: str | None = None,
    server_port: int | None = None,
) -> int:
    worker = DistributedCameraWorker(
        project_root=project_root,
        camera_id=camera_id,
        server_host=server_host,
        server_port=server_port,
    )
    worker.run_forever()
    return 0
