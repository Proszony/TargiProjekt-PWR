from __future__ import annotations

import logging
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
    MESSAGE_WORKER_CONFIG,
    pack_message,
    unpack_from_buffer,
)
from core.distributed_serialization import (
    camera_config_sha256,
    camera_tracking_packet_to_network_dict,
    preview_frame_to_network_dict,
    worker_config_from_network_dict,
)
from core.models import CameraConfig, CameraTrackingPacket, PlaybackSyncConfig, ProjectConfig
from core.statistics_service import StatisticsService
from core.streaming import CameraPipelineWorker


LOGGER = logging.getLogger("fair_monitor.worker")
LOG_FORMAT = "%(asctime)s %(levelname)s [worker] %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


def configure_worker_logging(level_name: str = "INFO") -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    LOGGER.setLevel(level)
    LOGGER.propagate = False
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        LOGGER.addHandler(handler)
    for handler in LOGGER.handlers:
        handler.setLevel(level)


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
        self._server_config_hash = ""
        self._pipeline_worker: CameraPipelineWorker | None = None
        self._pipeline_thread: threading.Thread | None = None
        self._preview_interval_s = 1.0 / max(self.project_config.distributed_runtime.preview_fps, 0.5)
        self._last_preview_sent_at = 0.0
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name="distributed-heartbeat", daemon=True)

    def run_forever(self) -> None:
        self._heartbeat_thread.start()
        LOGGER.info(
            "Starting worker camera_id=%s worker_id=%s target=%s:%s hostname=%s protocol=%s",
            self.camera_id,
            self.camera_config.remote_worker_id or "<auto>",
            self.server_host,
            self.server_port,
            socket.gethostname(),
            self.project_config.distributed_runtime.protocol_version,
        )
        if self.server_host in {"0.0.0.0", "::"}:
            LOGGER.warning(
                "Worker target host is %s, which is usually a server bind address. "
                "Use the server PC LAN IP with --server-host.",
                self.server_host,
            )
        backoff_s = 1.0
        while self._running:
            try:
                self._reload_project()
                self._connect_and_stream()
                backoff_s = 1.0
                if self._running:
                    LOGGER.warning(
                        "Disconnected from server %s:%s; retrying in %.1fs",
                        self.server_host,
                        self.server_port,
                        backoff_s,
                    )
                    time.sleep(backoff_s)
                    backoff_s = min(backoff_s * 2.0, 8.0)
            except OSError as exc:
                hint = _connection_failure_hint(exc, self.server_port)
                suffix = f" ({hint})" if hint else ""
                LOGGER.warning(
                    "Connection to %s:%s failed: %s; retrying in %.1fs%s",
                    self.server_host,
                    self.server_port,
                    exc,
                    backoff_s,
                    suffix,
                )
                LOGGER.debug("Worker connection failure details", exc_info=True)
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2.0, 8.0)
            except KeyboardInterrupt:
                LOGGER.info("Worker interrupted; shutting down")
                break
        self.stop()
        LOGGER.info("Worker stopped")

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
        LOGGER.info("Connecting to server %s:%s", self.server_host, self.server_port)
        sock = socket.create_connection((self.server_host, self.server_port), timeout=5.0)
        sock.settimeout(0.5)
        with self._socket_lock:
            self._socket = sock
        self._connected = True
        LOGGER.info("Connected to server %s:%s", self.server_host, self.server_port)
        hello = {
            "type": MESSAGE_HELLO,
            "worker_id": self.camera_config.remote_worker_id,
            "camera_id": self.camera_id,
            "app_version": "distributed-runtime-test",
            "hostname": socket.gethostname(),
            "protocol_version": self.project_config.distributed_runtime.protocol_version,
            "config_hash": camera_config_sha256(self.camera_config),
        }
        if self._send_message(hello):
            LOGGER.info(
                "Sent hello camera_id=%s worker_id=%s config_hash=%s",
                self.camera_id,
                self.camera_config.remote_worker_id or "<empty>",
                hello["config_hash"],
            )
        buffer = bytearray()
        while self._running and self._connected:
            try:
                chunk = sock.recv(65536)
            except TimeoutError:
                continue
            except OSError as exc:
                LOGGER.warning("Connection lost while reading from server: %s", exc)
                LOGGER.debug("Worker receive failure details", exc_info=True)
                break
            if not chunk:
                LOGGER.warning("Server closed the connection")
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
        if message_type == MESSAGE_WORKER_CONFIG:
            LOGGER.info("Received worker config from server")
            payload = message.get("payload", {})
            if isinstance(payload, dict):
                self._apply_worker_config(payload)
            return
        if message_type == MESSAGE_START_SESSION:
            self._session_sync_mode = str(message.get("session_sync_mode", "all_live_unsynced"))
            session_started_at = message.get("session_started_at_unix_s")
            self._session_started_at_unix_s = (
                float(session_started_at) if isinstance(session_started_at, (int, float)) else None
            )
            playback_sync = message.get("playback_sync", {})
            if isinstance(playback_sync, dict):
                self._playback_sync = PlaybackSyncConfig.from_dict(playback_sync)
            LOGGER.info("Received start session sync_mode=%s", self._session_sync_mode)
            self._start_pipeline()
            return
        if message_type == MESSAGE_STOP_SESSION:
            LOGGER.info("Received stop session")
            self._session_started_at_unix_s = None
            self._stop_pipeline()
            return
        if message_type == MESSAGE_ERROR:
            LOGGER.error("Server error for %s: %s", self.camera_id, message.get("message", ""))
            return
        LOGGER.debug("Ignoring server message type=%s", message_type or "<empty>")

    def _start_pipeline(self) -> None:
        if self._pipeline_thread is not None and self._pipeline_thread.is_alive():
            LOGGER.debug("Pipeline already running for camera_id=%s", self.camera_id)
            return
        file_playback_started_wall_time = self._file_playback_started_wall_time()
        LOGGER.info(
            "Starting camera pipeline camera_id=%s source=%s:%s sync_mode=%s",
            self.camera_id,
            self.camera_config.source_type,
            self.camera_config.source_value or "<empty>",
            self._session_sync_mode,
        )
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
        LOGGER.info("Camera pipeline thread started camera_id=%s", self.camera_id)

    def _apply_worker_config(self, payload: dict[str, object]) -> None:
        camera_config, venue_map, playback_sync, distributed_runtime, config_hash = (
            worker_config_from_network_dict(payload, getattr(self, "project_root", None))
        )
        if camera_config.camera_id != self.camera_id:
            self._handle_error(
                f"server sent config for '{camera_config.camera_id}' to worker '{self.camera_id}'"
            )
            return
        previous_source = self._source_signature(self.camera_config)
        next_source = self._source_signature(camera_config)
        self.camera_config = camera_config
        self.project_config.venue_map = venue_map
        self.project_config.playback_sync = playback_sync
        self.project_config.distributed_runtime = distributed_runtime
        self._playback_sync = PlaybackSyncConfig.from_dict(playback_sync.to_dict())
        self._server_config_hash = config_hash
        self._preview_interval_s = 1.0 / max(distributed_runtime.preview_fps, 0.5)
        LOGGER.info(
            "Applied worker config camera_id=%s worker_id=%s source=%s:%s preview_fps=%.2f config_hash=%s",
            self.camera_id,
            self.camera_config.remote_worker_id or "<empty>",
            self.camera_config.source_type,
            self.camera_config.source_value or "<empty>",
            distributed_runtime.preview_fps,
            config_hash,
        )
        worker = self._pipeline_worker
        thread = self._pipeline_thread
        if worker is None or thread is None or not thread.is_alive():
            return
        if previous_source != next_source:
            LOGGER.info("Camera source changed; restarting pipeline camera_id=%s", self.camera_id)
            self._stop_pipeline()
            self._start_pipeline()
            return
        worker.update_configs(
            CameraConfig.from_dict(self.camera_config.to_dict()),
            self.project_config.venue_map,
        )
        worker.set_playback_sync(
            self._playback_sync,
            self._file_playback_started_wall_time(),
            self._session_sync_mode,
        )

    def _stop_pipeline(self) -> None:
        worker = self._pipeline_worker
        thread = self._pipeline_thread
        if worker is not None or thread is not None:
            LOGGER.info("Stopping camera pipeline camera_id=%s", self.camera_id)
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
        LOGGER.info("%s: %s", self.camera_id, status_text)
        self._send_message(
            {
                "type": MESSAGE_STATUS,
                "camera_id": self.camera_id,
                "status_text": status_text,
            }
        )

    def _handle_error(self, message: str) -> None:
        LOGGER.error("%s: %s", self.camera_id, message)
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

    def _send_message(self, message: dict[str, object]) -> bool:
        if not self._connected:
            return False
        with self._socket_lock:
            sock = self._socket
            if sock is None:
                return False
            try:
                sock.sendall(pack_message(message))
            except OSError as exc:
                LOGGER.warning("Failed to send %s message to server: %s", message.get("type", "<unknown>"), exc)
                LOGGER.debug("Worker send failure details", exc_info=True)
                self._connected = False
                return False
        return True

    def _reload_project(self) -> None:
        self.project_config = self.config_repo.load_project()
        self.camera_config = self._camera_config(self.project_config)
        self._preview_interval_s = 1.0 / max(self.project_config.distributed_runtime.preview_fps, 0.5)

    def _camera_config(self, project_config: ProjectConfig) -> CameraConfig:
        for camera in project_config.cameras:
            if camera.camera_id == self.camera_id:
                return CameraConfig.from_dict(camera.to_dict())
        return CameraConfig(camera_id=self.camera_id, runtime_mode="remote", enabled=True)

    @staticmethod
    def _source_signature(camera_config: CameraConfig) -> tuple[str, str, bool]:
        return camera_config.source_type, camera_config.source_value, camera_config.loop_file

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
    log_level: str = "INFO",
) -> int:
    configure_worker_logging(log_level)
    worker = DistributedCameraWorker(
        project_root=project_root,
        camera_id=camera_id,
        server_host=server_host,
        server_port=server_port,
    )
    worker.run_forever()
    return 0


def _connection_failure_hint(exc: OSError, port: int) -> str:
    if isinstance(exc, ConnectionRefusedError):
        return (
            f"the host responded but no server accepted TCP port {port}; "
            "check that server mode is running/listening and that the firewall allows it"
        )
    if isinstance(exc, TimeoutError):
        return "the connection timed out; check the IP address, routing, and firewall"
    if isinstance(exc, socket.gaierror):
        return "the host name could not be resolved"
    if isinstance(exc, OSError) and exc.errno in {101, 113}:
        return "the network or host is unreachable; check both PCs are on the same LAN/VPN"
    return ""
