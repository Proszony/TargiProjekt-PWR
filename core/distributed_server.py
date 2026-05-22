from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QObject, Signal

from core.distributed_protocol import (
    DistributedProtocolError,
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
    camera_tracking_packet_from_network_dict,
    preview_frame_from_network_dict,
)
from core.models import CameraConfig, ProjectConfig


@dataclass(slots=True)
class _WorkerConnection:
    sock: socket.socket
    address: tuple[str, int]
    worker_id: str
    camera_id: str
    protocol_version: str
    config_hash: str
    config_mismatch: bool
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_heartbeat_at: float = field(default_factory=time.monotonic)


class DistributedRuntimeServer(QObject):
    camera_packet_received = Signal(object)
    preview_frame_received = Signal(object)
    camera_status_changed = Signal(str, str)
    camera_error = Signal(str, str)

    def __init__(self, project_config: ProjectConfig) -> None:
        super().__init__()
        self.project_config = ProjectConfig.from_dict(project_config.to_dict())
        self._lock = threading.Lock()
        self._server_socket: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None
        self._connections_by_camera: dict[str, _WorkerConnection] = {}
        self._running = False
        self._session_active = False
        self._session_sync_mode = "all_live_unsynced"

    def update_project_config(self, project_config: ProjectConfig) -> None:
        self.project_config = ProjectConfig.from_dict(project_config.to_dict())
        stale_connections: list[_WorkerConnection] = []
        remote_ids = self.remote_camera_ids()
        with self._lock:
            for camera_id, connection in self._connections_by_camera.items():
                if camera_id not in remote_ids:
                    stale_connections.append(connection)
        for connection in stale_connections:
            self._drop_connection(connection, "removed from project")

    def start(self) -> None:
        if self._running:
            return
        bind_host = self.project_config.distributed_runtime.server_bind_host
        bind_port = self.project_config.distributed_runtime.server_port
        self._server_socket = socket.create_server((bind_host, bind_port), reuse_port=False)
        self._server_socket.settimeout(0.5)
        self._running = True
        self._accept_thread = threading.Thread(target=self._accept_loop, name="distributed-accept", daemon=True)
        self._monitor_thread = threading.Thread(target=self._monitor_loop, name="distributed-monitor", daemon=True)
        self._accept_thread.start()
        self._monitor_thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._session_active = False
        server_socket = self._server_socket
        self._server_socket = None
        if server_socket is not None:
            try:
                server_socket.close()
            except OSError:
                pass
        with self._lock:
            connections = list(self._connections_by_camera.values())
            self._connections_by_camera.clear()
        for connection in connections:
            self._close_socket(connection.sock)

    def start_session(self, session_sync_mode: str) -> None:
        self._session_active = True
        self._session_sync_mode = session_sync_mode
        with self._lock:
            connections = list(self._connections_by_camera.values())
        for connection in connections:
            self._send_message(
                connection,
                {
                    "type": MESSAGE_START_SESSION,
                    "session_sync_mode": session_sync_mode,
                    "playback_sync": self.project_config.playback_sync.to_dict(),
                },
            )

    def stop_session(self) -> None:
        self._session_active = False
        with self._lock:
            connections = list(self._connections_by_camera.values())
        for connection in connections:
            self._send_message(connection, {"type": MESSAGE_STOP_SESSION})

    def remote_camera_ids(self) -> set[str]:
        return {
            camera.camera_id
            for camera in self.project_config.cameras
            if camera.enabled and camera.runtime_mode == "remote"
        }

    def unavailable_camera_ids(self) -> set[str]:
        with self._lock:
            connected = set(self._connections_by_camera)
        return self.remote_camera_ids() - connected

    def has_remote_cameras(self) -> bool:
        return bool(self.remote_camera_ids())

    def _accept_loop(self) -> None:
        while self._running:
            server_socket = self._server_socket
            if server_socket is None:
                return
            try:
                client_socket, address = server_socket.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._running:
                    time.sleep(0.1)
                continue
            worker_thread = threading.Thread(
                target=self._handle_client,
                args=(client_socket, address),
                name=f"distributed-client-{address[0]}:{address[1]}",
                daemon=True,
            )
            worker_thread.start()

    def _monitor_loop(self) -> None:
        while self._running:
            timeout_s = self.project_config.distributed_runtime.worker_timeout_s
            now = time.monotonic()
            timed_out: list[_WorkerConnection] = []
            with self._lock:
                for connection in self._connections_by_camera.values():
                    if now - connection.last_heartbeat_at > timeout_s:
                        timed_out.append(connection)
            for connection in timed_out:
                self._drop_connection(connection, "timed out")
            time.sleep(0.5)

    def _handle_client(self, client_socket: socket.socket, address: tuple[str, int]) -> None:
        client_socket.settimeout(0.5)
        buffer = bytearray()
        connection: _WorkerConnection | None = None
        try:
            while self._running:
                try:
                    chunk = client_socket.recv(65536)
                except TimeoutError:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buffer.extend(chunk)
                for message in unpack_from_buffer(buffer):
                    if connection is None:
                        connection = self._accept_hello(client_socket, address, message)
                        if connection is None:
                            return
                        continue
                    self._handle_message(connection, message)
        except DistributedProtocolError as exc:
            if connection is not None:
                self.camera_error.emit(connection.camera_id, f"Protocol error: {exc}")
        finally:
            if connection is not None:
                self._drop_connection(connection, "waiting for worker")
            else:
                self._close_socket(client_socket)

    def _accept_hello(
        self,
        client_socket: socket.socket,
        address: tuple[str, int],
        message: dict[str, Any],
    ) -> _WorkerConnection | None:
        if message.get("type") != MESSAGE_HELLO:
            self._close_socket(client_socket)
            return None
        protocol_version = str(message.get("protocol_version", ""))
        if protocol_version != self.project_config.distributed_runtime.protocol_version:
            self._send_raw_message(
                client_socket,
                {
                    "type": MESSAGE_ERROR,
                    "camera_id": str(message.get("camera_id", "")),
                    "message": "protocol version mismatch",
                },
            )
            self._close_socket(client_socket)
            return None
        camera_id = str(message.get("camera_id", ""))
        camera_config = self._camera_config(camera_id)
        if camera_config is None or camera_config.runtime_mode != "remote":
            self._send_raw_message(
                client_socket,
                {
                    "type": MESSAGE_ERROR,
                    "camera_id": camera_id,
                    "message": "camera not configured for remote runtime",
                },
            )
            self._close_socket(client_socket)
            return None
        worker_id = str(message.get("worker_id", ""))
        if camera_config.remote_worker_id and camera_config.remote_worker_id != worker_id:
            self._send_raw_message(
                client_socket,
                {
                    "type": MESSAGE_ERROR,
                    "camera_id": camera_id,
                    "message": "worker identity mismatch",
                },
            )
            self._close_socket(client_socket)
            return None
        config_hash = str(message.get("config_hash", ""))
        with self._lock:
            if camera_id in self._connections_by_camera:
                self._send_raw_message(
                    client_socket,
                    {
                        "type": MESSAGE_ERROR,
                        "camera_id": camera_id,
                        "message": "duplicate camera connection",
                    },
                )
                self._close_socket(client_socket)
                return None
            connection = _WorkerConnection(
                sock=client_socket,
                address=address,
                worker_id=worker_id,
                camera_id=camera_id,
                protocol_version=protocol_version,
                config_hash=config_hash,
                config_mismatch=(config_hash != camera_config_sha256(camera_config)),
            )
            self._connections_by_camera[camera_id] = connection
        status = "remote connected"
        if connection.config_mismatch:
            status += " | config mismatch"
        self.camera_status_changed.emit(camera_id, status)
        if self._session_active:
            self._send_message(
                connection,
                {
                    "type": MESSAGE_START_SESSION,
                    "session_sync_mode": self._session_sync_mode,
                    "playback_sync": self.project_config.playback_sync.to_dict(),
                },
            )
        return connection

    def _handle_message(self, connection: _WorkerConnection, message: dict[str, Any]) -> None:
        connection.last_heartbeat_at = time.monotonic()
        message_type = str(message.get("type", ""))
        if message_type == MESSAGE_HEARTBEAT:
            return
        if message_type == MESSAGE_CAMERA_PACKET:
            payload = message.get("payload", {})
            if isinstance(payload, dict):
                self.camera_packet_received.emit(camera_tracking_packet_from_network_dict(payload))
            return
        if message_type == MESSAGE_PREVIEW_FRAME:
            payload = message.get("payload", {})
            if isinstance(payload, dict):
                self.preview_frame_received.emit(preview_frame_from_network_dict(payload))
            return
        if message_type == MESSAGE_STATUS:
            self.camera_status_changed.emit(connection.camera_id, str(message.get("status_text", "")))
            return
        if message_type == MESSAGE_ERROR:
            self.camera_error.emit(connection.camera_id, str(message.get("message", "")))
            return
        raise DistributedProtocolError(f"Unsupported message type '{message_type}'.")

    def _drop_connection(self, connection: _WorkerConnection, status_text: str) -> None:
        removed = False
        with self._lock:
            current = self._connections_by_camera.get(connection.camera_id)
            if current is connection:
                self._connections_by_camera.pop(connection.camera_id, None)
                removed = True
        self._close_socket(connection.sock)
        if removed and self._running:
            self.camera_status_changed.emit(connection.camera_id, status_text)

    def _camera_config(self, camera_id: str) -> CameraConfig | None:
        for camera in self.project_config.cameras:
            if camera.camera_id == camera_id:
                return camera
        return None

    @staticmethod
    def _close_socket(sock: socket.socket) -> None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def _send_message(self, connection: _WorkerConnection, message: dict[str, Any]) -> None:
        with connection.lock:
            try:
                connection.sock.sendall(pack_message(message))
            except OSError:
                self._drop_connection(connection, "waiting for worker")

    @staticmethod
    def _send_raw_message(sock: socket.socket, message: dict[str, Any]) -> None:
        try:
            sock.sendall(pack_message(message))
        except OSError:
            pass
