import socket
import time
import unittest

from core.distributed_protocol import MESSAGE_WORKER_CONFIG, pack_message, unpack_from_buffer
from core.distributed_serialization import camera_config_sha256
from core.models import CameraConfig, DistributedRuntimeConfig, ProjectConfig

try:
    from core.distributed_server import DistributedRuntimeServer
except ModuleNotFoundError as exc:  # pragma: no cover - optional UI dependency
    DistributedRuntimeServer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@unittest.skipIf(DistributedRuntimeServer is None, f"UI dependencies unavailable: {_IMPORT_ERROR}")
class DistributedServerTests(unittest.TestCase):
    def setUp(self) -> None:
        port = _free_port()
        self.camera = CameraConfig(
            camera_id="camera-1",
            runtime_mode="remote",
            remote_worker_id="edge-1",
        )
        self.project = ProjectConfig(
            cameras=[self.camera],
            distributed_runtime=DistributedRuntimeConfig(
                enabled=True,
                server_bind_host="127.0.0.1",
                server_port=port,
                protocol_version="1.0",
                worker_timeout_s=0.6,
            ),
        )
        self.server = DistributedRuntimeServer(self.project)
        self.server.start()

    def tearDown(self) -> None:
        self.server.stop()

    def _connect_and_send_hello(self, *, worker_id: str = "edge-1", protocol_version: str = "1.0") -> socket.socket:
        sock = socket.create_connection(
            (
                self.project.distributed_runtime.server_bind_host,
                self.project.distributed_runtime.server_port,
            ),
            timeout=2.0,
        )
        hello = {
            "type": "hello",
            "worker_id": worker_id,
            "camera_id": "camera-1",
            "protocol_version": protocol_version,
            "config_hash": camera_config_sha256(self.camera),
        }
        sock.sendall(pack_message(hello))
        return sock

    def _read_one_message(self, sock: socket.socket) -> dict[str, object]:
        sock.settimeout(2.0)
        buffer = bytearray()
        while True:
            buffer.extend(sock.recv(65536))
            messages = unpack_from_buffer(buffer)
            if messages:
                return messages[0]

    def test_protocol_mismatch_is_rejected(self) -> None:
        sock = self._connect_and_send_hello(protocol_version="2.0")
        payload = bytearray(sock.recv(4096))
        messages = unpack_from_buffer(payload)
        sock.close()

        self.assertEqual(messages[0]["type"], "error")
        self.assertIn("protocol version mismatch", messages[0]["message"])
        self.assertEqual(self.server.unavailable_camera_ids(), {"camera-1"})

    def test_duplicate_camera_connection_is_rejected(self) -> None:
        first = self._connect_and_send_hello()
        time.sleep(0.1)
        second = self._connect_and_send_hello()
        payload = bytearray(second.recv(4096))
        messages = unpack_from_buffer(payload)
        second.close()
        first.close()

        self.assertEqual(messages[0]["type"], "error")
        self.assertIn("duplicate camera connection", messages[0]["message"])

    def test_worker_config_is_sent_after_hello(self) -> None:
        sock = self._connect_and_send_hello(worker_id="")
        message = self._read_one_message(sock)
        sock.close()

        self.assertEqual(message["type"], MESSAGE_WORKER_CONFIG)
        payload = message["payload"]
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["camera_id"], "camera-1")
        self.assertEqual(payload["camera_config"]["remote_worker_id"], "edge-1")

    def test_project_update_pushes_worker_config(self) -> None:
        sock = self._connect_and_send_hello()
        self._read_one_message(sock)

        updated_camera = CameraConfig(
            camera_id="camera-1",
            runtime_mode="remote",
            remote_worker_id="edge-1",
            source_value="udp://127.0.0.1:7001",
        )
        self.server.update_project_config(
            ProjectConfig(
                cameras=[updated_camera],
                distributed_runtime=self.project.distributed_runtime,
            )
        )
        message = self._read_one_message(sock)
        sock.close()

        self.assertEqual(message["type"], MESSAGE_WORKER_CONFIG)
        self.assertEqual(message["payload"]["camera_config"]["source_value"], "udp://127.0.0.1:7001")

    def test_timeout_marks_remote_camera_unavailable(self) -> None:
        sock = self._connect_and_send_hello()
        time.sleep(0.2)
        self.assertEqual(self.server.unavailable_camera_ids(), set())
        time.sleep(0.8)
        sock.close()

        self.assertEqual(self.server.unavailable_camera_ids(), {"camera-1"})


if __name__ == "__main__":
    unittest.main()
