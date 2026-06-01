from __future__ import annotations

import struct
from collections.abc import Iterable
from typing import Any

MESSAGE_HELLO = "hello"
MESSAGE_HEARTBEAT = "heartbeat"
MESSAGE_WORKER_CONFIG = "worker_config"
MESSAGE_START_SESSION = "start_session"
MESSAGE_STOP_SESSION = "stop_session"
MESSAGE_CAMERA_PACKET = "camera_packet"
MESSAGE_PREVIEW_FRAME = "preview_frame"
MESSAGE_STATUS = "status"
MESSAGE_ERROR = "error"


class DistributedProtocolError(RuntimeError):
    pass


def pack_message(message: dict[str, Any]) -> bytes:
    payload = _pack_value(message)
    return struct.pack(">I", len(payload)) + payload


def unpack_from_buffer(buffer: bytearray) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    while len(buffer) >= 4:
        payload_size = struct.unpack(">I", bytes(buffer[:4]))[0]
        if len(buffer) < 4 + payload_size:
            break
        payload = bytes(buffer[4 : 4 + payload_size])
        del buffer[: 4 + payload_size]
        decoded = _unpack_value(payload)
        if not isinstance(decoded, dict):
            raise DistributedProtocolError("Top-level distributed message must be a map.")
        messages.append(decoded)
    return messages


def _pack_value(value: Any) -> bytes:
    if value is None:
        return b"\xc0"
    if value is False:
        return b"\xc2"
    if value is True:
        return b"\xc3"
    if isinstance(value, int) and not isinstance(value, bool):
        return _pack_int(value)
    if isinstance(value, float):
        return b"\xcb" + struct.pack(">d", value)
    if isinstance(value, str):
        return _pack_str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _pack_bin(bytes(value))
    if isinstance(value, dict):
        return _pack_map(value)
    if isinstance(value, (list, tuple)):
        return _pack_array(value)
    raise DistributedProtocolError(f"Unsupported MessagePack value type: {type(value)!r}")


def _pack_int(value: int) -> bytes:
    if 0 <= value <= 0x7F:
        return bytes([value])
    if -32 <= value < 0:
        return struct.pack("b", value)
    if 0 <= value <= 0xFF:
        return b"\xcc" + struct.pack(">B", value)
    if 0 <= value <= 0xFFFF:
        return b"\xcd" + struct.pack(">H", value)
    if 0 <= value <= 0xFFFFFFFF:
        return b"\xce" + struct.pack(">I", value)
    if 0 <= value <= 0xFFFFFFFFFFFFFFFF:
        return b"\xcf" + struct.pack(">Q", value)
    if -0x80 <= value < 0:
        return b"\xd0" + struct.pack(">b", value)
    if -0x8000 <= value < -0x80:
        return b"\xd1" + struct.pack(">h", value)
    if -0x80000000 <= value < -0x8000:
        return b"\xd2" + struct.pack(">i", value)
    if -0x8000000000000000 <= value < -0x80000000:
        return b"\xd3" + struct.pack(">q", value)
    raise DistributedProtocolError(f"Integer out of supported range: {value}")


def _pack_str(value: str) -> bytes:
    encoded = value.encode("utf-8")
    length = len(encoded)
    if length <= 31:
        return bytes([0xA0 | length]) + encoded
    if length <= 0xFF:
        return b"\xd9" + struct.pack(">B", length) + encoded
    if length <= 0xFFFF:
        return b"\xda" + struct.pack(">H", length) + encoded
    return b"\xdb" + struct.pack(">I", length) + encoded


def _pack_bin(value: bytes) -> bytes:
    length = len(value)
    if length <= 0xFF:
        return b"\xc4" + struct.pack(">B", length) + value
    if length <= 0xFFFF:
        return b"\xc5" + struct.pack(">H", length) + value
    return b"\xc6" + struct.pack(">I", length) + value


def _pack_array(values: Iterable[Any]) -> bytes:
    encoded_items = [_pack_value(item) for item in values]
    length = len(encoded_items)
    prefix: bytes
    if length <= 15:
        prefix = bytes([0x90 | length])
    elif length <= 0xFFFF:
        prefix = b"\xdc" + struct.pack(">H", length)
    else:
        prefix = b"\xdd" + struct.pack(">I", length)
    return prefix + b"".join(encoded_items)


def _pack_map(values: dict[Any, Any]) -> bytes:
    encoded_items: list[bytes] = []
    length = len(values)
    if length <= 15:
        prefix = bytes([0x80 | length])
    elif length <= 0xFFFF:
        prefix = b"\xde" + struct.pack(">H", length)
    else:
        prefix = b"\xdf" + struct.pack(">I", length)
    for key, value in values.items():
        encoded_items.append(_pack_value(key))
        encoded_items.append(_pack_value(value))
    return prefix + b"".join(encoded_items)


def _unpack_value(payload: bytes) -> Any:
    value, offset = _decode(payload, 0)
    if offset != len(payload):
        raise DistributedProtocolError("Trailing bytes remain after MessagePack decode.")
    return value


def _decode(payload: bytes, offset: int) -> tuple[Any, int]:
    if offset >= len(payload):
        raise DistributedProtocolError("Unexpected end of payload.")
    prefix = payload[offset]
    offset += 1

    if prefix <= 0x7F:
        return prefix, offset
    if prefix >= 0xE0:
        return struct.unpack("b", bytes([prefix]))[0], offset
    if 0xA0 <= prefix <= 0xBF:
        length = prefix & 0x1F
        return _read_str(payload, offset, length)
    if 0x90 <= prefix <= 0x9F:
        length = prefix & 0x0F
        return _read_array(payload, offset, length)
    if 0x80 <= prefix <= 0x8F:
        length = prefix & 0x0F
        return _read_map(payload, offset, length)

    if prefix == 0xC0:
        return None, offset
    if prefix == 0xC2:
        return False, offset
    if prefix == 0xC3:
        return True, offset
    if prefix == 0xC4:
        return _read_bin(payload, offset, ">B", 1)
    if prefix == 0xC5:
        return _read_bin(payload, offset, ">H", 2)
    if prefix == 0xC6:
        return _read_bin(payload, offset, ">I", 4)
    if prefix == 0xCB:
        return _read_struct(payload, offset, ">d", 8)
    if prefix == 0xCC:
        return _read_struct(payload, offset, ">B", 1)
    if prefix == 0xCD:
        return _read_struct(payload, offset, ">H", 2)
    if prefix == 0xCE:
        return _read_struct(payload, offset, ">I", 4)
    if prefix == 0xCF:
        return _read_struct(payload, offset, ">Q", 8)
    if prefix == 0xD0:
        return _read_struct(payload, offset, ">b", 1)
    if prefix == 0xD1:
        return _read_struct(payload, offset, ">h", 2)
    if prefix == 0xD2:
        return _read_struct(payload, offset, ">i", 4)
    if prefix == 0xD3:
        return _read_struct(payload, offset, ">q", 8)
    if prefix == 0xD9:
        return _read_str_with_size(payload, offset, ">B", 1)
    if prefix == 0xDA:
        return _read_str_with_size(payload, offset, ">H", 2)
    if prefix == 0xDB:
        return _read_str_with_size(payload, offset, ">I", 4)
    if prefix == 0xDC:
        return _read_collection(payload, offset, ">H", 2, _read_array)
    if prefix == 0xDD:
        return _read_collection(payload, offset, ">I", 4, _read_array)
    if prefix == 0xDE:
        return _read_collection(payload, offset, ">H", 2, _read_map)
    if prefix == 0xDF:
        return _read_collection(payload, offset, ">I", 4, _read_map)
    raise DistributedProtocolError(f"Unsupported MessagePack prefix: 0x{prefix:02x}")


def _read_struct(payload: bytes, offset: int, fmt: str, size: int) -> tuple[Any, int]:
    end = offset + size
    if end > len(payload):
        raise DistributedProtocolError("Unexpected end of payload.")
    return struct.unpack(fmt, payload[offset:end])[0], end


def _read_str_with_size(payload: bytes, offset: int, fmt: str, size: int) -> tuple[str, int]:
    length, offset = _read_struct(payload, offset, fmt, size)
    return _read_str(payload, offset, length)


def _read_str(payload: bytes, offset: int, length: int) -> tuple[str, int]:
    end = offset + length
    if end > len(payload):
        raise DistributedProtocolError("Unexpected end of payload.")
    return payload[offset:end].decode("utf-8"), end


def _read_bin(payload: bytes, offset: int, fmt: str, size: int) -> tuple[bytes, int]:
    length, offset = _read_struct(payload, offset, fmt, size)
    end = offset + length
    if end > len(payload):
        raise DistributedProtocolError("Unexpected end of payload.")
    return payload[offset:end], end


def _read_collection(payload: bytes, offset: int, fmt: str, size: int, reader):
    length, offset = _read_struct(payload, offset, fmt, size)
    return reader(payload, offset, length)


def _read_array(payload: bytes, offset: int, length: int) -> tuple[list[Any], int]:
    values: list[Any] = []
    for _ in range(length):
        value, offset = _decode(payload, offset)
        values.append(value)
    return values, offset


def _read_map(payload: bytes, offset: int, length: int) -> tuple[dict[Any, Any], int]:
    values: dict[Any, Any] = {}
    for _ in range(length):
        key, offset = _decode(payload, offset)
        value, offset = _decode(payload, offset)
        values[key] = value
    return values, offset
