from __future__ import annotations

from core import runtime_defaults as rd


def camera_color(display_order: int, camera_id: str) -> str:
    palette = rd.DEFAULT_CAMERA_COLOR_PALETTE
    if not palette:
        return "#2563eb"
    seed = display_order if display_order >= 0 else sum(ord(char) for char in camera_id)
    return palette[seed % len(palette)]
