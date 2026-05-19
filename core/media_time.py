from __future__ import annotations

from fractions import Fraction
from typing import Any


def resolve_stream_fps(value: Fraction | float | None) -> float | None:
    if value is None:
        return None
    try:
        fps = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if fps <= 0.0:
        return None
    return fps


def resolve_media_time(frame: Any, nominal_fps: float | None, frame_index: int) -> float | None:
    frame_time = getattr(frame, "time", None)
    if frame_time is not None:
        try:
            media_time_s = float(frame_time)
        except (TypeError, ValueError):
            media_time_s = None
        else:
            if media_time_s >= 0.0:
                return media_time_s

    pts = getattr(frame, "pts", None)
    time_base = getattr(frame, "time_base", None)
    if pts is not None and time_base is not None:
        try:
            media_time_s = float(pts * time_base)
        except (TypeError, ValueError, ZeroDivisionError):
            media_time_s = None
        else:
            if media_time_s >= 0.0:
                return media_time_s

    if nominal_fps and nominal_fps > 0.0:
        return frame_index / nominal_fps
    return None
