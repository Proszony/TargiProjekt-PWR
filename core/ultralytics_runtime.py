from __future__ import annotations

import os

os.environ.setdefault("YOLO_OFFLINE", "true")

from ultralytics import YOLO  # noqa: E402

__all__ = ["YOLO"]
