from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from core.calibration import bottom_center, project_point
from core.detection import _extract_appearance_descriptor, _reject_fragmentary_border_detection
from core.models import CameraConfig, Detection


class UltralyticsTrackerAdapter:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        inference_size: int = 640,
        use_augmentation: bool = False,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.inference_size = inference_size
        self.use_augmentation = use_augmentation
        self._model: YOLO | None = None

    def track(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        camera_config: CameraConfig,
    ) -> list[Detection]:
        model = self._ensure_model()
        tracker_path = self._resolve_tracker_path(camera_config.tracker_config_path)
        results = model.track(
            source=frame_bgr,
            stream=False,
            persist=camera_config.tracker_persist,
            conf=self.confidence,
            imgsz=self.inference_size,
            augment=self.use_augmentation,
            classes=[0],
            tracker=str(tracker_path),
            verbose=False,
        )
        if not results:
            return []

        detections: list[Detection] = []
        homography = camera_config.homography_image_to_world
        for box in results[0].boxes:
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0].item())
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            bbox = (x1, y1, x2, y2)
            confidence = float(box.conf[0].item())
            if _reject_fragmentary_border_detection(frame_bgr.shape[:2], bbox, confidence):
                continue
            anchor_image = bottom_center(bbox)
            anchor_world = project_point(homography, anchor_image)
            appearance_descriptor = _extract_appearance_descriptor(frame_bgr, bbox)
            detections.append(
                Detection(
                    camera_id=camera_config.camera_id,
                    timestamp=timestamp,
                    person_bbox_xyxy=bbox,
                    confidence=confidence,
                    ground_anchor_image=anchor_image,
                    ground_anchor_world=anchor_world,
                    track_id=track_id,
                    appearance_descriptor=appearance_descriptor,
                )
            )
        return detections

    def reset(self) -> None:
        self._model = None

    def set_model_path(self, model_path: str) -> None:
        if model_path == self.model_path:
            return
        self.model_path = model_path
        self._model = None

    def _ensure_model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    @staticmethod
    def _resolve_tracker_path(path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return Path.cwd() / candidate
