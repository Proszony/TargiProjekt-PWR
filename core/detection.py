from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from core.calibration import bottom_center, invert_homography, project_point, project_points
from core.models import CameraConfig, Detection, LocalTrack, VenueMapConfig
from core.zones import zone_color


class YoloPersonDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        inference_size: int = 320,
        use_augmentation: bool = True,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.inference_size = inference_size
        self.use_augmentation = use_augmentation
        self._model: YOLO | None = None

    def detect(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        camera_config: CameraConfig,
    ) -> list[Detection]:
        model = self._ensure_model()
        results = model(
            frame_bgr,
            classes=[0],
            conf=self.confidence,
            imgsz=self.inference_size,
            augment=self.use_augmentation,
            verbose=False,
        )

        detections: list[Detection] = []
        homography = camera_config.homography_image_to_world
        for box in results[0].boxes:
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
                    appearance_descriptor=appearance_descriptor,
                )
            )
        return detections

    def _ensure_model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    def set_model_path(self, model_path: str) -> None:
        if model_path == self.model_path:
            return
        self.model_path = model_path
        self._model = None


def annotate_frame(
    frame_bgr: np.ndarray,
    tracks: dict[int, LocalTrack],
    camera_config: CameraConfig,
    venue_map: VenueMapConfig,
    *,
    render_timestamp: float | None = None,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    inverse_homography = invert_homography(camera_config.homography_image_to_world)

    if inverse_homography is not None:
        for zone in venue_map.zones:
            polygon = project_points(inverse_homography, zone.polygon_world)
            if len(polygon) < 3:
                continue
            points = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
            color = _hex_to_bgr(zone_color(zone.kind))
            cv2.polylines(annotated, [points], isClosed=True, color=color, thickness=2)
            label_position = tuple(points[0][0])
            cv2.putText(
                annotated,
                zone.name,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    for track in tracks.values():
        if track.current_bbox_xyxy is None:
            continue
        if (
            render_timestamp is not None
            and render_timestamp - track.last_seen_ts > camera_config.bbox_publish_ttl_s
        ):
            continue
        x1, y1, x2, y2 = track.current_bbox_xyxy
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        if track.ground_anchor_image is not None:
            anchor = (int(track.ground_anchor_image[0]), int(track.ground_anchor_image[1]))
            cv2.circle(annotated, anchor, 5, (255, 255, 0), -1)

    return annotated


def qimage_from_bgr(frame_bgr: np.ndarray):
    from PySide6.QtGui import QImage

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width, channels = frame_rgb.shape
    bytes_per_line = channels * width
    return QImage(
        frame_rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format_RGB888,
    ).copy()


def _hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        return (200, 200, 200)
    return (int(value[4:6], 16), int(value[2:4], 16), int(value[0:2], 16))


def _extract_appearance_descriptor(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> list[float]:
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    x2 = max(x1 + 1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(y1 + 1, min(y2, height))
    box_height = y2 - y1
    if box_height <= 1 or x2 - x1 <= 1:
        return []

    torso_y1 = y1 + int(box_height * 0.2)
    torso_y2 = y1 + int(box_height * 0.65)
    torso_y1 = max(y1, min(torso_y1, y2 - 1))
    torso_y2 = max(torso_y1 + 1, min(torso_y2, y2))
    crop = frame_bgr[torso_y1:torso_y2, x1:x2]
    if crop.size == 0:
        return []

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
    histogram = cv2.normalize(histogram, histogram, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return histogram.flatten().astype(np.float32).tolist()


def _reject_fragmentary_border_detection(
    frame_shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    confidence: float,
) -> bool:
    frame_height, frame_width = frame_shape
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    area = width * height
    aspect_ratio = height / width
    border_margin_x = max(8, int(frame_width * 0.01))
    border_margin_y = max(8, int(frame_height * 0.01))
    touches_border = (
        x1 <= border_margin_x
        or y1 <= border_margin_y
        or x2 >= frame_width - border_margin_x
        or y2 >= frame_height - border_margin_y
    )
    if not touches_border:
        return False
    if confidence >= 0.6:
        return False
    if width <= max(24, int(frame_width * 0.04)):
        return True
    if area <= int(frame_width * frame_height * 0.015):
        return True
    if aspect_ratio >= 4.2:
        return True
    return False
