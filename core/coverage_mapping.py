from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None

from core.models import CoverageProposal, Point


@dataclass(slots=True)
class CoverageMappingOptions:
    max_vertices: int = 8
    min_vertices: int = 4
    downscale_width: int = 640


def propose_coverage_polygon_image(
    frame_bgr: np.ndarray,
    frame_size: tuple[int, int],
    options: CoverageMappingOptions | None = None,
) -> CoverageProposal:
    options = options or CoverageMappingOptions()
    frame_width, frame_height = frame_size
    if frame_width <= 0 or frame_height <= 0:
        return CoverageProposal(
            polygon_image=[],
            confidence=0.0,
            warnings=["Coverage proposal failed: invalid frame size."],
            method="invalid-frame",
        )
    if cv2 is None or frame_bgr.size == 0:
        return CoverageProposal(
            polygon_image=default_coverage_polygon_image(frame_width, frame_height),
            confidence=0.25,
            warnings=["Coverage proposal fell back to default polygon."],
            method="fallback",
        )

    content_rect = _detect_content_rect(frame_bgr)
    scaled, scale = _downscale_frame(frame_bgr, options.downscale_width)
    proposal = _propose_from_mask(scaled, options, content_rect, scale)
    if proposal is not None:
        return proposal
    return CoverageProposal(
        polygon_image=default_coverage_polygon_image(frame_width, frame_height, content_rect=content_rect),
        confidence=0.25,
        warnings=["Coverage auto-detection failed; using fallback polygon."],
        method="fallback",
    )


def default_coverage_polygon_image(
    frame_width: int,
    frame_height: int,
    *,
    content_rect: tuple[int, int, int, int] | None = None,
) -> list[Point]:
    if content_rect is None:
        left, top, right, bottom = 0, 0, frame_width - 1, frame_height - 1
    else:
        left, top, right, bottom = content_rect
    height = max(bottom - top, 1)
    upper_y = top + int(height * 0.58)
    inset = max(int((right - left) * 0.08), 8)
    return [
        (float(left), float(bottom)),
        (float(right), float(bottom)),
        (float(max(left, right - inset)), float(upper_y)),
        (float(min(right, left + inset)), float(upper_y)),
    ]


def _detect_content_rect(frame_bgr: np.ndarray) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > 10
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        height, width = gray.shape
        return (0, 0, width - 1, height - 1)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _downscale_frame(frame_bgr: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    height, width = frame_bgr.shape[:2]
    if width <= max_width:
        return frame_bgr.copy(), 1.0
    scale = max_width / float(width)
    resized = cv2.resize(frame_bgr, (int(round(width * scale)), int(round(height * scale))))
    return resized, scale


def _propose_from_mask(
    frame_bgr: np.ndarray,
    options: CoverageMappingOptions,
    content_rect: tuple[int, int, int, int],
    scale: float,
) -> CoverageProposal | None:
    height, width = frame_bgr.shape[:2]
    rect = (
        int(content_rect[0] * scale),
        int(content_rect[1] * scale),
        int(content_rect[2] * scale),
        int(content_rect[3] * scale),
    )
    mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    left, top, right, bottom = rect
    left = max(0, min(left, width - 1))
    right = max(left + 1, min(right, width - 1))
    top = max(0, min(top, height - 1))
    bottom = max(top + 1, min(bottom, height - 1))
    mask[top:bottom + 1, left:right + 1] = cv2.GC_PR_BGD

    floor_seed_top = top + int((bottom - top) * 0.35)
    floor_seed_left = left + int((right - left) * 0.12)
    floor_seed_right = right - int((right - left) * 0.12)
    mask[floor_seed_top:bottom + 1, floor_seed_left:floor_seed_right + 1] = cv2.GC_PR_FGD
    mask[bottom - max(int((bottom - top) * 0.18), 12):bottom + 1, left:right + 1] = cv2.GC_FGD
    mask[: max(top + int(height * 0.30), top + 1), :] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(frame_bgr, mask, None, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    foreground = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    lower_half_y = height * 0.5
    best_contour = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0.0:
            continue
        points = contour.reshape(-1, 2)
        if np.max(points[:, 1]) < lower_half_y:
            continue
        if area > best_area:
            best_area = area
            best_contour = contour
    if best_contour is None:
        return None

    perimeter = cv2.arcLength(best_contour, True)
    epsilon = max(0.01 * perimeter, 4.0)
    approx = cv2.approxPolyDP(best_contour, epsilon, True)
    polygon = [(float(point[0][0] / scale), float(point[0][1] / scale)) for point in approx]
    polygon = _normalize_polygon_vertices(polygon, options.min_vertices, options.max_vertices)
    if len(polygon) < 3:
        return None

    x, y, w, h = cv2.boundingRect(best_contour)
    confidence = min(0.95, max(0.35, best_area / float(max(width * height, 1))))
    warnings: list[str] = []
    if len(polygon) < options.min_vertices:
        warnings.append("Coverage polygon is too simple; review manually.")
    return CoverageProposal(
        polygon_image=polygon,
        confidence=confidence,
        warnings=warnings,
        mask_bbox_xyxy=(
            int(round(x / scale)),
            int(round(y / scale)),
            int(round((x + w) / scale)),
            int(round((y + h) / scale)),
        ),
        method="grabcut-floor-mask",
    )


def _normalize_polygon_vertices(
    polygon: list[Point],
    min_vertices: int,
    max_vertices: int,
) -> list[Point]:
    if len(polygon) <= max_vertices:
        return polygon
    step = len(polygon) / float(max_vertices)
    return [polygon[int(round(index * step)) % len(polygon)] for index in range(max_vertices)]
