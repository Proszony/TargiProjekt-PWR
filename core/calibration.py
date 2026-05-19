from __future__ import annotations

import math
from typing import Iterable

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - exercised in tests without OpenCV
    cv2 = None

from core.models import (
    CalibrationResult,
    CameraConfig,
    CameraCoverageComputationResult,
    Point,
    WorldViewport,
    ZoneDefinition,
)

GOOD_RMSE_PX = 12.0
BAD_RMSE_PX = 30.0
MIN_USABLE_COVERAGE_AREA = 1.0
FALLBACK_VIEWPORT = WorldViewport(0.0, 0.0, 10.0, 10.0)


def bottom_center(bbox_xyxy: tuple[int, int, int, int]) -> Point:
    x1, _y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, float(y2))


def compute_homography(image_points: Iterable[Point], world_points: Iterable[Point]) -> np.ndarray | None:
    result = compute_homography_result(image_points, world_points)
    if result.homography_image_to_world is None:
        return None
    return np.asarray(result.homography_image_to_world, dtype=np.float64)


def compute_homography_result(
    image_points: Iterable[Point],
    world_points: Iterable[Point],
    *,
    use_ransac: bool = True,
) -> CalibrationResult:
    image = np.asarray(list(image_points), dtype=np.float32)
    world = np.asarray(list(world_points), dtype=np.float32)
    if len(image) < 4 or len(world) < 4 or len(image) != len(world):
        return CalibrationResult(
            used_anchor_count=min(len(image), len(world)),
            is_valid=False,
            warnings=["At least 4 matching anchors are required."],
        )
    matrix: np.ndarray | None
    if cv2 is not None:
        method = cv2.RANSAC if use_ransac else 0
        matrix, _mask = cv2.findHomography(image, world, method=method)
    else:
        matrix = _compute_homography_dlt(image, world)
    if matrix is None:
        return CalibrationResult(
            used_anchor_count=len(image),
            is_valid=False,
            warnings=["Homography computation failed."],
        )

    reprojection = project_points(invert_homography(matrix), world)
    errors = [
        math.dist((float(source[0]), float(source[1])), target)
        for source, target in zip(image, reprojection, strict=False)
    ]
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else None
    max_error = max(errors) if errors else None
    warnings: list[str] = []
    if rmse is not None:
        if rmse > BAD_RMSE_PX:
            warnings.append(f"High reprojection RMSE: {rmse:.1f}px.")
        elif rmse > GOOD_RMSE_PX:
            warnings.append(f"Moderate reprojection RMSE: {rmse:.1f}px.")

    return CalibrationResult(
        homography_image_to_world=homography_to_list(matrix),
        reprojection_rmse_px=rmse,
        max_reprojection_error_px=max_error,
        used_anchor_count=len(image),
        is_valid=rmse is None or rmse <= BAD_RMSE_PX,
        warnings=warnings,
    )


def project_point(matrix: np.ndarray | list[list[float]] | None, point: Point) -> Point | None:
    projected = project_points(matrix, [point])
    if not projected:
        return None
    return projected[0]


def project_points(
    matrix: np.ndarray | list[list[float]] | None,
    points: Iterable[Point],
) -> list[Point]:
    if matrix is None:
        return []
    array = np.asarray(matrix, dtype=np.float64)
    source = np.asarray(list(points), dtype=np.float64)
    if source.size == 0:
        return []
    if cv2 is not None:
        reshaped = source.reshape((-1, 1, 2))
        projected = cv2.perspectiveTransform(reshaped, array).reshape((-1, 2))
    else:
        ones = np.ones((source.shape[0], 1), dtype=np.float64)
        homogeneous = np.hstack([source, ones])
        projected_h = (array @ homogeneous.T).T
        projected = projected_h[:, :2] / projected_h[:, 2:3]
    return [(float(x), float(y)) for x, y in projected if math.isfinite(x) and math.isfinite(y)]


def invert_homography(matrix: np.ndarray | list[list[float]] | None) -> np.ndarray | None:
    if matrix is None:
        return None
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (3, 3):
        return None
    return np.linalg.inv(array)


def homography_to_list(matrix: np.ndarray | None) -> list[list[float]] | None:
    if matrix is None:
        return None
    return matrix.astype(float).tolist()


def project_image_polygon_to_world(
    matrix: np.ndarray | list[list[float]] | None,
    polygon_image: Iterable[Point],
) -> list[Point]:
    return project_points(matrix, polygon_image)


def sanitize_projected_polygon(polygon_world: Iterable[Point]) -> list[Point]:
    points = [
        (float(x), float(y))
        for x, y in polygon_world
        if math.isfinite(x) and math.isfinite(y)
    ]
    if len(points) < 3:
        return []
    points = _dedupe_consecutive(points)
    if len(points) < 3:
        return []
    if cv2 is not None:
        contour = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
        hull = cv2.convexHull(contour).reshape((-1, 2))
        sanitized = [(float(x), float(y)) for x, y in hull]
    else:
        sanitized = points
    if len(sanitized) < 3 or abs(_polygon_area(sanitized)) < MIN_USABLE_COVERAGE_AREA:
        return []
    if _polygon_area(sanitized) < 0:
        sanitized.reverse()
    return sanitized


def diagnose_coverage_polygon(
    raw_polygon_world: Iterable[Point],
    sanitized_polygon_world: Iterable[Point],
) -> list[str]:
    raw = list(raw_polygon_world)
    sanitized = list(sanitized_polygon_world)
    warnings: list[str] = []
    if len(raw) < 3:
        warnings.append("Coverage polygon could not be projected.")
        return warnings
    if len(sanitized) < 3:
        warnings.append("Projected coverage polygon is invalid after sanitization.")
        return warnings
    if len(raw) != len(sanitized):
        warnings.append("Projected coverage polygon was simplified during sanitization.")
    raw_area = abs(_polygon_area(raw))
    sanitized_area = abs(_polygon_area(sanitized))
    if raw_area > 0.0 and sanitized_area / raw_area < 0.55:
        warnings.append("Sanitized coverage differs significantly from raw projection.")
    span_x = max(point[0] for point in sanitized) - min(point[0] for point in sanitized)
    span_y = max(point[1] for point in sanitized) - min(point[1] for point in sanitized)
    if span_x < 0.25 or span_y < 0.25:
        warnings.append("Coverage footprint is very narrow in world space.")
    return warnings


def recompute_camera_coverage(camera_config: CameraConfig) -> CameraCoverageComputationResult:
    if camera_config.homography_image_to_world is None:
        return CameraCoverageComputationResult(
            warnings=["Coverage unavailable: homography is missing."],
            is_valid=False,
        )

    polygon_image = list(camera_config.coverage_polygon_image or [])
    if len(polygon_image) < 3:
        return CameraCoverageComputationResult(
            warnings=["Coverage unavailable: image-space coverage polygon is missing."],
            is_valid=False,
        )

    raw_polygon_world = project_image_polygon_to_world(
        camera_config.homography_image_to_world,
        polygon_image,
    )
    sanitized_polygon_world = sanitize_projected_polygon(raw_polygon_world)
    warnings = diagnose_coverage_polygon(raw_polygon_world, sanitized_polygon_world)
    warnings.extend(
        _diagnose_image_polygon(
            polygon_image,
            camera_config.frame_width,
            camera_config.frame_height,
        )
    )
    warnings = list(dict.fromkeys(warnings))
    return CameraCoverageComputationResult(
        raw_polygon_world=raw_polygon_world,
        sanitized_polygon_world=sanitized_polygon_world,
        warnings=warnings,
        is_valid=bool(camera_config.homography_image_to_world) and len(sanitized_polygon_world) >= 3,
    )


def compute_world_viewport(
    cameras: Iterable[CameraConfig],
    zones: Iterable[ZoneDefinition] = (),
    *,
    padding_ratio: float = 0.08,
    manual_override: WorldViewport | None = None,
) -> WorldViewport:
    if manual_override is not None:
        return manual_override
    polygons: list[list[Point]] = []
    for camera in cameras:
        if camera.calibration_valid and camera.coverage_polygon_world:
            polygons.append(list(camera.coverage_polygon_world))
    zone_polygons = [list(zone.polygon_world) for zone in zones if len(zone.polygon_world) >= 3]
    if zone_polygons:
        polygons.extend(zone_polygons)
    if not polygons:
        return FALLBACK_VIEWPORT

    xs = [point[0] for polygon in polygons for point in polygon]
    ys = [point[1] for polygon in polygons for point in polygon]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    pad_x = span_x * padding_ratio
    pad_y = span_y * padding_ratio
    return WorldViewport(
        min_x=min_x - pad_x,
        min_y=min_y - pad_y,
        max_x=max_x + pad_x,
        max_y=max_y + pad_y,
    )


def _compute_homography_dlt(image: np.ndarray, world: np.ndarray) -> np.ndarray | None:
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(image, world, strict=False):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, x * u, y * u, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, x * v, y * v, v])
    matrix = np.asarray(rows, dtype=np.float64)
    _u, _s, vh = np.linalg.svd(matrix)
    homography = vh[-1, :].reshape((3, 3))
    if homography[2, 2] == 0.0:
        return None
    return homography / homography[2, 2]


def _polygon_area(polygon: list[Point]) -> float:
    area = 0.0
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return area / 2.0


def _dedupe_consecutive(points: list[Point]) -> list[Point]:
    deduped: list[Point] = []
    for point in points:
        if not deduped or math.dist(point, deduped[-1]) > 1e-6:
            deduped.append(point)
    if len(deduped) >= 2 and math.dist(deduped[0], deduped[-1]) <= 1e-6:
        deduped.pop()
    return deduped


def _diagnose_image_polygon(
    polygon_image: list[Point],
    frame_width: int,
    frame_height: int,
) -> list[str]:
    warnings: list[str] = []
    if len(polygon_image) < 3:
        return warnings
    for x, y in polygon_image:
        if x < 0.0 or y < 0.0 or x > frame_width or y > frame_height:
            warnings.append("Coverage polygon extends outside the image bounds.")
            break
    if abs(_polygon_area(polygon_image)) <= 1.0:
        warnings.append("Coverage polygon is degenerate in image space.")
    return warnings
