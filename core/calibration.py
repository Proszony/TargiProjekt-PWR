from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - exercised in tests without OpenCV
    cv2 = None

from core.models import Point


def bottom_center(bbox_xyxy: tuple[int, int, int, int]) -> Point:
    x1, _y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, float(y2))


def compute_homography(image_points: Iterable[Point], world_points: Iterable[Point]) -> np.ndarray | None:
    image = np.asarray(list(image_points), dtype=np.float32)
    world = np.asarray(list(world_points), dtype=np.float32)
    if len(image) < 4 or len(world) < 4 or len(image) != len(world):
        return None
    if cv2 is not None:
        matrix, _mask = cv2.findHomography(image, world, method=0)
        return matrix
    return _compute_homography_dlt(image, world)


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
    return [(float(x), float(y)) for x, y in projected]


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
