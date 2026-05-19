from __future__ import annotations

import math

from core.models import CameraConfig, CameraOverlapGraph, CameraOverlapRelation, OverlapDedupConfig, Point


def build_camera_overlap_graph(
    cameras: list[CameraConfig],
    dedup_config: OverlapDedupConfig,
) -> CameraOverlapGraph:
    relations: dict[tuple[str, str], CameraOverlapRelation] = {}
    for index, camera_a in enumerate(cameras):
        polygon_a = (camera_a.coverage_polygon_world or []) if camera_a.calibration_valid else []
        for camera_b in cameras[index + 1 :]:
            polygon_b = (camera_b.coverage_polygon_world or []) if camera_b.calibration_valid else []
            intersection_polygon = coverage_intersection_polygon(polygon_a, polygon_b)
            overlap_area = abs(_polygon_area(intersection_polygon)) if len(intersection_polygon) >= 3 else 0.0
            min_distance = coverage_min_distance(polygon_a, polygon_b)
            manual_override = (
                camera_b.camera_id in camera_a.overlap_camera_ids
                or camera_a.camera_id in camera_b.overlap_camera_ids
            )
            auto_allowed = camera_a.allow_auto_overlap and camera_b.allow_auto_overlap
            is_adjacent = manual_override or (
                camera_a.calibration_valid
                and camera_b.calibration_valid
                and
                auto_allowed
                and (
                    overlap_area >= dedup_config.overlap_area_min_m2
                    or min_distance <= dedup_config.boundary_gap_m
                )
            )
            relation = CameraOverlapRelation(
                camera_a_id=camera_a.camera_id,
                camera_b_id=camera_b.camera_id,
                overlap_area_m2=overlap_area,
                min_boundary_distance_m=min_distance,
                is_adjacent=is_adjacent,
                intersection_polygon_world=list(intersection_polygon),
            )
            reverse = CameraOverlapRelation(
                camera_a_id=camera_b.camera_id,
                camera_b_id=camera_a.camera_id,
                overlap_area_m2=overlap_area,
                min_boundary_distance_m=min_distance,
                is_adjacent=is_adjacent,
                intersection_polygon_world=list(intersection_polygon),
            )
            relations[(camera_a.camera_id, camera_b.camera_id)] = relation
            relations[(camera_b.camera_id, camera_a.camera_id)] = reverse
    return CameraOverlapGraph(relations=relations)


def coverage_overlap_area(polygon_a: list[Point], polygon_b: list[Point]) -> float:
    intersection = coverage_intersection_polygon(polygon_a, polygon_b)
    if len(intersection) < 3:
        return 0.0
    return abs(_polygon_area(intersection))


def coverage_intersection_polygon(polygon_a: list[Point], polygon_b: list[Point]) -> list[Point]:
    if len(polygon_a) < 3 or len(polygon_b) < 3:
        return []
    clipped = _clip_convex_polygon(polygon_a, polygon_b)
    if len(clipped) < 3:
        return []
    return clipped


def coverage_min_distance(polygon_a: list[Point], polygon_b: list[Point]) -> float:
    if not polygon_a or not polygon_b:
        return float("inf")
    if coverage_overlap_area(polygon_a, polygon_b) > 0.0:
        return 0.0
    best = float("inf")
    for segment_a in _segments(polygon_a):
        for segment_b in _segments(polygon_b):
            best = min(best, _segment_distance(segment_a[0], segment_a[1], segment_b[0], segment_b[1]))
    return best


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / max((y2 - y1), 1e-12) + x1
        )
        if intersects:
            inside = not inside
        previous = current
    return inside


def point_distance_to_polygon(point: Point, polygon: list[Point]) -> float:
    if not polygon:
        return float("inf")
    if point_in_polygon(point, polygon):
        return 0.0
    return min(
        _point_to_segment_distance(point, start, end)
        for start, end in _segments(polygon)
    )


def point_inside_or_near_overlap(point: Point, polygon: list[Point], buffer_m: float) -> bool:
    return point_distance_to_polygon(point, polygon) <= buffer_m


def _polygon_area(polygon: list[Point]) -> float:
    area = 0.0
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return area / 2.0


def _clip_convex_polygon(subject: list[Point], clip: list[Point]) -> list[Point]:
    if _polygon_area(clip) < 0:
        clip = list(reversed(clip))
    output = list(subject)
    for clip_start, clip_end in _segments(clip):
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            current_inside = _is_left(clip_start, clip_end, current) >= 0
            previous_inside = _is_left(clip_start, clip_end, previous) >= 0
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(previous, current, clip_start, clip_end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, clip_start, clip_end))
            previous = current
    return output


def _is_left(a: Point, b: Point, p: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _line_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Point:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-9:
        return p2
    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4
    x = (det1 * (x3 - x4) - (x1 - x2) * det2) / denominator
    y = (det1 * (y3 - y4) - (y1 - y2) * det2) / denominator
    return (x, y)


def _segments(polygon: list[Point]) -> list[tuple[Point, Point]]:
    return [
        (polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    ]


def _segment_distance(a1: Point, a2: Point, b1: Point, b2: Point) -> float:
    if _segments_intersect(a1, a2, b1, b2):
        return 0.0
    return min(
        _point_to_segment_distance(a1, b1, b2),
        _point_to_segment_distance(a2, b1, b2),
        _point_to_segment_distance(b1, a1, a2),
        _point_to_segment_distance(b2, a1, a2),
    )


def _segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    d1 = _orientation(a1, a2, b1)
    d2 = _orientation(a1, a2, b2)
    d3 = _orientation(b1, b2, a1)
    d4 = _orientation(b1, b2, a2)
    return d1 * d2 < 0 and d3 * d4 < 0


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if dx == 0 and dy == 0:
        return math.dist(point, start)
    projection = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy)
    projection = max(0.0, min(1.0, projection))
    closest = (start[0] + projection * dx, start[1] + projection * dy)
    return math.dist(point, closest)
