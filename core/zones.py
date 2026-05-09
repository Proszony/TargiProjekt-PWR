from __future__ import annotations

from collections import defaultdict

from core.models import Point, ZoneDefinition

ZONE_COLORS = {
    "booth": "#C44900",
    "aisle": "#0A7B83",
    "entry": "#2E8B57",
    "exit": "#8B1E3F",
    "neutral": "#5C677D",
}


def zone_color(zone_kind: str) -> str:
    return ZONE_COLORS.get(zone_kind, ZONE_COLORS["neutral"])


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    x1, y1 = polygon[0]
    for index in range(len(polygon) + 1):
        x2, y2 = polygon[index % len(polygon)]
        if y > min(y1, y2) and y <= max(y1, y2) and x <= max(x1, x2):
            if y1 != y2:
                xinters = (y - y1) * (x2 - x1) / (y2 - y1 + 1e-9) + x1
            else:
                xinters = x1
            if x1 == x2 or x <= xinters:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def find_zone(point: Point | None, zones: list[ZoneDefinition]) -> ZoneDefinition | None:
    if point is None:
        return None
    for zone in zones:
        if point_in_polygon(point, zone.polygon_world):
            return zone
    return None


def summarize_zone_visits(zone_ids: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for zone_id in zone_ids:
        counts[zone_id] += 1
    return dict(counts)
