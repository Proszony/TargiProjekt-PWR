from __future__ import annotations

from typing import Iterable, Literal

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QBrush, QImage, QKeyEvent, QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from core.models import (
    AnalyticsSnapshot,
    CameraCoverageOverlay,
    CameraOverlapOverlay,
    Point,
    VenueMapConfig,
    WorldViewport,
    ZoneDefinition,
)
from core.zones import zone_color


class MapView(QWidget):
    world_point_clicked = Signal(float, float)
    zone_created = Signal(object)
    zone_updated = Signal(object)
    zone_selected = Signal(str)
    zone_edit_cancelled = Signal()
    world_point_moved = Signal(int, float, float)
    world_point_deleted = Signal(int)
    calibration_point_selected = Signal(object)
    draft_polygon_changed = Signal(object)
    escape_pressed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.venue_map = VenueMapConfig()
        self.analytics_snapshot = AnalyticsSnapshot(timestamp=0.0)
        self._camera_coverages: list[CameraCoverageOverlay] = []
        self._camera_overlaps: list[CameraOverlapOverlay] = []
        self._background = QImage()
        self._content_rect = QRectF()
        self._world_viewport = WorldViewport()
        self._mode: Literal["view", "pick_points", "draw_zone", "edit_zone"] = "view"
        self._draft_zone_name = ""
        self._draft_zone_kind = "neutral"
        self._draft_polygon: list[Point] = []
        self._selected_zone_id: str | None = None
        self._selected_zone_vertex_index: int | None = None
        self._calibration_points: list[Point] = []
        self._selected_calibration_index: int | None = None
        self._dragging_kind: Literal["calibration", "draft", "zone"] | None = None
        self._handle_radius = 8.0
        self.setMinimumSize(360, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_venue_map(self, venue_map: VenueMapConfig) -> None:
        self.venue_map = venue_map
        if venue_map.map_image_path:
            image = QImage(venue_map.map_image_path)
            self._background = image if not image.isNull() else QImage()
        else:
            self._background = QImage()
        if venue_map.manual_viewport_override is not None:
            self._world_viewport = venue_map.manual_viewport_override
        self.update()

    def set_world_viewport(self, viewport: WorldViewport) -> None:
        self._world_viewport = viewport
        self.update()

    def set_snapshot(self, snapshot: AnalyticsSnapshot) -> None:
        self.analytics_snapshot = snapshot
        self.update()

    def set_camera_coverages(self, coverages: list[CameraCoverageOverlay]) -> None:
        self._camera_coverages = coverages
        self.update()

    def set_camera_overlaps(self, overlaps: list[CameraOverlapOverlay]) -> None:
        self._camera_overlaps = overlaps
        self.update()

    def set_mode(self, mode: Literal["view", "pick_points", "draw_zone", "edit_zone"]) -> None:
        self._mode = mode
        if mode != "draw_zone":
            self._draft_zone_name = ""
            self._draft_polygon = []
        self.update()

    def begin_zone_drawing(self, name: str, kind: str) -> None:
        self._draft_zone_name = name
        self._draft_zone_kind = kind
        self._draft_polygon = []
        self._selected_zone_vertex_index = None
        self._mode = "draw_zone"
        self.update()

    def cancel_zone_drawing(self) -> None:
        self._draft_zone_name = ""
        self._draft_polygon = []
        self._selected_zone_vertex_index = None
        self._mode = "view"
        self.update()

    def set_pick_points_mode(self, enabled: bool) -> None:
        self._mode = "pick_points" if enabled else "view"
        self.update()

    def set_calibration_points(self, points: Iterable[Point]) -> None:
        self._calibration_points = list(points)
        if self._selected_calibration_index is not None and self._selected_calibration_index >= len(self._calibration_points):
            self._selected_calibration_index = None
        self.update()

    def set_selected_calibration_point(self, index: int | None) -> None:
        self._selected_calibration_index = index
        self.update()

    def remove_selected_zone(self) -> ZoneDefinition | None:
        if self._selected_zone_id is None:
            return None
        removed = None
        remaining: list[ZoneDefinition] = []
        for zone in self.venue_map.zones:
            if zone.zone_id == self._selected_zone_id:
                removed = zone
            else:
                remaining.append(zone)
        self.venue_map.zones = remaining
        self._selected_zone_id = None
        self._selected_zone_vertex_index = None
        self.update()
        return removed

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0f1720"))
        self._content_rect = self._calculate_content_rect()
        self._paint_background(painter)
        self._paint_grid(painter)
        self._paint_camera_coverages(painter)
        self._paint_camera_overlaps(painter)
        self._paint_zones(painter)
        self._paint_tracks(painter)
        self._paint_calibration_points(painter)
        self._paint_draft_polygon(painter)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        self.setFocus(Qt.MouseFocusReason)
        world_point = self.widget_to_world((event.position().x(), event.position().y()))
        if world_point is None:
            return

        if event.button() == Qt.LeftButton:
            self._handle_left_click(world_point, (event.position().x(), event.position().y()))
            return

        if event.button() == Qt.RightButton:
            self._handle_right_click((event.position().x(), event.position().y()))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._dragging_kind is None:
            return
        world_point = self.widget_to_world((event.position().x(), event.position().y()))
        if world_point is None:
            return

        if self._dragging_kind == "calibration" and self._selected_calibration_index is not None:
            self.world_point_moved.emit(self._selected_calibration_index, world_point[0], world_point[1])
            return

        if self._dragging_kind == "draft" and self._selected_zone_vertex_index is not None:
            self._draft_polygon[self._selected_zone_vertex_index] = world_point
            self.draft_polygon_changed.emit(list(self._draft_polygon))
            self.update()
            return

        if self._dragging_kind == "zone" and self._selected_zone_id is not None and self._selected_zone_vertex_index is not None:
            zone = self._selected_zone()
            if zone is None:
                return
            zone.polygon_world[self._selected_zone_vertex_index] = world_point
            self.zone_updated.emit(zone)
            self.update()

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:  # type: ignore[override]
        self._dragging_kind = None

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._mode != "draw_zone":
            return
        world_point = self.widget_to_world((event.position().x(), event.position().y()))
        if world_point is not None and not self._draft_polygon:
            self._draft_polygon.append(world_point)
        if len(self._draft_polygon) < 3:
            return
        zone_id = self._draft_zone_name.lower().replace(" ", "-")
        zone = ZoneDefinition(
            zone_id=zone_id,
            name=self._draft_zone_name,
            kind=self._draft_zone_kind,
            polygon_world=list(self._draft_polygon),
        )
        self.zone_created.emit(zone)
        self.cancel_zone_drawing()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == Qt.Key_Escape:
            self._dragging_kind = None
            self._selected_zone_vertex_index = None
            self._selected_calibration_index = None
            self.calibration_point_selected.emit(None)
            if self._mode == "draw_zone" and self._draft_polygon:
                self.cancel_zone_drawing()
                self.zone_edit_cancelled.emit()
            else:
                self.escape_pressed.emit()
            self.update()
            event.accept()
            return
        super().keyPressEvent(event)

    def _handle_left_click(self, world_point: Point, widget_position: tuple[float, float]) -> None:
        if self._mode == "pick_points":
            point_index = self._nearest_point_index(widget_position, self._calibration_points)
            if point_index is not None:
                self._selected_calibration_index = point_index
                self.calibration_point_selected.emit(point_index)
                self._dragging_kind = "calibration"
                self.update()
                return
            self.world_point_clicked.emit(world_point[0], world_point[1])
            return

        if self._mode == "draw_zone":
            point_index = self._nearest_point_index(widget_position, self._draft_polygon)
            if point_index is not None:
                self._selected_zone_vertex_index = point_index
                self._dragging_kind = "draft"
                self.update()
                return
            self._draft_polygon.append(world_point)
            self._selected_zone_vertex_index = len(self._draft_polygon) - 1
            self.draft_polygon_changed.emit(list(self._draft_polygon))
            self.update()
            return

        point_index = self._nearest_zone_vertex_index(widget_position)
        if point_index is not None and self._selected_zone_id is not None:
            self._selected_zone_vertex_index = point_index
            self._dragging_kind = "zone"
            self.update()
            return

        selected = self._zone_at(world_point)
        self._selected_zone_id = selected.zone_id if selected else None
        self._selected_zone_vertex_index = None
        if self._selected_zone_id is not None:
            self.zone_selected.emit(self._selected_zone_id)
            self._mode = "edit_zone"
        else:
            self._mode = "view"
        self.update()

    def _handle_right_click(self, widget_position: tuple[float, float]) -> None:
        if self._mode == "pick_points":
            point_index = self._nearest_point_index(widget_position, self._calibration_points)
            if point_index is not None:
                self.world_point_deleted.emit(point_index)
            return

        if self._mode == "draw_zone":
            point_index = self._nearest_point_index(widget_position, self._draft_polygon)
            if point_index is None and self._draft_polygon:
                point_index = len(self._draft_polygon) - 1
            if point_index is not None:
                del self._draft_polygon[point_index]
                self._selected_zone_vertex_index = None
                self.draft_polygon_changed.emit(list(self._draft_polygon))
                self.update()
            return

        if self._selected_zone_id is None:
            return
        zone = self._selected_zone()
        if zone is None:
            return
        point_index = self._nearest_point_index(widget_position, zone.polygon_world)
        if point_index is None or len(zone.polygon_world) <= 3:
            return
        del zone.polygon_world[point_index]
        self._selected_zone_vertex_index = None
        self.zone_updated.emit(zone)
        self.update()

    def _paint_background(self, painter: QPainter) -> None:
        if not self._background.isNull():
            painter.drawImage(self._content_rect, self._background)
        else:
            painter.fillRect(self._content_rect, QColor("#1b2635"))

    def _paint_grid(self, painter: QPainter) -> None:
        pen = QPen(QColor("#243447"))
        pen.setWidth(1)
        painter.setPen(pen)
        for step in range(1, 10):
            x = self._content_rect.left() + self._content_rect.width() * step / 10.0
            y = self._content_rect.top() + self._content_rect.height() * step / 10.0
            painter.drawLine(QPointF(x, self._content_rect.top()), QPointF(x, self._content_rect.bottom()))
            painter.drawLine(QPointF(self._content_rect.left(), y), QPointF(self._content_rect.right(), y))

    def _paint_zones(self, painter: QPainter) -> None:
        for zone in self.venue_map.zones:
            path = self._zone_path(zone.polygon_world)
            fill = QColor(zone_color(zone.kind))
            fill.setAlpha(70)
            painter.fillPath(path, fill)
            pen = QPen(QColor(zone_color(zone.kind)))
            pen.setWidth(3 if zone.zone_id == self._selected_zone_id else 2)
            painter.setPen(pen)
            painter.drawPath(path)
            if zone.polygon_world:
                first_point = self.world_to_widget(zone.polygon_world[0])
                count = self.analytics_snapshot.active_zone_counts.get(zone.zone_id, 0)
                painter.setPen(QColor("#f8f9fa"))
                painter.drawText(first_point + QPointF(8, -8), f"{zone.name} ({count})")
            if zone.zone_id == self._selected_zone_id:
                for index, point in enumerate(zone.polygon_world, start=1):
                    self._draw_handle(
                        painter,
                        point,
                        label=index,
                        selected=index - 1 == self._selected_zone_vertex_index,
                        fill="#fb923c",
                    )

    def _paint_camera_coverages(self, painter: QPainter) -> None:
        for coverage in self._camera_coverages:
            raw_polygon = coverage.raw_polygon_world
            clipped_polygon = coverage.polygon_world
            if len(raw_polygon) >= 3:
                path = self._zone_path(raw_polygon)
                pen = QPen(QColor(coverage.color))
                pen.setWidth(1)
                pen.setStyle(Qt.DashLine)
                raw_color = QColor(coverage.color)
                raw_color.setAlpha(70)
                pen.setColor(raw_color)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(path)
            if len(clipped_polygon) < 3:
                continue
            path = self._zone_path(clipped_polygon)
            fill = QColor(coverage.color)
            fill.setAlpha(36)
            painter.fillPath(path, fill)
            pen = QPen(QColor(coverage.color))
            pen.setWidth(2)
            pen.setStyle(Qt.SolidLine)
            painter.setPen(pen)
            painter.drawPath(path)
            painter.setPen(QColor("#cbd5e1"))
            painter.drawText(self._polygon_label_point(clipped_polygon), coverage.camera_name)

    def _paint_camera_overlaps(self, painter: QPainter) -> None:
        for overlay in self._camera_overlaps:
            if len(overlay.polygon_world) < 3:
                continue
            path = self._zone_path(overlay.polygon_world)
            fill = QColor("#f59e0b")
            fill.setAlpha(82)
            painter.fillPath(path, fill)
            brush = QBrush(QColor("#fbbf24"), Qt.BDiagPattern)
            painter.fillPath(path, brush)
            pen = QPen(QColor("#f97316"))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawPath(path)
            self._draw_text_chip(
                painter,
                self._polygon_label_point(overlay.polygon_world),
                overlay.label,
                background="#7c2d12",
                foreground="#fff7ed",
            )

    def _paint_tracks(self, painter: QPainter) -> None:
        for presence in self.analytics_snapshot.active_map_presences:
            point = self.world_to_widget(presence.world_point)
            fill_color = QColor("#ffe66d")
            border_color = QColor("#f59e0b") if presence.merged_for_counting else QColor("#0f1720")
            painter.setPen(QPen(border_color, 2 if presence.merged_for_counting else 1))
            painter.setBrush(fill_color)
            painter.drawEllipse(point, 6, 6)

    def _paint_calibration_points(self, painter: QPainter) -> None:
        for index, point in enumerate(self._calibration_points, start=1):
            self._draw_handle(
                painter,
                point,
                label=index,
                selected=index - 1 == self._selected_calibration_index,
                fill="#facc15",
            )

    def _paint_draft_polygon(self, painter: QPainter) -> None:
        if not self._draft_polygon:
            return
        painter.setPen(QPen(QColor("#ffd166"), 2, Qt.DashLine))
        path = self._zone_path(self._draft_polygon, close_path=False)
        painter.drawPath(path)
        for index, point in enumerate(self._draft_polygon, start=1):
            self._draw_handle(
                painter,
                point,
                label=index,
                selected=index - 1 == self._selected_zone_vertex_index,
                fill="#ffd166",
            )

    def _draw_handle(
        self,
        painter: QPainter,
        point: Point,
        *,
        label: int,
        selected: bool,
        fill: str,
    ) -> None:
        widget_point = self.world_to_widget(point)
        painter.setPen(QPen(QColor("#0f1720"), 2))
        painter.setBrush(QColor(fill))
        painter.drawEllipse(widget_point, self._handle_radius, self._handle_radius)
        if selected:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawEllipse(widget_point, self._handle_radius + 3, self._handle_radius + 3)
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(widget_point + QPointF(10, -10), str(label))

    def _zone_path(self, polygon: Iterable[Point], close_path: bool = True) -> QPainterPath:
        points = list(polygon)
        path = QPainterPath()
        if not points:
            return path
        path.moveTo(self.world_to_widget(points[0]))
        for point in points[1:]:
            path.lineTo(self.world_to_widget(point))
        if close_path and len(points) > 2:
            path.closeSubpath()
        return path

    def _polygon_label_point(self, polygon: list[Point]) -> QPointF:
        if not polygon:
            return QPointF()
        center_x = sum(point[0] for point in polygon) / len(polygon)
        center_y = sum(point[1] for point in polygon) / len(polygon)
        return self.world_to_widget((center_x, center_y)) + QPointF(6, -6)

    def _draw_text_chip(
        self,
        painter: QPainter,
        point: QPointF,
        text: str,
        *,
        background: str,
        foreground: str,
    ) -> None:
        metrics = painter.fontMetrics()
        text_rect = metrics.boundingRect(text)
        width = text_rect.width() + 12
        height = text_rect.height() + 8
        x = point.x()
        y = point.y() - height
        if x + width > self._content_rect.right() - 4:
            x = max(self._content_rect.left() + 4, point.x() - width - 12)
        if y < self._content_rect.top() + 4:
            y = min(self._content_rect.bottom() - height - 4, point.y() + 4)
        painter.setPen(QPen(QColor("#00000000")))
        chip_color = QColor(background)
        chip_color.setAlpha(220)
        painter.setBrush(chip_color)
        painter.drawRoundedRect(QRectF(x, y, width, height), 6, 6)
        painter.setPen(QColor(foreground))
        painter.drawText(QPointF(x + 6, y + metrics.ascent() + 4), text)

    def _calculate_content_rect(self) -> QRectF:
        margin = 12.0
        target = self.rect().adjusted(int(margin), int(margin), int(-margin), int(-margin))
        if self._background.isNull():
            return QRectF(target)
        scaled = self._background.size().scaled(target.size(), Qt.KeepAspectRatio)
        left = target.left() + (target.width() - scaled.width()) / 2.0
        top = target.top() + (target.height() - scaled.height()) / 2.0
        return QRectF(left, top, scaled.width(), scaled.height())

    def world_to_widget(self, point: Point) -> QPointF:
        width = max(self._world_viewport.max_x - self._world_viewport.min_x, 1e-6)
        height = max(self._world_viewport.max_y - self._world_viewport.min_y, 1e-6)
        x = self._content_rect.left() + ((point[0] - self._world_viewport.min_x) / width) * self._content_rect.width()
        y = self._content_rect.top() + ((point[1] - self._world_viewport.min_y) / height) * self._content_rect.height()
        return QPointF(x, y)

    def widget_to_world(self, position: tuple[float, float]) -> Point | None:
        point = QPointF(position[0], position[1])
        if not self._content_rect.contains(point):
            return None
        width = max(self._world_viewport.max_x - self._world_viewport.min_x, 1e-6)
        height = max(self._world_viewport.max_y - self._world_viewport.min_y, 1e-6)
        world_x = self._world_viewport.min_x + (
            (position[0] - self._content_rect.left()) / max(self._content_rect.width(), 1e-6)
        ) * width
        world_y = self._world_viewport.min_y + (
            (position[1] - self._content_rect.top()) / max(self._content_rect.height(), 1e-6)
        ) * height
        return (world_x, world_y)

    def _zone_at(self, point: Point) -> ZoneDefinition | None:
        for zone in reversed(self.venue_map.zones):
            widget_path = self._zone_path(zone.polygon_world)
            if widget_path.contains(self.world_to_widget(point)):
                return zone
        return None

    def _selected_zone(self) -> ZoneDefinition | None:
        if self._selected_zone_id is None:
            return None
        for zone in self.venue_map.zones:
            if zone.zone_id == self._selected_zone_id:
                return zone
        return None

    def _nearest_zone_vertex_index(self, widget_position: tuple[float, float]) -> int | None:
        zone = self._selected_zone()
        if zone is None:
            return None
        return self._nearest_point_index(widget_position, zone.polygon_world)

    def _nearest_point_index(self, widget_position: tuple[float, float], points: list[Point]) -> int | None:
        best_index: int | None = None
        best_distance = float("inf")
        for index, point in enumerate(points):
            widget_point = self.world_to_widget(point)
            distance = ((widget_point.x() - widget_position[0]) ** 2 + (widget_point.y() - widget_position[1]) ** 2) ** 0.5
            if distance <= self._handle_radius * 1.8 and distance < best_distance:
                best_index = index
                best_distance = distance
        return best_index
