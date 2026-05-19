from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class ImageCanvas(QWidget):
    point_clicked = Signal(float, float)
    point_added = Signal(float, float)
    point_moved = Signal(int, float, float)
    point_deleted = Signal(int)
    selected_point_changed = Signal(object)
    escape_pressed = Signal()

    def __init__(self, placeholder: str = "No image") -> None:
        super().__init__()
        self._image: QImage | None = None
        self._points: list[tuple[float, float]] = []
        self._pending_point: tuple[float, float] | None = None
        self._placeholder = placeholder
        self._selected_index: int | None = None
        self._drag_index: int | None = None
        self._drag_enabled = False
        self._delete_enabled = False
        self._add_enabled = False
        self._show_labels = True
        self._handle_radius = 8.0
        self._polygon_mode = False
        self._polygon_closed = True
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_image(self, image: QImage | None) -> None:
        self._image = image
        self.update()

    def set_points(self, points: Iterable[tuple[float, float]]) -> None:
        self.set_editable_points(points)

    def set_editable_points(self, points: Iterable[tuple[float, float]]) -> None:
        self._points = list(points)
        if self._selected_index is not None and self._selected_index >= len(self._points):
            self._selected_index = None
        self.update()

    def set_pending_point(self, point: tuple[float, float] | None) -> None:
        self._pending_point = point
        self.update()

    def clear_pending_point(self) -> None:
        self.set_pending_point(None)

    def set_drag_enabled(self, enabled: bool) -> None:
        self._drag_enabled = enabled

    def set_delete_enabled(self, enabled: bool) -> None:
        self._delete_enabled = enabled

    def set_add_enabled(self, enabled: bool) -> None:
        self._add_enabled = enabled

    def set_show_labels(self, enabled: bool) -> None:
        self._show_labels = enabled
        self.update()

    def set_selected_point(self, index: int | None) -> None:
        self._selected_index = index
        self.update()

    def selected_point(self) -> int | None:
        return self._selected_index

    def clear_points(self) -> None:
        self._points.clear()
        self._pending_point = None
        self._selected_index = None
        self.update()

    def set_polygon_mode(self, enabled: bool, *, closed: bool = True) -> None:
        self._polygon_mode = enabled
        self._polygon_closed = closed
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#11151c"))

        target_rect = self.image_rect()
        if self._image is None or target_rect is None:
            painter.setPen(QColor("#d8dee9"))
            painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
            return

        painter.drawImage(target_rect, self._image)
        if self._polygon_mode and len(self._points) >= 2:
            self._draw_polygon(painter, self._points, closed=self._polygon_closed)
        for index, point in enumerate(self._points):
            self._draw_point(
                painter,
                point,
                index + 1,
                selected=index == self._selected_index,
                fill="#ffd166",
                outline="#1f2937",
            )

        if self._pending_point is not None:
            if self._polygon_mode and self._points:
                previous = self.image_to_widget(self._points[-1])
                current = self.image_to_widget(self._pending_point)
                painter.setPen(QPen(QColor("#5eead4"), 2, Qt.DashLine))
                painter.drawLine(previous, current)
            self._draw_point(
                painter,
                self._pending_point,
                len(self._points) + 1,
                selected=True,
                fill="#5eead4",
                outline="#0f1720",
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        self.setFocus(Qt.MouseFocusReason)
        if self._image is None:
            return

        if event.button() == Qt.LeftButton:
            selected_index = self._nearest_point_index((event.position().x(), event.position().y()))
            if selected_index is not None:
                self._selected_index = selected_index
                self.selected_point_changed.emit(selected_index)
                if self._drag_enabled:
                    self._drag_index = selected_index
                self.update()
                return

            if not self._add_enabled:
                return
            mapped = self.widget_to_image((event.position().x(), event.position().y()))
            if mapped is None:
                return
            self.point_clicked.emit(mapped[0], mapped[1])
            self.point_added.emit(mapped[0], mapped[1])
            return

        if event.button() == Qt.RightButton and self._delete_enabled:
            selected_index = self._nearest_point_index((event.position().x(), event.position().y()))
            if selected_index is None:
                return
            self.point_deleted.emit(selected_index)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._drag_index is None or not self._drag_enabled:
            return
        mapped = self.widget_to_image((event.position().x(), event.position().y()))
        if mapped is None:
            return
        self.point_moved.emit(self._drag_index, mapped[0], mapped[1])

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:  # type: ignore[override]
        self._drag_index = None

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == Qt.Key_Escape:
            self._drag_index = None
            self._selected_index = None
            self.selected_point_changed.emit(None)
            self.escape_pressed.emit()
            self.update()
            event.accept()
            return
        super().keyPressEvent(event)

    def image_rect(self) -> QRectF | None:
        if self._image is None:
            return None
        image_size = self._image.size()
        scaled = image_size.scaled(self.size(), Qt.KeepAspectRatio)
        left = (self.width() - scaled.width()) / 2.0
        top = (self.height() - scaled.height()) / 2.0
        return QRectF(left, top, scaled.width(), scaled.height())

    def widget_to_image(self, position: tuple[float, float]) -> tuple[float, float] | None:
        rect = self.image_rect()
        if rect is None or self._image is None or not rect.contains(QPointF(*position)):
            return None
        image_x = (position[0] - rect.left()) * self._image.width() / rect.width()
        image_y = (position[1] - rect.top()) * self._image.height() / rect.height()
        return (image_x, image_y)

    def image_to_widget(self, point: tuple[float, float]) -> QPointF:
        rect = self.image_rect()
        if rect is None or self._image is None:
            return QPointF()
        widget_x = rect.left() + point[0] * rect.width() / self._image.width()
        widget_y = rect.top() + point[1] * rect.height() / self._image.height()
        return QPointF(widget_x, widget_y)

    def _nearest_point_index(self, position: tuple[float, float]) -> int | None:
        best_index: int | None = None
        best_distance = float("inf")
        for index, point in enumerate(self._points):
            widget_point = self.image_to_widget(point)
            distance = ((widget_point.x() - position[0]) ** 2 + (widget_point.y() - position[1]) ** 2) ** 0.5
            if distance <= self._handle_radius * 1.8 and distance < best_distance:
                best_index = index
                best_distance = distance
        return best_index

    def _draw_point(
        self,
        painter: QPainter,
        point: tuple[float, float],
        label: int,
        *,
        selected: bool,
        fill: str,
        outline: str,
    ) -> None:
        widget_point = self.image_to_widget(point)
        outer_radius = self._handle_radius if selected else self._handle_radius - 1.0
        painter.setPen(QPen(QColor(outline), 2))
        painter.setBrush(QColor(fill))
        painter.drawEllipse(widget_point, outer_radius, outer_radius)
        if selected:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawEllipse(widget_point, outer_radius + 3.0, outer_radius + 3.0)
        if self._show_labels:
            painter.setPen(QColor("#f8fafc"))
            painter.drawText(widget_point + QPointF(10, -10), str(label))

    def _draw_polygon(
        self,
        painter: QPainter,
        points: list[tuple[float, float]],
        *,
        closed: bool,
    ) -> None:
        if len(points) < 2:
            return
        path = QPainterPath()
        path.moveTo(self.image_to_widget(points[0]))
        for point in points[1:]:
            path.lineTo(self.image_to_widget(point))
        if closed and len(points) >= 3:
            path.closeSubpath()
            fill = QColor("#38bdf8")
            fill.setAlpha(42)
            painter.fillPath(path, fill)
        painter.setPen(QPen(QColor("#38bdf8"), 2))
        painter.drawPath(path)
