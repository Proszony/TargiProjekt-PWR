from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QPainter, QRadialGradient

from core.models import HeatmapSnapshot


def draw_heatmap_cells(
    painter: QPainter,
    heatmap: HeatmapSnapshot | None,
    cell_rect: Callable[[int, int], QRectF],
) -> None:
    if heatmap is None or heatmap.max_dwell_s <= 0.0:
        return
    painter.save()
    painter.setPen(QColor("#00000000"))
    for cell in heatmap.cells:
        ratio = max(0.0, min(cell.dwell_s / heatmap.max_dwell_s, 1.0))
        color = heatmap_color(ratio)
        rect = cell_rect(cell.x_index, cell.y_index)
        center = rect.center()
        radius = max(rect.width(), rect.height()) * 2.15
        blob_rect = QRectF(
            center.x() - radius,
            center.y() - radius,
            radius * 2.0,
            radius * 2.0,
        )
        gradient = QRadialGradient(center, radius)
        gradient.setColorAt(0.0, color)
        mid = QColor(color)
        mid.setAlpha(max(16, int(color.alpha() * 0.42)))
        gradient.setColorAt(0.58, mid)
        edge = QColor(color)
        edge.setAlpha(0)
        gradient.setColorAt(1.0, edge)
        painter.setBrush(gradient)
        painter.drawEllipse(blob_rect)
    painter.restore()


def heatmap_color(ratio: float) -> QColor:
    ratio = max(0.0, min(ratio, 1.0))
    if ratio < 0.5:
        local = ratio / 0.5
        color = _lerp_color(QColor("#38bdf8"), QColor("#f59e0b"), local)
    else:
        local = (ratio - 0.5) / 0.5
        color = _lerp_color(QColor("#f59e0b"), QColor("#f43f5e"), local)
    color.setAlpha(int(38 + ratio * 120))
    return color


def _lerp_color(start: QColor, end: QColor, ratio: float) -> QColor:
    ratio = max(0.0, min(ratio, 1.0))
    return QColor(
        round(start.red() + (end.red() - start.red()) * ratio),
        round(start.green() + (end.green() - start.green()) * ratio),
        round(start.blue() + (end.blue() - start.blue()) * ratio),
    )
