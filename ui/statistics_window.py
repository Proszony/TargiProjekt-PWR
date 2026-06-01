from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.models import AnalyticsSnapshot, HeatmapSnapshot, MultiCameraRuntimeSnapshot, VenueMapConfig, WorldViewport
from core.statistics_repository import StatisticsRepository
from core.zones import zone_color
from ui.heatmap_rendering import draw_heatmap_cells
from ui.style_system import COLORS, apply_chrome


@dataclass(frozen=True, slots=True)
class _HistorySessionPayload:
    session_id: int
    summary: dict[str, object] | None
    metrics: list[dict[str, object]]
    visits: list[dict[str, object]]
    heatmap: HeatmapSnapshot | None
    heatmap_preview: QImage | None


@dataclass(frozen=True, slots=True)
class _TimelinePayload:
    session_id: int
    zone_id: str
    timeline: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class _ExportPayload:
    title: str
    message: str


class _TaskSignals(QObject):
    finished = Signal(int, object)
    failed = Signal(int, str)


class _BackgroundTask(QRunnable):
    def __init__(self, token: int, fn: Callable[[], object]) -> None:
        super().__init__()
        self.token = token
        self.fn = fn
        self.signals = _TaskSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.fn()
        except Exception as exc:
            self.signals.failed.emit(self.token, str(exc))
            return
        self.signals.finished.emit(self.token, result)


def _clone_venue_map(venue_map: VenueMapConfig) -> VenueMapConfig:
    return VenueMapConfig.from_dict(venue_map.to_dict())


def _render_heatmap_image(
    heatmap: HeatmapSnapshot,
    width: int,
    height: int,
    venue_map: VenueMapConfig,
) -> QImage:
    image = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
    image.fill(QColor(COLORS["bg-canvas"]))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    content_rect = QRectF(18.0, 18.0, width - 36.0, height - 36.0)
    background = QLinearGradient(content_rect.left(), content_rect.top(), content_rect.right(), content_rect.bottom())
    background.setColorAt(0.0, QColor("#15222f"))
    background.setColorAt(1.0, QColor("#0b1620"))
    painter.fillRect(content_rect, background)

    map_image = QImage(venue_map.map_image_path) if venue_map.map_image_path else QImage()
    if not map_image.isNull():
        painter.drawImage(content_rect, map_image)

    def world_to_image(point: tuple[float, float], viewport: WorldViewport = heatmap.viewport) -> QPointF:
        viewport_width = max(viewport.max_x - viewport.min_x, 1e-6)
        viewport_height = max(viewport.max_y - viewport.min_y, 1e-6)
        x = content_rect.left() + ((point[0] - viewport.min_x) / viewport_width) * content_rect.width()
        y = content_rect.top() + ((point[1] - viewport.min_y) / viewport_height) * content_rect.height()
        return QPointF(x, y)

    viewport_width = max(heatmap.viewport.max_x - heatmap.viewport.min_x, 1e-6)
    viewport_height = max(heatmap.viewport.max_y - heatmap.viewport.min_y, 1e-6)

    def cell_rect(x_index: int, y_index: int) -> QRectF:
        world_left = heatmap.viewport.min_x + (x_index / heatmap.columns) * viewport_width
        world_right = heatmap.viewport.min_x + ((x_index + 1) / heatmap.columns) * viewport_width
        world_top = heatmap.viewport.min_y + (y_index / heatmap.rows) * viewport_height
        world_bottom = heatmap.viewport.min_y + ((y_index + 1) / heatmap.rows) * viewport_height
        return QRectF(world_to_image((world_left, world_top)), world_to_image((world_right, world_bottom))).normalized()

    draw_heatmap_cells(painter, heatmap, cell_rect)
    _draw_heatmap_zones(painter, world_to_image, venue_map)
    painter.end()
    return image


def _write_heatmap_png(path: Path, heatmap: HeatmapSnapshot, venue_map: VenueMapConfig) -> None:
    image = _render_heatmap_image(heatmap, 1280, 900, venue_map)
    image.save(str(path), "PNG")


def _draw_heatmap_zones(
    painter: QPainter,
    world_to_image: Callable[[tuple[float, float]], QPointF],
    venue_map: VenueMapConfig,
) -> None:
    for zone in venue_map.zones:
        if len(zone.polygon_world) < 3:
            continue
        path = QPainterPath()
        path.moveTo(world_to_image(zone.polygon_world[0]))
        for point in zone.polygon_world[1:]:
            path.lineTo(world_to_image(point))
        path.closeSubpath()
        color = QColor(zone_color(zone.kind))
        color.setAlpha(86)
        painter.fillPath(path, color)
        pen = QPen(QColor(zone_color(zone.kind)), 2)
        painter.setPen(pen)
        painter.drawPath(path)
        painter.setPen(QColor(COLORS["text-primary"]))
        painter.drawText(world_to_image(zone.polygon_world[0]) + QPointF(8, -8), zone.name)


class RankedMetricList(QWidget):
    def __init__(self, empty_text: str) -> None:
        super().__init__()
        self._empty_text = empty_text
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self.setObjectName("RankedMetricList")

    def set_rows(self, rows: list[tuple[str, float, str]], color: str) -> None:
        self._clear()
        ranked = sorted(rows, key=lambda item: item[1], reverse=True)
        if not ranked or max((value for _name, value, _display in ranked), default=0.0) <= 0.0:
            empty = QLabel(self._empty_text)
            empty.setObjectName("RankedEmpty")
            empty.setAlignment(Qt.AlignCenter)
            self._layout.addWidget(empty, 1)
            return

        max_value = max(value for _name, value, _display in ranked)
        for name, value, display_value in ranked[:5]:
            self._layout.addWidget(self._build_row(name, value / max_value, display_value, color))
        self._layout.addStretch(1)

    def _clear(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    @staticmethod
    def _build_row(name: str, ratio: float, display_value: str, color: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName("RankedRow")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 9)
        layout.setSpacing(7)

        header = QHBoxLayout()
        header.setSpacing(8)
        name_label = QLabel(name)
        name_label.setObjectName("RankedName")
        value_label = QLabel(display_value)
        value_label.setObjectName("RankedValue")
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header.addWidget(name_label, 1)
        header.addWidget(value_label)

        bar = QProgressBar()
        bar.setObjectName("RankedBar")
        bar.setRange(0, 1000)
        bar.setValue(int(max(0.0, min(ratio, 1.0)) * 1000))
        bar.setTextVisible(False)
        bar.setFixedHeight(8)
        bar.setStyleSheet(
            f"""
            QProgressBar#RankedBar {{
                background: {COLORS["bg-input"]};
                border: 1px solid {COLORS["border-muted"]};
                border-radius: 4px;
            }}
            QProgressBar#RankedBar::chunk {{
                background: {color};
                border-radius: 4px;
            }}
            """
        )

        layout.addLayout(header)
        layout.addWidget(bar)
        return frame


class StatisticsWindow(QMainWindow):
    def __init__(self, repository: StatisticsRepository, parent=None) -> None:
        super().__init__(parent)
        self.repository = repository
        self.current_session_id: int | None = None
        self.live_snapshot = AnalyticsSnapshot(timestamp=0.0)
        self.runtime_snapshot = MultiCameraRuntimeSnapshot(timestamp=0.0)
        self.live_venue_map = VenueMapConfig()
        self._last_live_timeline_refresh_at = 0.0
        self._thread_pool = QThreadPool.globalInstance()
        self._history_sessions: list[dict[str, object]] = []
        self._history_loaded_session_id: int | None = None
        self._history_loaded_metrics: list[dict[str, object]] = []
        self._history_loaded_visits: list[dict[str, object]] = []
        self._history_selected_heatmap: HeatmapSnapshot | None = None
        self._history_reload_token = 0
        self._history_session_token = 0
        self._history_timeline_token = 0
        self._history_export_token = 0
        self.setWindowTitle("Booth Analytics Dashboard")
        self.resize(1360, 920)
        self._build_ui()
        self._apply_styles()
        self.reload_history()

    def _build_ui(self) -> None:
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        self.live_tab = QWidget()
        self.history_tab = QWidget()
        self.sessions_tab = QWidget()
        self.tabs.addTab(self.live_tab, "Live")
        self.tabs.addTab(self.history_tab, "History")
        self.tabs.addTab(self.sessions_tab, "Visit sessions")

        self._build_live_tab()
        self._build_history_tab()
        self._build_sessions_tab()

    def _build_live_tab(self) -> None:
        layout = QVBoxLayout(self.live_tab)
        layout.setSpacing(14)

        header_row = QHBoxLayout()
        header_text = QVBoxLayout()
        self.live_session_label = QLabel("Live booth dashboard")
        self.live_session_label.setObjectName("SectionTitle")
        self.live_status_label = QLabel("Current session: none")
        self.live_status_label.setObjectName("MutedText")
        self.live_hint_label = QLabel(
            "No active session. Live occupancy stays at 0 when playback is stopped or the session has already ended."
        )
        self.live_hint_label.setObjectName("HintText")
        self.live_hint_label.setWordWrap(True)
        header_text.addWidget(self.live_session_label)
        header_text.addWidget(self.live_status_label)
        header_text.addWidget(self.live_hint_label)
        header_row.addLayout(header_text, 1)
        self.live_export_button = QPushButton("Export live CSV")
        self.live_export_button.clicked.connect(self._export_live_csv)
        self.live_bundle_export_button = QPushButton("Export session bundle")
        self.live_bundle_export_button.clicked.connect(self._export_current_session_bundle)
        self.live_bundle_export_button.setProperty("kind", "primary")
        header_row.addWidget(self.live_export_button)
        header_row.addWidget(self.live_bundle_export_button)

        cards_grid = QGridLayout()
        cards_grid.setHorizontalSpacing(12)
        cards_grid.setVerticalSpacing(12)
        self.live_occupancy_card, self.live_occupancy_value = self._create_metric_card(
            "Current occupancy", "0", "#22c55e"
        )
        self.live_visits_card, self.live_visits_value = self._create_metric_card(
            "Visits", "0", "#38bdf8"
        )
        self.live_avg_dwell_card, self.live_avg_dwell_value = self._create_metric_card(
            "Average time", "0.0s", "#f59e0b"
        )
        self.live_peak_card, self.live_peak_value = self._create_metric_card(
            "Peak occupancy", "0", "#a78bfa"
        )
        self.live_active_booths_card, self.live_active_booths_value = self._create_metric_card(
            "Active booths", "0", "#f43f5e"
        )
        cards_grid.addWidget(self.live_occupancy_card, 0, 0)
        cards_grid.addWidget(self.live_visits_card, 0, 1)
        cards_grid.addWidget(self.live_avg_dwell_card, 0, 2)
        cards_grid.addWidget(self.live_peak_card, 0, 3)
        cards_grid.addWidget(self.live_active_booths_card, 0, 4)

        self.live_table = QTableWidget(0, 8)
        self.live_table.setHorizontalHeaderLabels(
            [
                "Booth",
                "Type",
                "Now",
                "Visits",
                "Total [s]",
                "Avg [s]",
                "Median [s]",
                "Peak",
            ]
        )
        self._configure_table(self.live_table, [120, 76, 64, 70, 82, 76, 92, 70])

        timeline_header = QHBoxLayout()
        self.timeline_zone_combo = QComboBox()
        self.timeline_zone_combo.setMinimumWidth(260)
        timeline_header.addWidget(QLabel("Timeline booth"))
        timeline_header.addWidget(self.timeline_zone_combo, 0)
        timeline_header.addStretch(1)
        self.live_timeline_chart = self._create_chart_view("Live occupancy timeline")

        main_row = QHBoxLayout()
        main_row.setSpacing(12)
        live_table_panel = self._create_panel("Booth snapshot", "Current occupancy and time summary per booth")
        live_table_panel.layout().addWidget(self.live_table)  # type: ignore[union-attr]
        live_timeline_panel = self._create_panel("Occupancy trend", "Live stepped occupancy for the selected booth")
        live_timeline_panel.layout().addLayout(timeline_header)  # type: ignore[union-attr]
        live_timeline_panel.layout().addWidget(self.live_timeline_chart, 1)  # type: ignore[union-attr]
        main_row.addWidget(live_table_panel, 7)
        main_row.addWidget(live_timeline_panel, 6)

        charts_row = QHBoxLayout()
        charts_row.setSpacing(12)
        self.live_visits_rank = RankedMetricList("No visits yet")
        self.live_time_rank = RankedMetricList("No recorded time yet")
        self.live_peak_rank = RankedMetricList("No peak load yet")
        visits_panel = self._create_panel("Visits", "Booths ranked by completed visits")
        visits_panel.layout().addWidget(self.live_visits_rank, 1)  # type: ignore[union-attr]
        dwell_panel = self._create_panel("Time", "Average time per visit")
        dwell_panel.layout().addWidget(self.live_time_rank, 1)  # type: ignore[union-attr]
        peak_panel = self._create_panel("Peak load", "Highest simultaneous occupancy")
        peak_panel.layout().addWidget(self.live_peak_rank, 1)  # type: ignore[union-attr]
        charts_row.addWidget(visits_panel, 1)
        charts_row.addWidget(dwell_panel, 1)
        charts_row.addWidget(peak_panel, 1)

        layout.addLayout(header_row)
        layout.addLayout(cards_grid)
        layout.addLayout(main_row, 1)
        layout.addLayout(charts_row)

        self.timeline_zone_combo.currentTextChanged.connect(self._refresh_live_timeline)

    def _build_history_tab(self) -> None:
        layout = QVBoxLayout(self.history_tab)
        layout.setSpacing(14)

        top_row = QHBoxLayout()
        self.session_selector = QComboBox()
        self.refresh_history_button = QPushButton("Refresh")
        self.history_export_button = QPushButton("Export CSV")
        self.history_bundle_export_button = QPushButton("Export session bundle")
        self.history_bundle_export_button.setProperty("kind", "primary")
        top_row.addWidget(QLabel("Session"))
        top_row.addWidget(self.session_selector, 1)
        top_row.addWidget(self.refresh_history_button)
        top_row.addWidget(self.history_export_button)
        top_row.addWidget(self.history_bundle_export_button)

        self.history_summary_label = QLabel("No session selected")
        self.history_summary_label.setObjectName("SectionTitle")
        self.history_hint_label = QLabel(
            "History shows the last recorded occupancy at session end. It is often 0 because people already left; use peak occupancy and the timeline to assess booth traffic."
        )
        self.history_hint_label.setObjectName("HintText")
        self.history_hint_label.setWordWrap(True)

        history_cards = QGridLayout()
        history_cards.setHorizontalSpacing(12)
        history_cards.setVerticalSpacing(12)
        self.history_visits_card, self.history_visits_value = self._create_metric_card(
            "Session visits", "0", "#38bdf8"
        )
        self.history_avg_card, self.history_avg_value = self._create_metric_card(
            "Average time", "0.0s", "#f59e0b"
        )
        self.history_peak_card, self.history_peak_value = self._create_metric_card(
            "Peak occupancy", "0", "#a78bfa"
        )
        self.history_booths_card, self.history_booths_value = self._create_metric_card(
            "Booths with traffic", "0", "#22c55e"
        )
        history_cards.addWidget(self.history_visits_card, 0, 0)
        history_cards.addWidget(self.history_avg_card, 0, 1)
        history_cards.addWidget(self.history_peak_card, 0, 2)
        history_cards.addWidget(self.history_booths_card, 0, 3)

        self.history_table = QTableWidget(0, 8)
        self.history_table.setHorizontalHeaderLabels(
            [
                "Booth",
                "Type",
                "Final",
                "Visits",
                "Total [s]",
                "Avg [s]",
                "Median [s]",
                "Peak",
            ]
        )
        self._configure_table(self.history_table, [120, 76, 64, 70, 82, 76, 92, 70])

        timeline_header = QHBoxLayout()
        self.history_timeline_zone_combo = QComboBox()
        self.history_timeline_zone_combo.setMinimumWidth(260)
        timeline_header.addWidget(QLabel("Timeline booth"))
        timeline_header.addWidget(self.history_timeline_zone_combo, 0)
        timeline_header.addStretch(1)

        self.history_timeline_chart = self._create_chart_view("Selected session occupancy timeline")
        self.history_heatmap_preview = QLabel("No heatmap recorded")
        self.history_heatmap_preview.setObjectName("HeatmapPreview")
        self.history_heatmap_preview.setAlignment(Qt.AlignCenter)
        self.history_heatmap_preview.setMinimumSize(320, 220)
        self.history_heatmap_preview.setScaledContents(False)
        self.history_heatmap_export_button = QPushButton("Export heatmap PNG")
        self.history_heatmap_export_button.setEnabled(False)

        history_main_row = QHBoxLayout()
        history_main_row.setSpacing(12)
        history_table_panel = self._create_panel("Session booth summary", "Final occupancy, visits and time metrics")
        history_table_panel.layout().addWidget(self.history_table)  # type: ignore[union-attr]
        history_timeline_panel = self._create_panel("Session timeline", "Selected booth occupancy across the session")
        history_timeline_panel.layout().addLayout(timeline_header)  # type: ignore[union-attr]
        history_timeline_panel.layout().addWidget(self.history_timeline_chart, 1)  # type: ignore[union-attr]
        history_heatmap_panel = self._create_panel("Session heatmap", "Where tracked presences spent the most time")
        history_heatmap_panel.layout().addWidget(self.history_heatmap_preview, 1)  # type: ignore[union-attr]
        history_heatmap_panel.layout().addWidget(self.history_heatmap_export_button)  # type: ignore[union-attr]
        history_main_row.addWidget(history_table_panel, 7)
        history_main_row.addWidget(history_timeline_panel, 6)
        history_main_row.addWidget(history_heatmap_panel, 5)

        layout.addLayout(top_row)
        layout.addWidget(self.history_summary_label)
        layout.addWidget(self.history_hint_label)
        layout.addLayout(history_cards)
        layout.addLayout(history_main_row, 1)

        self.refresh_history_button.clicked.connect(self.reload_history)
        self.history_export_button.clicked.connect(self._export_history_csv)
        self.history_bundle_export_button.clicked.connect(self._export_current_session_bundle)
        self.history_heatmap_export_button.clicked.connect(self._export_history_heatmap_png)
        self.session_selector.currentIndexChanged.connect(self._load_selected_session)
        self.history_timeline_zone_combo.currentTextChanged.connect(self._refresh_history_timeline)

    def _build_sessions_tab(self) -> None:
        layout = QVBoxLayout(self.sessions_tab)
        layout.setSpacing(14)

        top_row = QHBoxLayout()
        title_box = QVBoxLayout()
        sessions_title = QLabel("Visit sessions")
        sessions_title.setObjectName("SectionTitle")
        sessions_subtitle = QLabel("Finalized booth visits for the selected recorded session")
        sessions_subtitle.setObjectName("MutedText")
        title_box.addWidget(sessions_title)
        title_box.addWidget(sessions_subtitle)
        top_row.addLayout(title_box)
        top_row.addStretch(1)
        self.sessions_bundle_export_button = QPushButton("Export session bundle")
        self.sessions_bundle_export_button.clicked.connect(self._export_current_session_bundle)
        self.sessions_bundle_export_button.setProperty("kind", "primary")
        top_row.addWidget(self.sessions_bundle_export_button)
        self.sessions_export_button = QPushButton("Export CSV")
        self.sessions_export_button.clicked.connect(self._export_sessions_csv)
        top_row.addWidget(self.sessions_export_button)

        self.visit_sessions_table = QTableWidget(0, 7)
        self.visit_sessions_table.setHorizontalHeaderLabels(
            [
                "Visit",
                "Track",
                "Booth",
                "Entered",
                "Left",
                "Time [s]",
                "Dedup",
            ]
        )
        self._configure_table(self.visit_sessions_table, [180, 150, 110, 82, 82, 82, 220])

        layout.addLayout(top_row)
        layout.addWidget(self.visit_sessions_table)

    def _apply_styles(self) -> None:
        apply_chrome(self)
        self.setStyleSheet(
            self.styleSheet()
            + f"""
            QWidget#RankedMetricList {{
                background: transparent;
            }}
            QFrame#RankedRow {{
                background: {COLORS["bg-shell-deep"]};
                border: 1px solid {COLORS["border-muted"]};
                border-radius: 10px;
            }}
            QLabel#RankedName {{
                color: {COLORS["text-primary"]};
                font-size: 12px;
                font-weight: 700;
            }}
            QLabel#RankedValue {{
                color: {COLORS["text-secondary"]};
                font-size: 13px;
                font-weight: 800;
            }}
            QLabel#RankedEmpty {{
                color: {COLORS["text-faint"]};
                font-size: 12px;
                padding: 22px;
            }}
            """
        )

    def set_current_session_id(self, session_id: int | None) -> None:
        self.current_session_id = session_id
        session_text = f"Current session: #{session_id}" if session_id is not None else "Current session: none"
        self.live_status_label.setText(session_text)
        self.live_hint_label.setText(
            "Live occupancy is computed only for an active session."
            if session_id is not None
            else "No active session. Live occupancy stays at 0 when playback is stopped or the session has already ended."
        )
        self._last_live_timeline_refresh_at = 0.0
        self._refresh_live_timeline()

    def _set_ranked_live_lists(self, rows: list[object]) -> None:
        visits: list[tuple[str, float, str]] = []
        time_values: list[tuple[str, float, str]] = []
        peaks: list[tuple[str, float, str]] = []
        for row in rows:
            if isinstance(row, dict):
                name = str(row["zone_name"])
                visit_count = float(row["unique_visits"])
                avg_time = float(row["avg_dwell_s"])
                peak = float(row.get("peak_occupancy", 0))
            else:
                name = row.zone_name
                visit_count = float(row.unique_visits)
                avg_time = float(row.avg_dwell_s)
                peak = float(row.peak_occupancy)
            visits.append((name, visit_count, str(int(visit_count))))
            time_values.append((name, avg_time, f"{avg_time:.1f}s"))
            peaks.append((name, peak, str(int(peak))))
        self.live_visits_rank.set_rows(visits, COLORS["accent-blue"])
        self.live_time_rank.set_rows(time_values, COLORS["accent-amber"])
        self.live_peak_rank.set_rows(peaks, COLORS["accent-violet"])

    def set_live_snapshot(self, snapshot: AnalyticsSnapshot, venue_map: VenueMapConfig) -> None:
        self.live_snapshot = snapshot
        self.live_venue_map = venue_map
        metrics_rows = list(snapshot.zone_metrics.values())
        peak = max((row.peak_occupancy for row in metrics_rows), default=0)
        active_booths = sum(1 for row in metrics_rows if row.current_occupancy > 0)
        avg_values = [value for value in snapshot.avg_dwell_times.values() if value > 0.0]
        mean_avg_dwell = sum(avg_values) / len(avg_values) if avg_values else 0.0
        self.live_occupancy_value.setText(str(snapshot.total_current_occupancy))
        self.live_visits_value.setText(str(snapshot.total_entries))
        self.live_avg_dwell_value.setText(f"{mean_avg_dwell:.1f}s")
        self.live_peak_value.setText(str(peak))
        self.live_active_booths_value.setText(str(active_booths))

        self._populate_metrics_table(self.live_table, metrics_rows)
        self._set_ranked_live_lists(metrics_rows)

        zone_names = [row.zone_name for row in metrics_rows]
        current_name = self.timeline_zone_combo.currentText()
        self.timeline_zone_combo.blockSignals(True)
        self.timeline_zone_combo.clear()
        self.timeline_zone_combo.addItems(zone_names)
        if current_name in zone_names:
            self.timeline_zone_combo.setCurrentText(current_name)
        self.timeline_zone_combo.blockSignals(False)
        self._last_live_timeline_refresh_at = 0.0
        self._refresh_live_timeline()

    def set_runtime_snapshot(self, snapshot: MultiCameraRuntimeSnapshot) -> None:
        self.runtime_snapshot = snapshot
        total_drops = sum(snapshot.dropped_frames_by_camera.values())
        missing = ", ".join(snapshot.missing_cameras) if snapshot.missing_cameras else "none"
        self.live_status_label.setText(
            f"{'Current session: #' + str(self.current_session_id) if self.current_session_id is not None else 'Current session: none'} | "
            f"Sync: {snapshot.session_sync_mode} | "
            f"Media t: {(snapshot.session_media_time_s or 0.0):.2f}s | "
            f"Drops: {total_drops} | "
            f"Missing: {missing} | "
            f"Map presences: {snapshot.active_map_presence_count}"
        )

    def _run_background(
        self,
        token: int,
        fn: Callable[[], object],
        on_finished: Callable[[int, Any], None],
        on_failed: Callable[[int, str], None],
    ) -> None:
        task = _BackgroundTask(token, fn)
        task.signals.finished.connect(on_finished)
        task.signals.failed.connect(on_failed)
        self._thread_pool.start(task)

    def reload_history(self) -> None:
        self._history_reload_token += 1
        token = self._history_reload_token
        self.refresh_history_button.setEnabled(False)
        self.history_summary_label.setText("Loading recorded sessions...")
        self._run_background(
            token,
            self.repository.list_sessions,
            self._apply_history_sessions,
            self._handle_history_reload_error,
        )

    def _apply_history_sessions(self, token: int, payload: object) -> None:
        if token != self._history_reload_token:
            return
        self.refresh_history_button.setEnabled(True)
        sessions = list(payload) if isinstance(payload, list) else []
        self._history_sessions = sessions
        current_session_id = self.session_selector.currentData()
        self.session_selector.blockSignals(True)
        self.session_selector.clear()
        for session in sessions:
            started_at = datetime.fromtimestamp(float(session["started_at"])).strftime("%Y-%m-%d %H:%M:%S")
            label = f"#{session['id']} | {started_at} | {session['camera_id']} | {session['source_label']}"
            self.session_selector.addItem(label, int(session["id"]))
        if isinstance(current_session_id, int):
            index = self.session_selector.findData(current_session_id)
            if index >= 0:
                self.session_selector.setCurrentIndex(index)
        self.session_selector.blockSignals(False)
        if self.session_selector.count() > 0:
            if self.session_selector.currentIndex() < 0:
                self.session_selector.setCurrentIndex(0)
            self._load_selected_session()
        else:
            self.history_summary_label.setText("No recorded sessions")
            self._history_loaded_session_id = None
            self._history_loaded_metrics = []
            self._history_loaded_visits = []
            self._history_selected_heatmap = None
            self._populate_metrics_table(self.history_table, [])
            self._populate_visit_sessions([])
            self._set_history_heatmap_preview(None, None, None)

    def _handle_history_reload_error(self, token: int, message: str) -> None:
        if token != self._history_reload_token:
            return
        self.refresh_history_button.setEnabled(True)
        self.history_summary_label.setText(f"Statistics unavailable: {message}")

    def _load_selected_session(self) -> None:
        session_id = self.session_selector.currentData()
        if not isinstance(session_id, int):
            return
        self._history_session_token += 1
        token = self._history_session_token
        summary = next((item for item in self._history_sessions if int(item["id"]) == session_id), None)
        self._history_loaded_session_id = None
        self._history_loaded_metrics = []
        self._history_loaded_visits = []
        self._populate_metrics_table(self.history_table, [])
        self._populate_visit_sessions([])
        if summary is not None:
            started_at = datetime.fromtimestamp(float(summary["started_at"])).strftime("%Y-%m-%d %H:%M:%S")
            ended_at = summary["ended_at"]
            ended_text = datetime.fromtimestamp(float(ended_at)).strftime("%Y-%m-%d %H:%M:%S") if ended_at else "running / interrupted"
            self.history_summary_label.setText(f"Loading session #{session_id} | start: {started_at} | end: {ended_text}")
        else:
            self.history_summary_label.setText(f"Loading session #{session_id}...")
        self._history_selected_heatmap = None
        self._set_history_heatmap_preview(session_id, None, None, "Loading heatmap...")
        self._style_chart(self.history_timeline_chart.chart(), "Loading selected session timeline")
        self._run_background(
            token,
            lambda session_id=session_id, summary=summary, venue_map=_clone_venue_map(self.live_venue_map): self._load_history_session_payload(
                session_id,
                summary,
                venue_map,
            ),
            self._apply_history_session,
            self._handle_history_session_error,
        )

    def _load_history_session_payload(
        self,
        session_id: int,
        summary: dict[str, object] | None,
        venue_map: VenueMapConfig,
    ) -> _HistorySessionPayload:
        metrics = self.repository.load_session_zone_metrics(session_id)
        visits = self.repository.load_booth_visit_sessions(session_id)
        heatmap = self.repository.load_session_heatmap(session_id)
        heatmap_preview = _render_heatmap_image(heatmap, 640, 420, venue_map) if heatmap is not None else None
        return _HistorySessionPayload(
            session_id=session_id,
            summary=summary,
            metrics=metrics,
            visits=visits,
            heatmap=heatmap,
            heatmap_preview=heatmap_preview,
        )

    def _apply_history_session(self, token: int, payload: object) -> None:
        if token != self._history_session_token or not isinstance(payload, _HistorySessionPayload):
            return
        if self.session_selector.currentData() != payload.session_id:
            return
        if payload.summary is not None:
            started_at = datetime.fromtimestamp(float(payload.summary["started_at"])).strftime("%Y-%m-%d %H:%M:%S")
            ended_at = payload.summary["ended_at"]
            ended_text = datetime.fromtimestamp(float(ended_at)).strftime("%Y-%m-%d %H:%M:%S") if ended_at else "running / interrupted"
            self.history_summary_label.setText(f"Session #{payload.session_id} | start: {started_at} | end: {ended_text}")

        metrics = payload.metrics
        self._history_loaded_session_id = payload.session_id
        self._history_loaded_metrics = metrics
        self._history_loaded_visits = payload.visits
        self._history_selected_heatmap = payload.heatmap
        self._populate_metrics_table(self.history_table, metrics)
        self._update_history_cards(metrics)
        zone_names = [str(row["zone_name"]) for row in metrics]
        current_name = self.history_timeline_zone_combo.currentText()
        self.history_timeline_zone_combo.blockSignals(True)
        self.history_timeline_zone_combo.clear()
        self.history_timeline_zone_combo.addItems(zone_names)
        if current_name in zone_names:
            self.history_timeline_zone_combo.setCurrentText(current_name)
        self.history_timeline_zone_combo.blockSignals(False)
        self._refresh_history_timeline()
        self._populate_visit_sessions(payload.visits)
        self._set_history_heatmap_preview(payload.session_id, payload.heatmap, payload.heatmap_preview)

    def _handle_history_session_error(self, token: int, message: str) -> None:
        if token != self._history_session_token:
            return
        self.history_summary_label.setText(f"Statistics unavailable: {message}")

    def _refresh_live_timeline(self) -> None:
        if self.current_session_id is None or not self.isVisible():
            self.live_timeline_chart.chart().removeAllSeries()
            self._style_chart(self.live_timeline_chart.chart(), "Live occupancy timeline")
            return
        now = time.monotonic()
        if now - self._last_live_timeline_refresh_at < 1.0:
            return
        zone_name = self.timeline_zone_combo.currentText()
        zone_id = self._zone_id_by_name(zone_name, self.live_venue_map)
        if zone_id is None:
            self.live_timeline_chart.chart().removeAllSeries()
            self._style_chart(self.live_timeline_chart.chart(), "Live occupancy timeline")
            return
        timeline = self.repository.load_zone_timeline(self.current_session_id, zone_id)
        self._last_live_timeline_refresh_at = now
        self._set_line_chart(self.live_timeline_chart.chart(), "Live occupancy timeline", timeline)

    def _refresh_history_timeline(self) -> None:
        session_id = self.session_selector.currentData()
        if not isinstance(session_id, int):
            return
        zone_name = self.history_timeline_zone_combo.currentText()
        zone_id = self._zone_id_by_name(zone_name, self.live_venue_map)
        if zone_id is None:
            for row in self._history_loaded_metrics:
                if str(row["zone_name"]) == zone_name:
                    zone_id = str(row["zone_id"])
                    break
        if zone_id is None:
            self.history_timeline_chart.chart().removeAllSeries()
            self._style_chart(self.history_timeline_chart.chart(), "Selected session occupancy timeline")
            return
        self._history_timeline_token += 1
        token = self._history_timeline_token
        self.history_timeline_chart.chart().removeAllSeries()
        self._style_chart(self.history_timeline_chart.chart(), "Loading selected session timeline")
        self._run_background(
            token,
            lambda session_id=session_id, zone_id=zone_id: _TimelinePayload(
                session_id=session_id,
                zone_id=zone_id,
                timeline=self.repository.load_zone_timeline(session_id, zone_id),
            ),
            self._apply_history_timeline,
            self._handle_history_timeline_error,
        )

    def _apply_history_timeline(self, token: int, payload: object) -> None:
        if token != self._history_timeline_token or not isinstance(payload, _TimelinePayload):
            return
        if self.session_selector.currentData() != payload.session_id:
            return
        self._set_line_chart(self.history_timeline_chart.chart(), "Selected session occupancy timeline", payload.timeline)

    def _handle_history_timeline_error(self, token: int, message: str) -> None:
        if token != self._history_timeline_token:
            return
        self.history_timeline_chart.chart().removeAllSeries()
        self._style_chart(self.history_timeline_chart.chart(), f"Timeline unavailable: {message}")

    def _populate_metrics_table(self, table: QTableWidget, rows: list[object]) -> None:
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            if isinstance(row, dict):
                values = [
                    str(row["zone_name"]),
                    str(row["zone_kind"]),
                    str(row["current_occupancy"]),
                    str(row["unique_visits"]),
                    f"{float(row['total_dwell_s']):.1f}",
                    f"{float(row['avg_dwell_s']):.1f}",
                    f"{float(row['median_dwell_s']):.1f}",
                    str(row.get("peak_occupancy", 0)),
                ]
            else:
                values = [
                    row.zone_name,
                    row.zone_kind,
                    str(row.current_occupancy),
                    str(row.unique_visits),
                    f"{row.total_dwell_s:.1f}",
                    f"{row.avg_dwell_s:.1f}",
                    f"{row.median_dwell_s:.1f}",
                    str(row.peak_occupancy),
                ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                table.setItem(row_index, column, item)
        table.resizeRowsToContents()

    def _populate_visit_sessions(self, sessions: list[dict[str, object]]) -> None:
        self.visit_sessions_table.setRowCount(len(sessions))
        for row_index, session in enumerate(sessions):
            source_cameras = ""
            try:
                source_cameras = ", ".join(json.loads(str(session["source_camera_ids_json"])))
            except Exception:
                source_cameras = str(session.get("source_camera_ids_json", ""))
            values = [
                str(session["visit_id"]),
                str(session["analytics_track_id"]),
                str(session["zone_id"]),
                datetime.fromtimestamp(float(session["entered_at"])).strftime("%H:%M:%S"),
                datetime.fromtimestamp(float(session["left_at"])).strftime("%H:%M:%S") if session["left_at"] is not None else "",
                f"{float(session['dwell_s']):.1f}",
                f"{session['dedup_mode']} | {source_cameras}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.visit_sessions_table.setItem(row_index, column, item)
        self.visit_sessions_table.resizeRowsToContents()

    def _export_live_csv(self) -> None:
        rows = []
        for metrics in self.live_snapshot.zone_metrics.values():
            rows.append(
                {
                    "booth": metrics.zone_name,
                    "kind": metrics.zone_kind,
                    "current_occupancy": metrics.current_occupancy,
                    "visits": metrics.unique_visits,
                    "total_dwell_s": round(metrics.total_dwell_s, 3),
                    "avg_dwell_s": round(metrics.avg_dwell_s, 3),
                    "median_dwell_s": round(metrics.median_dwell_s, 3),
                    "peak_occupancy": metrics.peak_occupancy,
                }
            )
        self._export_rows_to_csv(rows, "live_booth_analytics.csv")

    def _export_history_csv(self) -> None:
        session_id = self.session_selector.currentData()
        if not isinstance(session_id, int):
            QMessageBox.information(self, "Export CSV", "Select a history session first.")
            return
        rows = self._history_loaded_metrics if self._history_loaded_session_id == session_id else []
        self._export_rows_to_csv(rows, f"session_{session_id}_booth_summary.csv")

    def _export_sessions_csv(self) -> None:
        session_id = self.session_selector.currentData()
        if not isinstance(session_id, int):
            QMessageBox.information(self, "Export CSV", "Select a history session first.")
            return
        rows = self._history_loaded_visits if self._history_loaded_session_id == session_id else []
        normalized_rows = []
        for row in rows:
            normalized_rows.append(
                {
                    "visit_id": row["visit_id"],
                    "analytics_track_id": row["analytics_track_id"],
                    "zone_id": row["zone_id"],
                    "entered_at": row["entered_at"],
                    "left_at": row["left_at"],
                    "dwell_s": row["dwell_s"],
                    "source_camera_ids_json": row["source_camera_ids_json"],
                    "dedup_mode": row["dedup_mode"],
                }
            )
        self._export_rows_to_csv(normalized_rows, f"session_{session_id}_visit_sessions.csv")

    def export_preferred(self) -> None:
        if self.tabs.currentWidget() is self.live_tab:
            self._export_live_csv()
            return
        self._export_current_session_bundle()

    def _export_current_session_bundle(self) -> None:
        session_id = self.session_selector.currentData()
        if not isinstance(session_id, int):
            if self.current_session_id is None:
                QMessageBox.information(self, "Export", "No recorded session is available to export.")
                return
            self._export_live_csv()
            return
        target_dir = QFileDialog.getExistingDirectory(
            self,
            "Export session bundle",
            str(Path.cwd()),
        )
        if not target_dir:
            return
        self._history_export_token += 1
        token = self._history_export_token
        self.history_bundle_export_button.setEnabled(False)
        self.sessions_bundle_export_button.setEnabled(False)
        self.history_summary_label.setText(f"Exporting session #{session_id} bundle...")
        self._run_background(
            token,
            lambda session_id=session_id, output_dir=Path(target_dir), venue_map=_clone_venue_map(self.live_venue_map): self._write_session_bundle(
                session_id,
                output_dir,
                venue_map,
            ),
            self._apply_history_export,
            self._handle_history_export_error,
        )

    def _write_session_bundle(
        self,
        session_id: int,
        output_dir: Path,
        venue_map: VenueMapConfig,
    ) -> _ExportPayload:
        summary_rows = self.repository.load_session_zone_metrics(session_id)
        visits_rows = self.repository.load_booth_visit_sessions(session_id)
        heatmap = self.repository.load_session_heatmap(session_id)
        if not summary_rows and not visits_rows and heatmap is None:
            return _ExportPayload("Export", "There is no session data to export yet.")
        written_paths: list[Path] = []
        if summary_rows:
            summary_path = output_dir / f"session_{session_id}_booth_summary.csv"
            self._write_csv(summary_path, summary_rows)
            written_paths.append(summary_path)
        if visits_rows:
            visit_rows = [
                {
                    "visit_id": row["visit_id"],
                    "analytics_track_id": row["analytics_track_id"],
                    "zone_id": row["zone_id"],
                    "entered_at": row["entered_at"],
                    "left_at": row["left_at"],
                    "dwell_s": row["dwell_s"],
                    "source_camera_ids_json": row["source_camera_ids_json"],
                    "dedup_mode": row["dedup_mode"],
                }
                for row in visits_rows
            ]
            visits_path = output_dir / f"session_{session_id}_visit_sessions.csv"
            self._write_csv(visits_path, visit_rows)
            written_paths.append(visits_path)
        if heatmap is not None:
            heatmap_path = output_dir / f"session_{session_id}_heatmap.png"
            _write_heatmap_png(heatmap_path, heatmap, venue_map)
            written_paths.append(heatmap_path)
        return _ExportPayload("Export", "Saved files:\n" + "\n".join(str(path) for path in written_paths))

    def _export_history_heatmap_png(self) -> None:
        session_id = self.session_selector.currentData()
        if not isinstance(session_id, int):
            QMessageBox.information(self, "Export heatmap", "Select a history session first.")
            return
        heatmap = self._history_selected_heatmap
        if heatmap is None:
            QMessageBox.information(self, "Export heatmap", "This session has no heatmap data.")
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export heatmap PNG",
            str(Path.cwd() / f"session_{session_id}_heatmap.png"),
            "PNG files (*.png)",
        )
        if not path_str:
            return
        path = Path(path_str)
        self._history_export_token += 1
        token = self._history_export_token
        self.history_heatmap_export_button.setEnabled(False)
        self.history_summary_label.setText(f"Exporting session #{session_id} heatmap...")
        self._run_background(
            token,
            lambda path=path, heatmap=heatmap, venue_map=_clone_venue_map(self.live_venue_map): self._write_history_heatmap_export(
                path,
                heatmap,
                venue_map,
            ),
            self._apply_history_export,
            self._handle_history_export_error,
        )

    @staticmethod
    def _write_history_heatmap_export(
        path: Path,
        heatmap: HeatmapSnapshot,
        venue_map: VenueMapConfig,
    ) -> _ExportPayload:
        _write_heatmap_png(path, heatmap, venue_map)
        return _ExportPayload("Export heatmap", f"Saved PNG to:\n{path}")

    def _apply_history_export(self, token: int, payload: object) -> None:
        if token != self._history_export_token or not isinstance(payload, _ExportPayload):
            return
        self._restore_history_export_buttons()
        QMessageBox.information(self, payload.title, payload.message)

    def _handle_history_export_error(self, token: int, message: str) -> None:
        if token != self._history_export_token:
            return
        self._restore_history_export_buttons()
        QMessageBox.warning(self, "Export", f"Export failed: {message}")

    def _restore_history_export_buttons(self) -> None:
        self.history_bundle_export_button.setEnabled(True)
        self.sessions_bundle_export_button.setEnabled(True)
        self.history_heatmap_export_button.setEnabled(
            isinstance(self.session_selector.currentData(), int) and self._history_selected_heatmap is not None
        )

    def _export_rows_to_csv(self, rows: list[dict[str, object]], default_name: str) -> None:
        if not rows:
            QMessageBox.information(self, "Export CSV", "There is no data to export yet.")
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            str(Path.cwd() / default_name),
            "CSV files (*.csv)",
        )
        if not path_str:
            return
        path = Path(path_str)
        self._write_csv(path, rows)
        QMessageBox.information(self, "Export CSV", f"Saved CSV to:\n{path}")

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _set_history_heatmap_preview(
        self,
        session_id: int | None,
        heatmap: HeatmapSnapshot | None,
        image: QImage | None,
        empty_text: str = "No heatmap recorded",
    ) -> None:
        self.history_heatmap_export_button.setEnabled(session_id is not None and heatmap is not None)
        if heatmap is None or image is None:
            self.history_heatmap_preview.setText(empty_text)
            self.history_heatmap_preview.setPixmap(QPixmap())
            return
        pixmap = QPixmap.fromImage(image).scaled(
            self.history_heatmap_preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.history_heatmap_preview.setText("")
        self.history_heatmap_preview.setPixmap(pixmap)

    def _write_heatmap_png(self, path: Path, heatmap: HeatmapSnapshot) -> None:
        _write_heatmap_png(path, heatmap, self.live_venue_map)

    def _render_heatmap_image(self, heatmap: HeatmapSnapshot, width: int, height: int) -> QImage:
        return _render_heatmap_image(heatmap, width, height, self.live_venue_map)

    def _draw_heatmap_zones(
        self,
        painter: QPainter,
        world_to_image,
    ) -> None:
        _draw_heatmap_zones(painter, world_to_image, self.live_venue_map)

    @staticmethod
    def _create_metric_card(title: str, value: str, accent: str) -> tuple[QFrame, QLabel]:
        frame = QFrame()
        frame.setObjectName("MetricCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(5)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        accent_label = QLabel()
        accent_label.setStyleSheet(
            f"background: {accent}; border-radius: 5px;"
        )
        accent_label.setFixedSize(10, 10)
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {COLORS['text-muted']}; font-size: 12px; font-weight: 600;")
        title_row.addWidget(accent_label)
        title_row.addWidget(title_label, 1)

        value_label = QLabel(value)
        value_label.setStyleSheet(f"color: {COLORS['text-primary']}; font-size: 26px; font-weight: 800;")
        layout.addLayout(title_row)
        layout.addWidget(value_label)
        return frame, value_label

    @staticmethod
    def _configure_table(table: QTableWidget, column_widths: list[int]) -> None:
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.setShowGrid(False)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        header.setMinimumSectionSize(54)
        for column, width in enumerate(column_widths):
            table.setColumnWidth(column, width)
        table.verticalHeader().setVisible(False)

    def _create_chart_view(self, title: str) -> QChartView:
        chart = QChart()
        self._style_chart(chart, title)
        view = QChartView(chart)
        view.setRenderHint(QPainter.Antialiasing)
        view.setFrameShape(QFrame.NoFrame)
        return view

    @staticmethod
    def _style_chart(chart: QChart, title: str) -> None:
        chart.setTitle(title)
        chart.setBackgroundVisible(True)
        chart.setBackgroundRoundness(12)
        chart.setBackgroundBrush(QColor(COLORS["bg-shell"]))
        chart.setPlotAreaBackgroundVisible(True)
        chart.setPlotAreaBackgroundBrush(QColor(COLORS["bg-input"]))
        chart.setTitleBrush(QColor(COLORS["text-primary"]))
        chart.legend().setVisible(False)

    @staticmethod
    def _create_panel(title: str, subtitle: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName("PanelCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        title_label = QLabel(title)
        title_label.setObjectName("SectionTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("MutedText")
        subtitle_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        return frame

    def _set_line_chart(self, chart: QChart, title: str, timeline: list[dict[str, object]]) -> None:
        chart.removeAllSeries()
        self._style_chart(chart, title)
        for axis in chart.axes():
            chart.removeAxis(axis)
        if not timeline:
            return
        baseline = float(timeline[0]["ts"])
        series = QLineSeries()
        series.setColor(QColor(COLORS["accent-blue"]))
        series.setPen(QPen(QColor(COLORS["accent-blue"]), 2.2))
        previous_value = float(timeline[0]["occupancy"])
        series.append(0.0, previous_value)
        for item in timeline[1:]:
            x_value = float(item["ts"]) - baseline
            current_value = float(item["occupancy"])
            series.append(x_value, previous_value)
            series.append(x_value, current_value)
            previous_value = current_value
        chart.addSeries(series)

        axis_x = QValueAxis()
        axis_x.setTitleText("Seconds since first snapshot")
        axis_x.setTitleBrush(QColor(COLORS["text-subtle"]))
        axis_x.setLabelsColor(QColor(COLORS["text-muted"]))
        axis_x.setGridLineColor(QColor(COLORS["border-strong"]))

        axis_y = QValueAxis()
        axis_y.setTitleText("Occupancy")
        axis_y.setTitleBrush(QColor(COLORS["text-subtle"]))
        axis_y.setLabelsColor(QColor(COLORS["text-muted"]))
        axis_y.setGridLineColor(QColor(COLORS["border-strong"]))
        axis_y.setLabelFormat("%d")

        chart.addAxis(axis_x, Qt.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)

    def _update_history_cards(self, metrics: list[dict[str, object]]) -> None:
        total_visits = sum(int(row["unique_visits"]) for row in metrics)
        peak_occupancy = max((int(row.get("peak_occupancy", 0)) for row in metrics), default=0)
        avg_values = [float(row["avg_dwell_s"]) for row in metrics if float(row["avg_dwell_s"]) > 0.0]
        booths_with_traffic = sum(1 for row in metrics if int(row["unique_visits"]) > 0)
        mean_avg_dwell = sum(avg_values) / len(avg_values) if avg_values else 0.0
        self.history_visits_value.setText(str(total_visits))
        self.history_avg_value.setText(f"{mean_avg_dwell:.1f}s")
        self.history_peak_value.setText(str(peak_occupancy))
        self.history_booths_value.setText(str(booths_with_traffic))

    @staticmethod
    def _zone_id_by_name(zone_name: str, venue_map: VenueMapConfig) -> str | None:
        for zone in venue_map.zones:
            if zone.name == zone_name:
                return zone.zone_id
        return None
