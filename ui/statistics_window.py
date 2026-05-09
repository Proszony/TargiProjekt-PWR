from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime

from PySide6.QtCharts import QBarCategoryAxis, QBarSeries, QBarSet, QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.models import AnalyticsSnapshot, VenueMapConfig
from core.statistics_repository import StatisticsRepository


class StatisticsWindow(QMainWindow):
    def __init__(self, repository: StatisticsRepository, parent=None) -> None:
        super().__init__(parent)
        self.repository = repository
        self.current_session_id: int | None = None
        self.live_snapshot = AnalyticsSnapshot(timestamp=0.0)
        self.live_venue_map = VenueMapConfig()
        self._last_live_timeline_refresh_at = 0.0
        self.setWindowTitle("Statistics")
        self.resize(1200, 800)
        self._build_ui()
        self.reload_history()

    def _build_ui(self) -> None:
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        self.live_tab = QWidget()
        self.history_tab = QWidget()
        self.events_tab = QWidget()
        self.tabs.addTab(self.live_tab, "Live")
        self.tabs.addTab(self.history_tab, "History")
        self.tabs.addTab(self.events_tab, "Events")

        self._build_live_tab()
        self._build_history_tab()
        self._build_events_tab()

    def _build_live_tab(self) -> None:
        layout = QVBoxLayout(self.live_tab)
        summary_row = QHBoxLayout()
        self.live_active_label = QLabel("Active tracks: 0")
        self.live_entries_label = QLabel("Total entries: 0")
        self.live_returns_label = QLabel("Total returns: 0")
        self.live_dwell_label = QLabel("Total dwell: 0.0s")
        for widget in (
            self.live_active_label,
            self.live_entries_label,
            self.live_returns_label,
            self.live_dwell_label,
        ):
            widget.setAlignment(Qt.AlignCenter)
            summary_row.addWidget(widget)

        self.live_table = QTableWidget(0, 7)
        self.live_table.setHorizontalHeaderLabels(
            [
                "Zone",
                "Kind",
                "Occupancy",
                "Unique entries",
                "Total dwell [s]",
                "Average dwell [s]",
                "Returns",
            ]
        )
        self.live_table.horizontalHeader().setStretchLastSection(True)

        chart_row = QHBoxLayout()
        self.live_occupancy_chart = self._create_chart_view("Occupancy per booth")
        self.live_entries_chart = self._create_chart_view("Unique entries per booth")
        self.live_returns_chart = self._create_chart_view("Returns per booth")
        chart_row.addWidget(self.live_occupancy_chart, 1)
        chart_row.addWidget(self.live_entries_chart, 1)
        chart_row.addWidget(self.live_returns_chart, 1)

        self.timeline_zone_combo = QComboBox()
        self.live_timeline_chart = self._create_chart_view("Occupancy timeline")
        self.timeline_zone_combo.currentTextChanged.connect(self._refresh_live_timeline)

        timeline_row = QVBoxLayout()
        timeline_row.addWidget(QLabel("Timeline zone"))
        timeline_row.addWidget(self.timeline_zone_combo)
        timeline_row.addWidget(self.live_timeline_chart, 1)

        layout.addLayout(summary_row)
        layout.addWidget(self.live_table, 1)
        layout.addLayout(chart_row)
        layout.addLayout(timeline_row, 1)

    def _build_history_tab(self) -> None:
        layout = QVBoxLayout(self.history_tab)
        top_row = QHBoxLayout()
        self.session_selector = QComboBox()
        self.refresh_history_button = QPushButton("Refresh history")
        top_row.addWidget(QLabel("Session"))
        top_row.addWidget(self.session_selector, 1)
        top_row.addWidget(self.refresh_history_button)

        self.history_summary_label = QLabel("No session selected")
        self.history_table = QTableWidget(0, 7)
        self.history_table.setHorizontalHeaderLabels(
            [
                "Zone",
                "Kind",
                "Occupancy",
                "Unique entries",
                "Total dwell [s]",
                "Average dwell [s]",
                "Returns",
            ]
        )
        self.history_table.horizontalHeader().setStretchLastSection(True)

        self.history_timeline_zone_combo = QComboBox()
        self.history_timeline_chart = self._create_chart_view("Selected session timeline")
        self.history_timeline_zone_combo.currentTextChanged.connect(self._refresh_history_timeline)

        layout.addLayout(top_row)
        layout.addWidget(self.history_summary_label)
        layout.addWidget(self.history_table, 1)
        layout.addWidget(QLabel("Timeline zone"))
        layout.addWidget(self.history_timeline_zone_combo)
        layout.addWidget(self.history_timeline_chart, 1)

        self.refresh_history_button.clicked.connect(self.reload_history)
        self.session_selector.currentIndexChanged.connect(self._load_selected_session)

    def _build_events_tab(self) -> None:
        layout = QVBoxLayout(self.events_tab)
        self.events_table = QTableWidget(0, 5)
        self.events_table.setHorizontalHeaderLabels(
            ["Timestamp", "Global ID", "Event", "Zone", "Payload"]
        )
        self.events_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.events_table)

    def set_current_session_id(self, session_id: int | None) -> None:
        self.current_session_id = session_id
        self._last_live_timeline_refresh_at = 0.0
        self._refresh_live_timeline()

    def set_live_snapshot(self, snapshot: AnalyticsSnapshot, venue_map: VenueMapConfig) -> None:
        self.live_snapshot = snapshot
        self.live_venue_map = venue_map
        total_dwell = sum(snapshot.dwell_times.values())
        self.live_active_label.setText(f"Active tracks: {len(snapshot.active_global_tracks)}")
        self.live_entries_label.setText(f"Total entries: {snapshot.total_entries}")
        self.live_returns_label.setText(f"Total returns: {snapshot.session_total_returns}")
        self.live_dwell_label.setText(f"Total dwell: {total_dwell:.1f}s")

        metrics_rows = list(snapshot.zone_metrics.values())
        self._populate_metrics_table(self.live_table, metrics_rows)
        self._set_bar_chart(
            self.live_occupancy_chart.chart(),
            "Occupancy per booth",
            metrics_rows,
            lambda row: row.current_occupancy,
        )
        self._set_bar_chart(
            self.live_entries_chart.chart(),
            "Unique entries per booth",
            metrics_rows,
            lambda row: row.unique_entries,
        )
        self._set_bar_chart(
            self.live_returns_chart.chart(),
            "Returns per booth",
            metrics_rows,
            lambda row: row.return_count,
        )

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

    def reload_history(self) -> None:
        try:
            sessions = self.repository.list_sessions()
        except sqlite3.Error as exc:
            self.history_summary_label.setText(f"Statistics unavailable: {exc}")
            return
        self.session_selector.blockSignals(True)
        self.session_selector.clear()
        for session in sessions:
            started_at = datetime.fromtimestamp(float(session["started_at"])).strftime("%Y-%m-%d %H:%M:%S")
            label = f"#{session['id']} | {started_at} | {session['camera_id']} | {session['source_label']}"
            self.session_selector.addItem(label, int(session["id"]))
        self.session_selector.blockSignals(False)
        if self.session_selector.count() > 0:
            self.session_selector.setCurrentIndex(0)
            self._load_selected_session()
        else:
            self.history_summary_label.setText("No recorded sessions")
            self._populate_metrics_table(self.history_table, [])
            self._populate_events([])

    def _load_selected_session(self) -> None:
        session_id = self.session_selector.currentData()
        if not isinstance(session_id, int):
            return
        try:
            sessions = self.repository.list_sessions()
        except sqlite3.Error as exc:
            self.history_summary_label.setText(f"Statistics unavailable: {exc}")
            return
        summary = next((item for item in sessions if int(item["id"]) == session_id), None)
        if summary is not None:
            started_at = datetime.fromtimestamp(float(summary["started_at"])).strftime("%Y-%m-%d %H:%M:%S")
            ended_at = summary["ended_at"]
            ended_text = (
                datetime.fromtimestamp(float(ended_at)).strftime("%Y-%m-%d %H:%M:%S")
                if ended_at
                else "running / interrupted"
            )
            self.history_summary_label.setText(
                f"Session #{session_id} | start: {started_at} | end: {ended_text}"
            )

        try:
            metrics = self.repository.load_session_zone_metrics(session_id)
        except sqlite3.Error as exc:
            self.history_summary_label.setText(f"Statistics unavailable: {exc}")
            return
        self._populate_metrics_table(self.history_table, metrics)
        zone_names = [str(row["zone_name"]) for row in metrics]
        current_name = self.history_timeline_zone_combo.currentText()
        self.history_timeline_zone_combo.blockSignals(True)
        self.history_timeline_zone_combo.clear()
        self.history_timeline_zone_combo.addItems(zone_names)
        if current_name in zone_names:
            self.history_timeline_zone_combo.setCurrentText(current_name)
        self.history_timeline_zone_combo.blockSignals(False)
        self._refresh_history_timeline()
        try:
            self._populate_events(self.repository.load_events(session_id))
        except sqlite3.Error as exc:
            self.history_summary_label.setText(f"Statistics unavailable: {exc}")

    def _refresh_live_timeline(self) -> None:
        if self.current_session_id is None or not self.isVisible():
            self.live_timeline_chart.chart().removeAllSeries()
            return
        now = time.monotonic()
        if now - self._last_live_timeline_refresh_at < 1.0:
            return
        zone_name = self.timeline_zone_combo.currentText()
        zone_id = self._zone_id_by_name(zone_name, self.live_venue_map)
        if zone_id is None:
            self.live_timeline_chart.chart().removeAllSeries()
            return
        try:
            timeline = self.repository.load_zone_timeline(self.current_session_id, zone_id)
        except sqlite3.Error:
            return
        self._last_live_timeline_refresh_at = now
        self._set_line_chart(self.live_timeline_chart.chart(), "Occupancy timeline", timeline)

    def _refresh_history_timeline(self) -> None:
        session_id = self.session_selector.currentData()
        if not isinstance(session_id, int):
            return
        zone_name = self.history_timeline_zone_combo.currentText()
        zone_id = self._zone_id_by_name(zone_name, self.live_venue_map)
        if zone_id is None:
            for row in self.repository.load_session_zone_metrics(session_id):
                if str(row["zone_name"]) == zone_name:
                    zone_id = str(row["zone_id"])
                    break
        if zone_id is None:
            self.history_timeline_chart.chart().removeAllSeries()
            return
        try:
            timeline = self.repository.load_zone_timeline(session_id, zone_id)
        except sqlite3.Error as exc:
            self.history_summary_label.setText(f"Statistics unavailable: {exc}")
            return
        self._set_line_chart(self.history_timeline_chart.chart(), "Selected session timeline", timeline)

    def _populate_metrics_table(self, table: QTableWidget, rows: list[object]) -> None:
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            if isinstance(row, dict):
                values = [
                    str(row["zone_name"]),
                    str(row["zone_kind"]),
                    str(row["current_occupancy"]),
                    str(row["unique_entries"]),
                    f"{float(row['total_dwell_s']):.1f}",
                    f"{float(row['avg_dwell_s']):.1f}",
                    str(row["return_count"]),
                ]
            else:
                values = [
                    row.zone_name,
                    row.zone_kind,
                    str(row.current_occupancy),
                    str(row.unique_entries),
                    f"{row.total_dwell_s:.1f}",
                    f"{row.avg_dwell_s:.1f}",
                    str(row.return_count),
                ]
            for column, value in enumerate(values):
                table.setItem(row_index, column, QTableWidgetItem(value))
        table.resizeColumnsToContents()

    def _populate_events(self, events: list[dict[str, object]]) -> None:
        self.events_table.setRowCount(len(events))
        for row_index, event in enumerate(events):
            payload = ""
            try:
                payload = json.dumps(json.loads(str(event["payload_json"])), ensure_ascii=True)
            except Exception:
                payload = str(event.get("payload_json", ""))
            values = [
                datetime.fromtimestamp(float(event["ts"])).strftime("%H:%M:%S"),
                str(event["global_track_id"]),
                str(event["event_type"]),
                str(event["zone_id"] or ""),
                payload,
            ]
            for column, value in enumerate(values):
                self.events_table.setItem(row_index, column, QTableWidgetItem(value))
        self.events_table.resizeColumnsToContents()

    @staticmethod
    def _create_chart_view(title: str) -> QChartView:
        chart = QChart()
        chart.setTitle(title)
        view = QChartView(chart)
        view.setRenderHint(QPainter.Antialiasing)
        return view

    @staticmethod
    def _set_bar_chart(chart: QChart, title: str, rows: list[object], value_getter) -> None:
        chart.removeAllSeries()
        chart.setTitle(title)
        if not rows:
            return
        categories: list[str] = []
        values = QBarSet(title)
        for row in rows:
            categories.append(row["zone_name"] if isinstance(row, dict) else row.zone_name)
            values.append(float(value_getter(row)))
        series = QBarSeries()
        series.append(values)
        chart.addSeries(series)
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        axis_y = QValueAxis()
        axis_y.setLabelFormat("%d")
        chart.createDefaultAxes()
        chart.setAxisX(axis_x, series)
        chart.setAxisY(axis_y, series)
        chart.legend().hide()

    @staticmethod
    def _set_line_chart(chart: QChart, title: str, timeline: list[dict[str, object]]) -> None:
        chart.removeAllSeries()
        chart.setTitle(title)
        if not timeline:
            return
        baseline = float(timeline[0]["ts"])
        series = QLineSeries()
        for item in timeline:
            series.append(float(item["ts"]) - baseline, float(item["occupancy"]))
        chart.addSeries(series)
        axis_x = QValueAxis()
        axis_x.setTitleText("Seconds since first snapshot")
        axis_y = QValueAxis()
        axis_y.setTitleText("Occupancy")
        axis_y.setLabelFormat("%d")
        chart.createDefaultAxes()
        chart.setAxisX(axis_x, series)
        chart.setAxisY(axis_y, series)
        chart.legend().hide()

    @staticmethod
    def _zone_id_by_name(zone_name: str, venue_map: VenueMapConfig) -> str | None:
        for zone in venue_map.zones:
            if zone.name == zone_name:
                return zone.zone_id
        return None
