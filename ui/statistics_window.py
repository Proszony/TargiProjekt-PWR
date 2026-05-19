from __future__ import annotations

import csv
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCharts import QBarCategoryAxis, QBarSeries, QBarSet, QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
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
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.models import AnalyticsSnapshot, MultiCameraRuntimeSnapshot, VenueMapConfig
from core.statistics_repository import StatisticsRepository


class StatisticsWindow(QMainWindow):
    def __init__(self, repository: StatisticsRepository, parent=None) -> None:
        super().__init__(parent)
        self.repository = repository
        self.current_session_id: int | None = None
        self.live_snapshot = AnalyticsSnapshot(timestamp=0.0)
        self.runtime_snapshot = MultiCameraRuntimeSnapshot(timestamp=0.0)
        self.live_venue_map = VenueMapConfig()
        self._last_live_timeline_refresh_at = 0.0
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
            "Average dwell", "0.0s", "#f59e0b"
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
                "Current occupancy",
                "Visits",
                "Total dwell [s]",
                "Avg dwell [s]",
                "Median dwell [s]",
                "Peak occupancy",
            ]
        )
        self._configure_table(self.live_table)

        timeline_header = QHBoxLayout()
        self.timeline_zone_combo = QComboBox()
        self.timeline_zone_combo.setMinimumWidth(260)
        timeline_header.addWidget(QLabel("Timeline booth"))
        timeline_header.addWidget(self.timeline_zone_combo, 0)
        timeline_header.addStretch(1)
        self.live_timeline_chart = self._create_chart_view("Live occupancy timeline")

        main_row = QHBoxLayout()
        main_row.setSpacing(12)
        live_table_panel = self._create_panel("Booth snapshot", "Current occupancy and dwell summary per booth")
        live_table_panel.layout().addWidget(self.live_table)  # type: ignore[union-attr]
        live_timeline_panel = self._create_panel("Occupancy trend", "Live stepped occupancy for the selected booth")
        live_timeline_panel.layout().addLayout(timeline_header)  # type: ignore[union-attr]
        live_timeline_panel.layout().addWidget(self.live_timeline_chart, 1)  # type: ignore[union-attr]
        main_row.addWidget(live_table_panel, 7)
        main_row.addWidget(live_timeline_panel, 6)

        charts_row = QHBoxLayout()
        charts_row.setSpacing(12)
        self.live_visits_chart = self._create_chart_view("Visits by booth")
        self.live_dwell_chart = self._create_chart_view("Average dwell by booth")
        self.live_peak_chart = self._create_chart_view("Peak occupancy by booth")
        visits_panel = self._create_panel("Visits", "Which booths are drawing the most traffic")
        visits_panel.layout().addWidget(self.live_visits_chart)  # type: ignore[union-attr]
        dwell_panel = self._create_panel("Dwell", "Average time spent at each booth")
        dwell_panel.layout().addWidget(self.live_dwell_chart)  # type: ignore[union-attr]
        peak_panel = self._create_panel("Peak load", "Highest simultaneous occupancy per booth")
        peak_panel.layout().addWidget(self.live_peak_chart)  # type: ignore[union-attr]
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
            "Average dwell", "0.0s", "#f59e0b"
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
                "Final occupancy",
                "Visits",
                "Total dwell [s]",
                "Avg dwell [s]",
                "Median dwell [s]",
                "Peak occupancy",
            ]
        )
        self._configure_table(self.history_table)

        timeline_header = QHBoxLayout()
        self.history_timeline_zone_combo = QComboBox()
        self.history_timeline_zone_combo.setMinimumWidth(260)
        timeline_header.addWidget(QLabel("Timeline booth"))
        timeline_header.addWidget(self.history_timeline_zone_combo, 0)
        timeline_header.addStretch(1)

        self.history_timeline_chart = self._create_chart_view("Selected session occupancy timeline")

        history_main_row = QHBoxLayout()
        history_main_row.setSpacing(12)
        history_table_panel = self._create_panel("Session booth summary", "Final occupancy, visits and dwell metrics")
        history_table_panel.layout().addWidget(self.history_table)  # type: ignore[union-attr]
        history_timeline_panel = self._create_panel("Session timeline", "Selected booth occupancy across the session")
        history_timeline_panel.layout().addLayout(timeline_header)  # type: ignore[union-attr]
        history_timeline_panel.layout().addWidget(self.history_timeline_chart, 1)  # type: ignore[union-attr]
        history_main_row.addWidget(history_table_panel, 7)
        history_main_row.addWidget(history_timeline_panel, 6)

        layout.addLayout(top_row)
        layout.addWidget(self.history_summary_label)
        layout.addWidget(self.history_hint_label)
        layout.addLayout(history_cards)
        layout.addLayout(history_main_row, 1)

        self.refresh_history_button.clicked.connect(self.reload_history)
        self.history_export_button.clicked.connect(self._export_history_csv)
        self.history_bundle_export_button.clicked.connect(self._export_current_session_bundle)
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
        top_row.addWidget(self.sessions_bundle_export_button)
        self.sessions_export_button = QPushButton("Export CSV")
        self.sessions_export_button.clicked.connect(self._export_sessions_csv)
        top_row.addWidget(self.sessions_export_button)

        self.visit_sessions_table = QTableWidget(0, 7)
        self.visit_sessions_table.setHorizontalHeaderLabels(
            [
                "Visit ID",
                "Analytics track",
                "Booth",
                "Entered",
                "Left",
                "Dwell [s]",
                "Dedup mode",
            ]
        )
        self._configure_table(self.visit_sessions_table)

        layout.addLayout(top_row)
        layout.addWidget(self.visit_sessions_table)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0f172a;
                color: #e5e7eb;
            }
            QTabWidget::pane {
                border: 1px solid #243041;
                background: #111827;
            }
            QTabBar::tab {
                background: #172033;
                color: #cbd5e1;
                padding: 10px 16px;
                border: 1px solid #243041;
                border-bottom: none;
                min-width: 84px;
            }
            QTabBar::tab:selected {
                background: #1e293b;
                color: #f8fafc;
            }
            QFrame#PanelCard, QGroupBox#PanelCard {
                background: #111827;
                border: 1px solid #243041;
                border-radius: 14px;
            }
            QPushButton {
                background: #1d4ed8;
                color: #f8fafc;
                border: none;
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #2563eb;
            }
            QComboBox {
                background: #111827;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 6px 10px;
                min-height: 28px;
            }
            QLabel#SectionTitle {
                font-size: 16px;
                font-weight: 700;
                color: #f8fafc;
            }
            QLabel#MutedText {
                color: #94a3b8;
            }
            QLabel#HintText {
                color: #60a5fa;
            }
            QTableWidget {
                background: #0b1220;
                alternate-background-color: #101827;
                gridline-color: #1f2937;
                border: 1px solid #243041;
                border-radius: 10px;
                selection-background-color: #1d4ed8;
                selection-color: #f8fafc;
            }
            QHeaderView::section {
                background: #162033;
                color: #cbd5e1;
                border: none;
                border-right: 1px solid #243041;
                padding: 8px;
                font-weight: 600;
            }
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
        self._set_bar_chart(self.live_visits_chart.chart(), "Visits by booth", metrics_rows, "unique_visits")
        self._set_bar_chart(self.live_dwell_chart.chart(), "Average dwell by booth", metrics_rows, "avg_dwell_s")
        self._set_bar_chart(self.live_peak_chart.chart(), "Peak occupancy by booth", metrics_rows, "peak_occupancy")

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
            f"Map presences: {snapshot.active_map_presence_count} | "
            f"Overlap merges: {snapshot.merged_map_presence_count} | "
            f"Rejected g/t/m: {snapshot.map_presence_matches_rejected_geometry}/"
            f"{snapshot.map_presence_matches_rejected_time}/"
            f"{snapshot.map_presence_matches_rejected_margin}"
        )

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
            self._populate_visit_sessions([])

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
            ended_text = datetime.fromtimestamp(float(ended_at)).strftime("%Y-%m-%d %H:%M:%S") if ended_at else "running / interrupted"
            self.history_summary_label.setText(f"Session #{session_id} | start: {started_at} | end: {ended_text}")

        metrics = self.repository.load_session_zone_metrics(session_id)
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
        self._populate_visit_sessions(self.repository.load_booth_visit_sessions(session_id))

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
            for row in self.repository.load_session_zone_metrics(session_id):
                if str(row["zone_name"]) == zone_name:
                    zone_id = str(row["zone_id"])
                    break
        if zone_id is None:
            self.history_timeline_chart.chart().removeAllSeries()
            self._style_chart(self.history_timeline_chart.chart(), "Selected session occupancy timeline")
            return
        timeline = self.repository.load_zone_timeline(session_id, zone_id)
        self._set_line_chart(self.history_timeline_chart.chart(), "Selected session occupancy timeline", timeline)

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
        rows = self.repository.load_session_zone_metrics(session_id)
        self._export_rows_to_csv(rows, f"session_{session_id}_booth_summary.csv")

    def _export_sessions_csv(self) -> None:
        session_id = self.session_selector.currentData()
        if not isinstance(session_id, int):
            QMessageBox.information(self, "Export CSV", "Select a history session first.")
            return
        rows = self.repository.load_booth_visit_sessions(session_id)
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
        summary_rows = self.repository.load_session_zone_metrics(session_id)
        visits_rows = self.repository.load_booth_visit_sessions(session_id)
        if not summary_rows and not visits_rows:
            QMessageBox.information(self, "Export", "There is no session data to export yet.")
            return
        target_dir = QFileDialog.getExistingDirectory(
            self,
            "Export session bundle",
            str(Path.cwd()),
        )
        if not target_dir:
            return
        output_dir = Path(target_dir)
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
        QMessageBox.information(
            self,
            "Export",
            "Saved files:\n" + "\n".join(str(path) for path in written_paths),
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

    @staticmethod
    def _create_metric_card(title: str, value: str, accent: str) -> tuple[QFrame, QLabel]:
        frame = QFrame()
        frame.setObjectName("MetricCard")
        frame.setStyleSheet(
            f"""
            QFrame#MetricCard {{
                background: #111827;
                border: 1px solid #243041;
                border-left: 4px solid {accent};
                border-radius: 12px;
            }}
            """
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #94a3b8; font-size: 12px; font-weight: 600;")
        value_label = QLabel(value)
        value_label.setStyleSheet("color: #f8fafc; font-size: 26px; font-weight: 800;")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        return frame, value_label

    @staticmethod
    def _configure_table(table: QTableWidget) -> None:
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.setShowGrid(False)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
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
        chart.setBackgroundBrush(QColor("#111827"))
        chart.setPlotAreaBackgroundVisible(True)
        chart.setPlotAreaBackgroundBrush(QColor("#0b1220"))
        chart.setTitleBrush(QColor("#f8fafc"))
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

    def _set_bar_chart(self, chart: QChart, title: str, rows: list[object], attribute: str) -> None:
        chart.removeAllSeries()
        self._style_chart(chart, title)
        for axis in chart.axes():
            chart.removeAxis(axis)
        if not rows:
            return
        categories: list[str] = []
        values = QBarSet(title)
        values.setColor(QColor("#38bdf8" if attribute == "unique_visits" else "#22c55e" if attribute == "current_occupancy" else "#f59e0b"))
        values.setBorderColor(QColor("#00000000"))
        for row in rows:
            if isinstance(row, dict):
                categories.append(str(row["zone_name"]))
                values.append(float(row[attribute]))
            else:
                categories.append(row.zone_name)
                values.append(float(getattr(row, attribute)))
        series = QBarSeries()
        series.append(values)
        chart.addSeries(series)

        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        axis_x.setLabelsColor(QColor("#cbd5e1"))
        axis_x.setGridLineVisible(False)

        axis_y = QValueAxis()
        axis_y.setLabelsColor(QColor("#cbd5e1"))
        axis_y.setGridLineColor(QColor("#243041"))
        axis_y.setMinorGridLineVisible(False)
        axis_y.setLabelFormat("%d" if attribute in {"current_occupancy", "unique_visits", "peak_occupancy"} else "%.1f")

        chart.addAxis(axis_x, Qt.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)

    def _set_line_chart(self, chart: QChart, title: str, timeline: list[dict[str, object]]) -> None:
        chart.removeAllSeries()
        self._style_chart(chart, title)
        for axis in chart.axes():
            chart.removeAxis(axis)
        if not timeline:
            return
        baseline = float(timeline[0]["ts"])
        series = QLineSeries()
        series.setColor(QColor("#38bdf8"))
        series.setPen(QPen(QColor("#38bdf8"), 2.2))
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
        axis_x.setTitleBrush(QColor("#94a3b8"))
        axis_x.setLabelsColor(QColor("#cbd5e1"))
        axis_x.setGridLineColor(QColor("#243041"))

        axis_y = QValueAxis()
        axis_y.setTitleText("Occupancy")
        axis_y.setTitleBrush(QColor("#94a3b8"))
        axis_y.setLabelsColor(QColor("#cbd5e1"))
        axis_y.setGridLineColor(QColor("#243041"))
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
