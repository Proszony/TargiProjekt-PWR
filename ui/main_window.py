from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QAction, QCloseEvent, QImage
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core import runtime_defaults as rd
from core.config import ConfigRepository
from core.calibration import compute_world_viewport
from core.camera_overlap import build_camera_overlap_graph
from core.model_catalog import available_detection_models
from core.models import (
    AnalyticsSnapshot,
    CameraCoverageOverlay,
    CameraOverlapOverlay,
    CameraConfig,
    MultiCameraRuntimeSnapshot,
    Point,
    ProjectConfig,
    WorldViewport,
    ZoneDefinition,
)
from core.multi_camera_runtime import MultiCameraPipelineManager
from core.statistics_service import StatisticsService
from ui.camera_grid_view import CameraGridView
from ui.camera_colors import camera_color
from ui.camera_manager_dialog import CameraManagerDialog
from ui.map_view import MapView
from ui.multi_camera_calibration_dialog import MultiCameraCalibrationDialog
from ui.runtime_presenter import RuntimePresenter
from ui.settings_dialog import SettingsDialog
from ui.statistics_window import StatisticsWindow


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        self.config_repo = ConfigRepository(project_root)
        self.statistics_service = StatisticsService(project_root)
        self.project_config = self.config_repo.ensure_defaults()
        self.detector_models = available_detection_models(project_root)
        self.statistics_window = StatisticsWindow(self.statistics_service.repository, self)
        self.runtime_manager: MultiCameraPipelineManager | None = None
        self.camera_frames: dict[str, QImage] = {}
        self.last_snapshot = AnalyticsSnapshot(timestamp=0.0)
        self.last_runtime_snapshot = MultiCameraRuntimeSnapshot(timestamp=0.0)
        self._last_error: str | None = None
        self._runtime_presenter = RuntimePresenter(
            refresh_interval_s=1.0 / max(rd.DEFAULT_UI_LIVE_SNAPSHOT_RATE_HZ, 1.0)
        )
        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.setInterval(int(round(1000.0 / max(rd.DEFAULT_UI_LIVE_SNAPSHOT_RATE_HZ, 1.0))))
        self._telemetry_timer.timeout.connect(self._flush_runtime_presentation)
        self._build_ui()
        self._connect_signals()
        self._refresh_from_project()
        self.statusBar().hide()
        self.update_status("Idle")

    def _build_ui(self) -> None:
        self.setWindowTitle("Fair Monitor | Booth Analytics")
        self.resize(1800, 1000)
        self._apply_window_style()

        central = QWidget(self)
        central.setObjectName("appRoot")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 8)
        root.setSpacing(8)

        self.settings_button = QPushButton("Settings")
        self.start_button = QPushButton("Start all")
        self.stop_button = QPushButton("Stop all")
        self.stop_button.setEnabled(False)
        self.load_map_button = QPushButton("Load map")
        self.manage_cameras_button = QPushButton("Cameras")
        self.calibrate_button = QPushButton("Calibrate")
        self.add_zone_button = QPushButton("Add zone")
        self.delete_zone_button = QPushButton("Delete zone")
        self.save_config_button = QPushButton("Save config")
        self.export_button = QPushButton("Export")
        self.statistics_button = QPushButton("Analytics")
        self.map_coverage_button = self._build_toggle_button("Coverage", checked=True)
        self.map_people_button = self._build_toggle_button("People", checked=True)
        self.map_overlap_button = self._build_toggle_button("Overlap", checked=False)

        self.start_button.setProperty("kind", "primary")
        self.stop_button.setProperty("kind", "danger")

        command_bar = QFrame()
        command_bar.setObjectName("commandBar")
        command_layout = QHBoxLayout(command_bar)
        command_layout.setContentsMargins(12, 8, 12, 8)
        command_layout.setSpacing(12)

        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(6)
        (
            self.booths_metric_value,
            self.booths_metric_detail,
        ) = self._build_metric_chip(metrics_row, "Booths")
        (
            self.occupancy_metric_value,
            self.occupancy_metric_detail,
        ) = self._build_metric_chip(metrics_row, "Now")
        (
            self.visits_metric_value,
            self.visits_metric_detail,
        ) = self._build_metric_chip(metrics_row, "Visits")
        (
            self.time_metric_value,
            self.time_metric_detail,
        ) = self._build_metric_chip(metrics_row, "Time")
        (
            self.fps_metric_value,
            self.fps_metric_detail,
        ) = self._build_metric_chip(metrics_row, "FPS")

        command_layout.addLayout(metrics_row)
        command_layout.addStretch(1)
        for button in (
            self.start_button,
            self.stop_button,
            self.statistics_button,
            self.manage_cameras_button,
            self.calibrate_button,
            self.load_map_button,
            self.add_zone_button,
            self.delete_zone_button,
            self.save_config_button,
            self.export_button,
            self.settings_button,
        ):
            command_layout.addWidget(button)

        self.camera_grid = CameraGridView()
        self.map_view = MapView()
        self.map_view.set_show_coverages(self.map_coverage_button.isChecked())
        self.map_view.set_show_overlaps(self.map_overlap_button.isChecked())
        self.map_view.set_show_tracks(self.map_people_button.isChecked())

        camera_workspace = QFrame()
        camera_workspace.setObjectName("workspacePanel")
        camera_layout = QVBoxLayout(camera_workspace)
        camera_layout.setContentsMargins(8, 8, 8, 8)
        camera_layout.setSpacing(6)
        self.camera_focus_label = QLabel("No camera selected")
        self.camera_focus_label.setObjectName("panelMeta")
        camera_layout.addWidget(self.camera_focus_label, 0, Qt.AlignRight)
        camera_layout.addWidget(self.camera_grid, 1)

        map_workspace = QFrame()
        map_workspace.setObjectName("workspacePanel")
        map_layout = QVBoxLayout(map_workspace)
        map_layout.setContentsMargins(8, 8, 8, 8)
        map_layout.setSpacing(6)
        self.map_focus_label = QLabel("Focus follows the selected camera")
        self.map_focus_label.setObjectName("panelMeta")

        map_controls = QHBoxLayout()
        map_controls.setSpacing(6)
        map_controls.addWidget(self.map_coverage_button)
        map_controls.addWidget(self.map_people_button)
        map_controls.addWidget(self.map_overlap_button)
        map_controls.addStretch(1)
        map_controls.addWidget(self.map_focus_label)
        map_layout.addLayout(map_controls)
        map_layout.addWidget(self.map_view, 1)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(camera_workspace)
        self.splitter.addWidget(map_workspace)
        self.splitter.setSizes([1060, 840])

        root.addWidget(command_bar)
        root.addWidget(self.splitter, 1)

        exit_fullscreen = QAction(self)
        exit_fullscreen.setShortcut(Qt.Key_Escape)
        exit_fullscreen.triggered.connect(self._exit_fullscreen)
        self.addAction(exit_fullscreen)

        toggle_fullscreen = QAction(self)
        toggle_fullscreen.setShortcut(Qt.Key_F11)
        toggle_fullscreen.triggered.connect(self.toggle_fullscreen)
        self.addAction(toggle_fullscreen)

    def _apply_window_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget#appRoot {
                background: #071018;
                color: #e7eef5;
            }
            QFrame#commandBar, QFrame#workspacePanel, QFrame#metricChip,
            QFrame#toolbarGroup, QFrame#cameraRail, QFrame#cameraStage {
                background: #0d1721;
                border: 1px solid #1f3141;
                border-radius: 10px;
            }
            QFrame#toolbarGroup {
                background: #0a131b;
                border-radius: 14px;
                padding: 0;
            }
            QLabel#toolbarGroupTitle {
                color: #9fb1c0;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            QPushButton {
                background: #132230;
                border: 1px solid #23394b;
                border-radius: 8px;
                color: #eef3f8;
                padding: 7px 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #182b3b;
                border-color: #2d5369;
            }
            QPushButton:pressed {
                background: #0f1d29;
            }
            QPushButton[kind="primary"] {
                background: #7de3e1;
                color: #06222b;
                border-color: #7de3e1;
            }
            QPushButton[kind="primary"]:hover {
                background: #98edeb;
                border-color: #98edeb;
            }
            QPushButton[kind="danger"] {
                background: #4d1820;
                color: #ffdbe0;
                border-color: #7a2b3a;
            }
            QPushButton[kind="danger"]:hover {
                background: #61202a;
            }
            QPushButton:disabled {
                color: #6d8091;
                background: #101923;
                border-color: #192734;
            }
            QPushButton#mapToggle {
                min-width: 72px;
                padding: 7px 10px;
            }
            QPushButton#mapToggle:checked {
                background: #1a3040;
                border-color: #75d3e0;
                color: #dff8fb;
            }
            QFrame#metricChip {
                background: #0a131b;
                border-radius: 8px;
                min-width: 112px;
            }
            QLabel#metricTitle {
                color: #89a0b2;
                font-size: 10px;
                font-weight: 700;
                text-transform: uppercase;
            }
            QLabel#metricValue {
                color: #f2f7fb;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#metricDetail {
                color: #9fb1c0;
                font-size: 10px;
            }
            QLabel#panelTitle {
                color: #f2f7fb;
                font-size: 18px;
                font-weight: 700;
            }
            QLabel#panelSubtitle, QLabel#panelMeta {
                color: #9fb1c0;
                font-size: 12px;
            }
            QLabel#cameraRailTitle, QLabel#cameraStageTitle {
                color: #f2f7fb;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#cameraRailHint, QLabel#cameraStageSubtitle, QLabel#cameraStageMeta {
                color: #90a5b6;
                font-size: 12px;
            }
            QLabel#cameraStageImage {
                background: #071018;
                border: 1px solid #1f3141;
                border-radius: 8px;
                color: #9fb1c0;
            }
            QLabel#cameraStageBadge {
                background: #112130;
                border-radius: 12px;
                color: #dce7f0;
                font-size: 12px;
                font-weight: 700;
                padding: 6px 12px;
            }
            QPushButton#cameraSelector {
                text-align: left;
                padding: 8px 10px;
                font-size: 12px;
                line-height: 1.35;
            }
            QPushButton#cameraSelector:checked {
                background: #172b3a;
                border-color: #75d3e0;
                color: #f2fbfd;
            }
            QScrollArea {
                border: 0;
                background: transparent;
            }
            QSplitter::handle {
                background: #071018;
                width: 10px;
            }
            """
        )

    def _build_action_group(self, title: str, *buttons: QPushButton) -> QFrame:
        frame = QFrame()
        frame.setObjectName("toolbarGroup")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)
        label = QLabel(title)
        label.setObjectName("toolbarGroupTitle")
        row = QHBoxLayout()
        row.setSpacing(8)
        for button in buttons:
            row.addWidget(button)
        row.addStretch(1)
        layout.addWidget(label)
        layout.addLayout(row)
        return frame

    def _build_metric_chip(self, parent_layout: QHBoxLayout, title: str) -> tuple[QLabel, QLabel]:
        frame = QFrame()
        frame.setObjectName("metricChip")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(1)
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        value_label = QLabel("0")
        value_label.setObjectName("metricValue")
        detail_label = QLabel("")
        detail_label.setObjectName("metricDetail")
        detail_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(detail_label)
        parent_layout.addWidget(frame, 1)
        return value_label, detail_label

    def _build_panel_heading(self, title: str, subtitle: str) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(12)
        text_stack = QVBoxLayout()
        text_stack.setSpacing(4)
        title_label = QLabel(title)
        title_label.setObjectName("panelTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("panelSubtitle")
        subtitle_label.setWordWrap(True)
        text_stack.addWidget(title_label)
        text_stack.addWidget(subtitle_label)
        layout.addLayout(text_stack, 1)
        return layout

    def _build_toggle_button(self, text: str, *, checked: bool) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("mapToggle")
        button.setCheckable(True)
        button.setChecked(checked)
        return button

    def _connect_signals(self) -> None:
        self.settings_button.clicked.connect(self.open_settings_dialog)
        self.start_button.clicked.connect(self.start_streams)
        self.stop_button.clicked.connect(self.stop_streams)
        self.load_map_button.clicked.connect(self.load_map_image)
        self.manage_cameras_button.clicked.connect(self.open_camera_manager)
        self.calibrate_button.clicked.connect(self.open_calibration_dialog)
        self.add_zone_button.clicked.connect(self.add_zone)
        self.delete_zone_button.clicked.connect(self.delete_selected_zone)
        self.save_config_button.clicked.connect(self.save_configuration)
        self.export_button.clicked.connect(self.export_statistics)
        self.statistics_button.clicked.connect(self.open_statistics_window)
        self.camera_grid.camera_selected.connect(self._handle_camera_selected)
        self.map_coverage_button.toggled.connect(self.map_view.set_show_coverages)
        self.map_overlap_button.toggled.connect(self.map_view.set_show_overlaps)
        self.map_people_button.toggled.connect(self.map_view.set_show_tracks)
        self.map_view.zone_created.connect(self._handle_zone_created)
        self.map_view.zone_updated.connect(self._handle_zone_updated)
        self.map_view.zone_edit_cancelled.connect(self._handle_zone_edit_cancelled)

    def _connect_runtime_manager(self, manager: MultiCameraPipelineManager) -> None:
        manager.camera_frame_ready.connect(self.update_camera_frame)
        manager.camera_status_changed.connect(self.update_camera_status)
        manager.camera_error.connect(self.show_camera_error)
        manager.camera_fps_changed.connect(self.update_camera_fps)
        manager.analytics_ready.connect(self.update_analytics)
        manager.runtime_snapshot_ready.connect(self.update_runtime_snapshot)
        manager.started.connect(lambda: self._set_running(True))
        manager.stopped.connect(lambda: self._set_running(False))

    @Slot()
    def open_settings_dialog(self) -> None:
        dialog = SettingsDialog(self.project_config, self.runtime_manager is not None, self)
        dialog.settings_applied.connect(self._apply_project_settings)
        dialog.exec()

    @Slot(object)
    def _apply_project_settings(self, project_config: object) -> None:
        if not isinstance(project_config, ProjectConfig):
            return
        self.project_config = ProjectConfig.from_dict(project_config.to_dict())
        self._refresh_from_project()
        if self.runtime_manager is not None:
            self.runtime_manager.update_project_config(self.project_config)
        self.statusBar().showMessage("Project settings applied")

    @Slot()
    def open_camera_manager(self) -> None:
        self.detector_models = available_detection_models(self.project_root)
        dialog = CameraManagerDialog(self.project_config.cameras, self.project_root, self)
        dialog.cameras_applied.connect(self._apply_camera_list)
        dialog.exec()

    @Slot(object)
    def _apply_camera_list(self, cameras: object) -> None:
        if not isinstance(cameras, list):
            return
        normalized: list[CameraConfig] = []
        for display_order, camera in enumerate(cameras):
            if not isinstance(camera, CameraConfig):
                continue
            camera_copy = CameraConfig.from_dict(camera.to_dict())
            camera_copy.display_order = display_order
            normalized.append(camera_copy)
        old_ids = {camera.camera_id for camera in self.project_config.cameras}
        new_ids = {camera.camera_id for camera in normalized}
        self.project_config.cameras = normalized
        self.camera_frames = {
            camera_id: image
            for camera_id, image in self.camera_frames.items()
            if camera_id in new_ids
        }
        self._refresh_from_project()
        if self.runtime_manager is not None:
            self.runtime_manager.update_project_config(self.project_config)
            if old_ids != new_ids:
                self.statusBar().showMessage("Camera topology updated.")
            else:
                self.statusBar().showMessage("Camera settings applied")

    @Slot()
    def start_streams(self) -> None:
        if self.runtime_manager is not None:
            return
        if not any(camera.enabled for camera in self.project_config.cameras):
            QMessageBox.warning(self, "No cameras", "Enable at least one camera before starting.")
            return
        self.runtime_manager = MultiCameraPipelineManager(
            self.project_config,
            self.statistics_service,
            self.project_root,
        )
        self._connect_runtime_manager(self.runtime_manager)
        self._set_running(True)
        try:
            self.runtime_manager.start_all()
        except Exception as exc:
            self.runtime_manager = None
            self._set_running(False)
            QMessageBox.critical(self, "Runtime start failed", str(exc))

    @Slot()
    def stop_streams(self) -> None:
        if self.runtime_manager is None:
            self._set_running(False)
            self.update_status("Stopped")
            return
        self.update_status("Stopping...")
        self.stop_button.setEnabled(False)
        self.runtime_manager.stop_all()

    @Slot(str, object)
    def update_camera_frame(self, camera_id: str, image: object) -> None:
        if not isinstance(image, QImage):
            return
        self.camera_frames[camera_id] = image
        self.camera_grid.update_frame(camera_id, image)

    @Slot(str, str)
    def update_camera_status(self, camera_id: str, text: str) -> None:
        self.camera_grid.update_status(camera_id, text)
        if self.camera_grid.selected_camera_id() == camera_id:
            self._sync_selected_camera_context()

    @Slot(str, float)
    def update_camera_fps(self, camera_id: str, fps: float) -> None:
        self.camera_grid.update_fps(camera_id, fps)
        if self.camera_grid.selected_camera_id() == camera_id:
            self._sync_selected_camera_context()
        self._schedule_runtime_presentation()

    @Slot(str)
    def _handle_camera_selected(self, camera_id: str) -> None:
        self.map_view.set_focused_camera(camera_id)
        self._sync_selected_camera_context()

    @Slot(object)
    def update_analytics(self, snapshot: object) -> None:
        if not isinstance(snapshot, AnalyticsSnapshot):
            return
        self.last_snapshot = snapshot
        self.map_view.set_snapshot(snapshot)
        self.statistics_window.set_live_snapshot(snapshot, self.project_config.venue_map)
        self.statistics_window.set_current_session_id(self.statistics_service.current_session_id)
        self._schedule_runtime_presentation()

    @Slot(object)
    def update_runtime_snapshot(self, snapshot: object) -> None:
        if not isinstance(snapshot, MultiCameraRuntimeSnapshot):
            return
        self.last_runtime_snapshot = snapshot
        self.map_view.set_world_viewport(self._build_world_viewport())
        self.map_view.set_camera_coverages(
            self._build_camera_coverages()
        )
        self.map_view.set_camera_overlaps(self._build_camera_overlap_overlays())
        for camera in self.project_config.cameras:
            packet = snapshot.camera_packets.get(camera.camera_id)
            self.camera_grid.update_sync_state(
                camera.camera_id,
                media_time_s=packet.media_time_s if packet is not None else None,
                sync_drift_s=snapshot.sync_drift_by_camera_s.get(camera.camera_id, 0.0),
                dropped_frames=snapshot.dropped_frames_by_camera.get(camera.camera_id, 0),
                missing=camera.camera_id in snapshot.missing_cameras,
                source_fps=packet.source_fps if packet is not None else None,
                processing_latency_s=packet.processing_latency_s if packet is not None else None,
            )
        self._sync_selected_camera_context()
        self.statistics_window.set_runtime_snapshot(snapshot)
        self._schedule_runtime_presentation()

    @Slot(str, str)
    def show_camera_error(self, camera_id: str, message: str) -> None:
        full_message = f"{camera_id}: {message}"
        if full_message != self._last_error:
            print(f"Worker error: {full_message}", file=sys.stderr)
        self._last_error = full_message
        self.camera_grid.update_status(camera_id, f"Error: {message}")
        if self.camera_grid.selected_camera_id() == camera_id:
            self._sync_selected_camera_context()
        self.statusBar().showMessage(f"Error: {full_message}")

    @Slot(str)
    def update_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    @Slot()
    def load_map_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select venue map",
            str(self.project_root),
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not path:
            return
        self.project_config.venue_map.map_image_path = path
        self._refresh_from_project()
        if self.runtime_manager is not None:
            self.runtime_manager.update_project_config(self.project_config)

    @Slot()
    def save_configuration(self) -> None:
        self.config_repo.save_project(self.project_config)
        self.statusBar().showMessage("Configuration saved")

    @Slot()
    def open_calibration_dialog(self) -> None:
        dialog = MultiCameraCalibrationDialog(self.project_config, self.camera_frames, self)
        dialog.calibration_applied.connect(self._apply_calibration_project)
        dialog.exec()

    @Slot(object)
    def _apply_calibration_project(self, project_config: object) -> None:
        if not isinstance(project_config, ProjectConfig):
            return
        self.project_config = ProjectConfig.from_dict(project_config.to_dict())
        self._refresh_from_project()
        if self.runtime_manager is not None:
            self.runtime_manager.update_project_config(self.project_config)
        self.statusBar().showMessage("Calibration updated")

    @Slot()
    def add_zone(self) -> None:
        name, ok = QInputDialog.getText(self, "Zone name", "Enter zone name")
        if not ok or not name.strip():
            return
        kind, ok = QInputDialog.getItem(
            self,
            "Zone kind",
            "Choose zone kind",
            ["booth", "aisle", "entry", "exit", "neutral"],
            editable=False,
        )
        if not ok:
            return
        self.map_view.begin_zone_drawing(name.strip(), kind)
        self.statusBar().showMessage(
            "Zone drawing: left click add, drag to adjust, right click delete point, double click finish, Esc cancel."
        )

    @Slot()
    def delete_selected_zone(self) -> None:
        removed = self.map_view.remove_selected_zone()
        if removed is None:
            return
        self._replace_zone_list(self.map_view.venue_map.zones)
        self.statusBar().showMessage(f"Removed zone {removed.name}")

    @Slot()
    def open_statistics_window(self) -> None:
        self.statistics_window.reload_history()
        self.statistics_window.set_live_snapshot(self.last_snapshot, self.project_config.venue_map)
        self.statistics_window.set_runtime_snapshot(self.last_runtime_snapshot)
        self.statistics_window.set_current_session_id(self.statistics_service.current_session_id)
        self.statistics_window.show()
        self.statistics_window.raise_()
        self.statistics_window.activateWindow()

    @Slot()
    def export_statistics(self) -> None:
        self.statistics_window.reload_history()
        self.statistics_window.set_live_snapshot(self.last_snapshot, self.project_config.venue_map)
        self.statistics_window.set_runtime_snapshot(self.last_runtime_snapshot)
        self.statistics_window.set_current_session_id(self.statistics_service.current_session_id)
        self.statistics_window.export_preferred()

    @Slot()
    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    @Slot(object)
    def _handle_zone_created(self, zone: object) -> None:
        if not isinstance(zone, ZoneDefinition):
            return
        zones = [item for item in self.project_config.venue_map.zones if item.zone_id != zone.zone_id]
        zones.append(zone)
        self._replace_zone_list(zones)
        self.statusBar().showMessage(f"Added zone {zone.name}")

    @Slot(object)
    def _handle_zone_updated(self, zone: object) -> None:
        if not isinstance(zone, ZoneDefinition):
            return
        zones: list[ZoneDefinition] = []
        replaced = False
        for existing in self.project_config.venue_map.zones:
            if existing.zone_id == zone.zone_id:
                zones.append(zone)
                replaced = True
            else:
                zones.append(existing)
        if not replaced:
            zones.append(zone)
        self._replace_zone_list(zones)
        self.statusBar().showMessage(f"Updated zone {zone.name}")

    @Slot()
    def _handle_zone_edit_cancelled(self) -> None:
        self.statusBar().showMessage("Editing cancelled.")

    def _replace_zone_list(self, zones: list[ZoneDefinition]) -> None:
        self.project_config.venue_map.zones = zones
        self.map_view.set_venue_map(self.project_config.venue_map)
        self.map_view.set_snapshot(self.last_snapshot)
        if self.runtime_manager is not None:
            self.runtime_manager.update_project_config(self.project_config)
        self._schedule_runtime_presentation(force=True)

    def _refresh_from_project(self) -> None:
        self._runtime_presenter.refresh_interval_s = 1.0 / max(rd.DEFAULT_UI_LIVE_SNAPSHOT_RATE_HZ, 1.0)
        self._telemetry_timer.setInterval(int(round(1000.0 / max(rd.DEFAULT_UI_LIVE_SNAPSHOT_RATE_HZ, 1.0))))
        self.camera_grid.set_cameras(self.project_config.cameras)
        self.map_view.set_venue_map(self.project_config.venue_map)
        self.map_view.set_world_viewport(self._build_world_viewport())
        self.map_view.set_snapshot(self.last_snapshot)
        self.map_view.set_camera_coverages(self._build_camera_coverages())
        self.map_view.set_camera_overlaps(self._build_camera_overlap_overlays())
        self._sync_selected_camera_context()
        self._schedule_runtime_presentation(force=True)

    def _sync_selected_camera_context(self) -> None:
        state = self.camera_grid.selected_camera_state()
        if state is None:
            self.camera_focus_label.setText("No camera selected")
            self.map_focus_label.setText("Focus follows the selected camera")
            self.fps_metric_value.setText("0.0")
            self.fps_metric_detail.setText("no camera")
            self.map_view.set_focused_camera(None)
            return
        self.camera_focus_label.setText(
            f"Focused feed: {state.name} | {state.status_text} | {state.fps:.1f} FPS"
        )
        self.map_focus_label.setText(f"Coverage focus: {state.name}")
        self.fps_metric_value.setText(f"{state.fps:.1f}")
        self.fps_metric_detail.setText(state.name)
        self.map_view.set_focused_camera(state.camera_id)

    def _build_camera_overlap_overlays(self) -> list[CameraOverlapOverlay]:
        overlap_graph = build_camera_overlap_graph(self.project_config.cameras, self.project_config.overlap_dedup)
        overlays: list[CameraOverlapOverlay] = []
        seen_pairs: set[tuple[str, str]] = set()
        for relation in overlap_graph.relations.values():
            if not relation.is_adjacent or len(relation.intersection_polygon_world) < 3 or relation.overlap_area_m2 <= 0.0:
                continue
            pair = tuple(sorted((relation.camera_a_id, relation.camera_b_id)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            overlays.append(
                CameraOverlapOverlay(
                    camera_a_id=pair[0],
                    camera_b_id=pair[1],
                    polygon_world=list(relation.intersection_polygon_world),
                    overlap_area_m2=relation.overlap_area_m2,
                    label=f"{pair[0]} <-> {pair[1]}",
                )
            )
        return overlays

    def _build_camera_coverages(self) -> list[CameraCoverageOverlay]:
        return [
            CameraCoverageOverlay(
                camera_id=camera.camera_id,
                camera_name=camera.name,
                color=camera_color(camera.display_order, camera.camera_id),
                polygon_world=list(camera.coverage_polygon_world or []),
                raw_polygon_world=list(camera.coverage_polygon_world_raw or []),
                calibration_valid=camera.calibration_valid,
                calibration_warning_text=camera.calibration_warning_text,
            )
            for camera in sorted(self.project_config.cameras, key=lambda item: (item.display_order, item.camera_id))
        ]

    def _build_world_viewport(self):
        zone_polygons = [
            zone.polygon_world
            for zone in self.project_config.venue_map.zones
            if len(zone.polygon_world) >= 3
        ]
        if zone_polygons or self.project_config.venue_map.manual_viewport_override is not None:
            return compute_world_viewport(
                [],
                self.project_config.venue_map.zones,
                padding_ratio=0.12,
                manual_override=self.project_config.venue_map.manual_viewport_override,
            )
        anchor_points = [anchor.world_point for anchor in self.project_config.shared_anchors]
        if anchor_points:
            return self._viewport_from_points(anchor_points)
        return compute_world_viewport(
            self.project_config.cameras,
            self.project_config.venue_map.zones,
            manual_override=self.project_config.venue_map.manual_viewport_override,
        )

    @staticmethod
    def _viewport_from_points(points: list[Point]) -> WorldViewport:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        pad_x = span_x * 0.12
        pad_y = span_y * 0.12
        return WorldViewport(
            min_x=min_x - pad_x,
            min_y=min_y - pad_y,
            max_x=max_x + pad_x,
            max_y=max_y + pad_y,
        )

    def _calibration_summary_suffix(self) -> str:
        flagged = [
            camera.name
            for camera in self.project_config.cameras
            if camera.enabled and (not camera.calibration_valid or camera.calibration_warning_text)
        ]
        if not flagged:
            return ""
        preview = ", ".join(flagged[:2])
        if len(flagged) > 2:
            preview += f" +{len(flagged) - 2}"
        return f" | Calibration: {preview}"

    def _refresh_runtime_summary_labels(self) -> None:
        zone_count = len(self.project_config.venue_map.zones)
        active_booths = sum(1 for count in self.last_snapshot.active_zone_counts.values() if count > 0)
        mean_avg_time = 0.0
        non_zero_avg = [value for value in self.last_snapshot.avg_dwell_times.values() if value > 0.0]
        if non_zero_avg:
            mean_avg_time = sum(non_zero_avg) / len(non_zero_avg)
        total_drops = sum(self.last_runtime_snapshot.dropped_frames_by_camera.values())
        enabled_cameras = sum(1 for camera in self.project_config.cameras if camera.enabled)
        selected_state = self.camera_grid.selected_camera_state()

        self.booths_metric_value.setText(f"{active_booths}/{zone_count or 0}")
        self.booths_metric_detail.setText("active")
        self.occupancy_metric_value.setText(str(self.last_snapshot.total_current_occupancy))
        self.occupancy_metric_detail.setText("people")
        self.visits_metric_value.setText(str(self.last_snapshot.total_entries))
        self.visits_metric_detail.setText("entries")
        self.time_metric_value.setText(f"{mean_avg_time:.1f}s")
        self.time_metric_detail.setText("average")
        if selected_state is None:
            self.fps_metric_value.setText("0.0")
            self.fps_metric_detail.setText(f"{enabled_cameras} cams | {total_drops} drops")
        else:
            self.fps_metric_value.setText(f"{selected_state.fps:.1f}")
            self.fps_metric_detail.setText(selected_state.name)

    def _schedule_runtime_presentation(self, *, force: bool = False) -> None:
        presentation = self._runtime_presenter.submit(
            self.last_snapshot,
            self.last_runtime_snapshot,
            calibration_suffix=self._calibration_summary_suffix(),
            now_s=time.monotonic(),
        )
        if presentation is not None:
            self._apply_runtime_presentation(presentation)
            return
        if force:
            forced = self._runtime_presenter.flush(now_s=time.monotonic())
            if forced is not None:
                self._apply_runtime_presentation(forced)
                return
        if not self._telemetry_timer.isActive():
            self._telemetry_timer.start()

    @Slot()
    def _flush_runtime_presentation(self) -> None:
        presentation = self._runtime_presenter.flush(now_s=time.monotonic())
        if presentation is None:
            self._telemetry_timer.stop()
            return
        self._apply_runtime_presentation(presentation)

    def _apply_runtime_presentation(self, presentation) -> None:
        self._refresh_runtime_summary_labels()

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.settings_button.setEnabled(True)
        if not running:
            self.runtime_manager = None

    @Slot()
    def _exit_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self.stop_streams()
        event.accept()


def run_application(project_root: Path) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(project_root=project_root)
    window.showMaximized()
    return app.exec()
