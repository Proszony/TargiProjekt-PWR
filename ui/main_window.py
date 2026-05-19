from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QAction, QCloseEvent, QImage
from PySide6.QtWidgets import (
    QApplication,
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
    ProjectConfig,
    ZoneDefinition,
)
from core.multi_camera_runtime import MultiCameraPipelineManager
from core.statistics_service import StatisticsService
from ui.camera_grid_view import CameraGridView
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
            refresh_interval_s=1.0 / max(self.project_config.analytics.live_snapshot_rate_hz, 1.0)
        )
        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.setInterval(int(round(1000.0 / max(self.project_config.analytics.live_snapshot_rate_hz, 1.0))))
        self._telemetry_timer.timeout.connect(self._flush_runtime_presentation)
        self._build_ui()
        self._connect_signals()
        self._refresh_from_project()
        self.update_status("Idle")

    def _build_ui(self) -> None:
        self.setWindowTitle("Fair Monitor | Booth Analytics")
        self.resize(1800, 1000)

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self.settings_button = QPushButton("Settings")
        self.start_button = QPushButton("Start all")
        self.stop_button = QPushButton("Stop all")
        self.stop_button.setEnabled(False)
        self.load_map_button = QPushButton("Load map")
        self.manage_cameras_button = QPushButton("Manage cameras")
        self.calibrate_button = QPushButton("Calibrate cameras")
        self.add_zone_button = QPushButton("Add zone")
        self.delete_zone_button = QPushButton("Delete selected zone")
        self.save_config_button = QPushButton("Save config")
        self.export_button = QPushButton("Export")
        self.statistics_button = QPushButton("Statistics")

        toolbar = QHBoxLayout()
        for button in (
            self.settings_button,
            self.start_button,
            self.stop_button,
            self.load_map_button,
            self.manage_cameras_button,
            self.calibrate_button,
            self.add_zone_button,
            self.delete_zone_button,
            self.save_config_button,
            self.export_button,
            self.statistics_button,
        ):
            toolbar.addWidget(button)
        toolbar.addStretch(1)

        self.camera_grid = CameraGridView()
        self.map_view = MapView()
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.camera_grid)
        splitter.addWidget(self.map_view)
        splitter.setSizes([900, 1100])

        self.zone_stats_label = QLabel("Booth occupancy: no zones configured")
        self.tracks_stats_label = QLabel("Current occupancy: 0 | Visits: 0")
        self.fps_label = QLabel("Aggregate FPS: 0.0")
        info_row = QHBoxLayout()
        info_row.addWidget(self.zone_stats_label, 1)
        info_row.addWidget(self.tracks_stats_label)
        info_row.addWidget(self.fps_label)

        root.addLayout(toolbar)
        root.addWidget(splitter, 1)
        root.addLayout(info_row)

        exit_fullscreen = QAction(self)
        exit_fullscreen.setShortcut(Qt.Key_Escape)
        exit_fullscreen.triggered.connect(self._exit_fullscreen)
        self.addAction(exit_fullscreen)

        toggle_fullscreen = QAction(self)
        toggle_fullscreen.setShortcut(Qt.Key_F11)
        toggle_fullscreen.triggered.connect(self.toggle_fullscreen)
        self.addAction(toggle_fullscreen)

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
        self._runtime_presenter.refresh_interval_s = 1.0 / max(self.project_config.analytics.live_snapshot_rate_hz, 1.0)
        self._telemetry_timer.setInterval(int(round(1000.0 / max(self.project_config.analytics.live_snapshot_rate_hz, 1.0))))
        self._refresh_from_project()
        if self.runtime_manager is not None:
            self.runtime_manager.update_project_config(self.project_config)
        self.statusBar().showMessage("Project settings applied")

    @Slot()
    def open_camera_manager(self) -> None:
        dialog = CameraManagerDialog(self.project_config.cameras, self.detector_models, self)
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
        self.detector_models = available_detection_models(self.project_root)
        self._refresh_from_project()
        if self.runtime_manager is not None:
            self.runtime_manager.update_project_config(self.project_config)
            if old_ids != new_ids:
                self.statusBar().showMessage("Camera topology changed. New or removed cameras apply fully on next start.")
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
        self.runtime_manager.start_all()

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

    @Slot(str, float)
    def update_camera_fps(self, camera_id: str, fps: float) -> None:
        self.camera_grid.update_fps(camera_id, fps)
        self._schedule_runtime_presentation()

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
            )
            if packet is None and camera.camera_id in snapshot.missing_cameras:
                self.camera_grid.update_status(camera.camera_id, "sync waiting")
        self.statistics_window.set_runtime_snapshot(snapshot)
        self._schedule_runtime_presentation()

    @Slot(str, str)
    def show_camera_error(self, camera_id: str, message: str) -> None:
        full_message = f"{camera_id}: {message}"
        if full_message != self._last_error:
            print(f"Worker error: {full_message}", file=sys.stderr)
        self._last_error = full_message
        self.camera_grid.update_status(camera_id, f"Error: {message}")
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
        self._runtime_presenter.refresh_interval_s = 1.0 / max(self.project_config.analytics.live_snapshot_rate_hz, 1.0)
        self._telemetry_timer.setInterval(int(round(1000.0 / max(self.project_config.analytics.live_snapshot_rate_hz, 1.0))))
        self.camera_grid.set_cameras(self.project_config.cameras)
        self.map_view.set_venue_map(self.project_config.venue_map)
        self.map_view.set_world_viewport(self._build_world_viewport())
        self.map_view.set_snapshot(self.last_snapshot)
        self.map_view.set_camera_coverages(self._build_camera_coverages())
        self.map_view.set_camera_overlaps(self._build_camera_overlap_overlays())
        self._schedule_runtime_presentation(force=True)

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
                color=camera.panel_color,
                polygon_world=list(camera.coverage_polygon_world or []),
                raw_polygon_world=list(camera.coverage_polygon_world_raw or []),
                calibration_valid=camera.calibration_valid,
                calibration_warning_text=camera.calibration_warning_text,
            )
            for camera in sorted(self.project_config.cameras, key=lambda item: (item.display_order, item.camera_id))
        ]

    def _build_world_viewport(self):
        return compute_world_viewport(
            self.project_config.cameras,
            self.project_config.venue_map.zones,
            manual_override=self.project_config.venue_map.manual_viewport_override,
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
        occupancy = ", ".join(
            f"{zone.name}: {self.last_snapshot.active_zone_counts.get(zone.zone_id, 0)}"
            for zone in self.project_config.venue_map.zones
        )
        self.zone_stats_label.setText(f"Booth occupancy: {occupancy or 'no zones configured'}")

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
        self.tracks_stats_label.setText(presentation.tracks_stats_text)
        self.fps_label.setText(presentation.fps_text)
        self.statusBar().showMessage(presentation.status_bar_text)

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
