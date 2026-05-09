from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Slot
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
from core.model_catalog import available_detection_models
from core.models import AnalyticsSnapshot, CameraConfig, VenueMapConfig, ZoneDefinition
from core.statistics_service import StatisticsService
from core.streaming import CameraPipelineWorker
from ui.calibration_widget import CalibrationDialog
from ui.canvas import ImageCanvas
from ui.map_view import MapView
from ui.settings_dialog import SettingsDialog
from ui.statistics_window import StatisticsWindow


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        self.config_repo = ConfigRepository(project_root)
        self.statistics_service = StatisticsService(project_root)
        self.venue_map, self.camera_config = self.config_repo.ensure_defaults()
        self.detector_models = available_detection_models(project_root)
        self.statistics_window = StatisticsWindow(self.statistics_service.repository, self)
        self.worker: CameraPipelineWorker | None = None
        self.worker_thread: QThread | None = None
        self.last_error: str | None = None
        self.last_frame: QImage | None = None
        self.last_snapshot = AnalyticsSnapshot(timestamp=0.0)
        self.detection_enabled = True
        self.current_confidence = 0.25
        self.current_inference_size = 640
        self._build_ui()
        self._connect_signals()
        self.map_view.set_venue_map(self.venue_map)
        self._refresh_runtime_summary_labels()
        self.update_status("Idle")

    def _build_ui(self) -> None:
        self.setWindowTitle("Fair Monitor | 2D Venue Map")
        self.resize(1600, 900)

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self.settings_button = QPushButton("Settings")
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.load_map_button = QPushButton("Load map")
        self.calibrate_button = QPushButton("Calibrate camera")
        self.add_zone_button = QPushButton("Add zone")
        self.delete_zone_button = QPushButton("Delete selected zone")
        self.save_config_button = QPushButton("Save config")
        self.statistics_button = QPushButton("Statistics")

        toolbar = QHBoxLayout()
        toolbar.addWidget(self.settings_button)
        toolbar.addWidget(self.start_button)
        toolbar.addWidget(self.stop_button)
        toolbar.addWidget(self.load_map_button)
        toolbar.addWidget(self.calibrate_button)
        toolbar.addWidget(self.add_zone_button)
        toolbar.addWidget(self.delete_zone_button)
        toolbar.addWidget(self.save_config_button)
        toolbar.addWidget(self.statistics_button)
        toolbar.addStretch(1)

        self.camera_canvas = ImageCanvas("Waiting for stream")
        self.camera_canvas.set_add_enabled(False)
        self.camera_canvas.set_drag_enabled(False)
        self.camera_canvas.set_delete_enabled(False)
        self.map_view = MapView()

        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.addWidget(self.camera_canvas)
        content_splitter.addWidget(self.map_view)
        content_splitter.setSizes([950, 650])

        self.zone_stats_label = QLabel("Zone occupancy: no data")
        self.tracks_stats_label = QLabel("Tracks: 0 | Returns: 0")
        self.fps_label = QLabel("FPS: 0.0")
        info_row = QHBoxLayout()
        info_row.addWidget(self.zone_stats_label, 1)
        info_row.addWidget(self.tracks_stats_label)
        info_row.addWidget(self.fps_label)

        root.addLayout(toolbar)
        root.addWidget(content_splitter, 1)
        root.addLayout(info_row)

        exit_fullscreen = QAction(self)
        exit_fullscreen.setShortcut(Qt.Key_Escape)
        exit_fullscreen.triggered.connect(self._exit_fullscreen)
        self.addAction(exit_fullscreen)

        toggle_fullscreen = QAction(self)
        toggle_fullscreen.setShortcut(Qt.Key_F11)
        toggle_fullscreen.triggered.connect(self.toggle_fullscreen)
        self.addAction(toggle_fullscreen)

        self.statusBar().showMessage("Idle")

    def _connect_signals(self) -> None:
        self.settings_button.clicked.connect(self.open_settings_dialog)
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        self.load_map_button.clicked.connect(self.load_map_image)
        self.calibrate_button.clicked.connect(self.open_calibration_dialog)
        self.add_zone_button.clicked.connect(self.add_zone)
        self.delete_zone_button.clicked.connect(self.delete_selected_zone)
        self.save_config_button.clicked.connect(self.save_configuration)
        self.statistics_button.clicked.connect(self.open_statistics_window)
        self.map_view.zone_created.connect(self._handle_zone_created)
        self.map_view.zone_updated.connect(self._handle_zone_updated)
        self.map_view.zone_edit_cancelled.connect(self._handle_zone_edit_cancelled)

    @Slot()
    def open_settings_dialog(self) -> None:
        dialog = SettingsDialog(
            camera_config=self.camera_config,
            venue_map=self.venue_map,
            detector_models=self.detector_models,
            is_running=self.worker is not None,
            parent=self,
        )
        dialog.settings_applied.connect(self._apply_settings_from_dialog)
        dialog.enable_detection_checkbox.setChecked(self.detection_enabled)
        dialog.confidence_spin.setValue(self.current_confidence)
        dialog.inference_size_combo.setCurrentText(str(self.current_inference_size))
        dialog.exec()

    @Slot(object, object)
    def _apply_settings_from_dialog(self, camera_config: object, venue_map: object) -> None:
        if not isinstance(camera_config, CameraConfig) or not isinstance(venue_map, VenueMapConfig):
            return
        sender = self.sender()
        detection_enabled = self.detection_enabled
        confidence = self.current_confidence
        inference_size = self.current_inference_size
        if isinstance(sender, SettingsDialog):
            detection_enabled = sender.enable_detection_checkbox.isChecked()
            confidence = sender.confidence_spin.value()
            inference_size = int(sender.inference_size_combo.currentText())

        self.camera_config = CameraConfig.from_dict(camera_config.to_dict())
        self.venue_map = VenueMapConfig.from_dict(venue_map.to_dict())
        self.detection_enabled = detection_enabled
        self.current_confidence = confidence
        self.current_inference_size = inference_size
        self.detector_models = available_detection_models(self.project_root)
        self.map_view.set_venue_map(self.venue_map)
        self.map_view.set_snapshot(self.last_snapshot)
        self._refresh_runtime_summary_labels()

        if self.worker is not None:
            self.worker.set_detection_enabled(self.detection_enabled)
            self.worker.set_confidence(self.current_confidence)
            self.worker.set_detector_model_path(self.camera_config.detector_model_path)
            self.worker.set_detector_augmentation(self.camera_config.detector_use_augmentation)
            self.worker.set_inference_size(self.current_inference_size)
            self._push_runtime_config()

        self.statusBar().showMessage("Settings applied")

    @Slot()
    def start_stream(self) -> None:
        if self.worker is not None or self.worker_thread is not None:
            return

        self.worker_thread = QThread(self)
        self.worker = CameraPipelineWorker(
            camera_config=self.camera_config,
            venue_map=self.venue_map,
            statistics_service=self.statistics_service,
            confidence=self.current_confidence,
            inference_size=self.current_inference_size,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.update_camera_frame)
        self.worker.analytics_ready.connect(self.update_analytics)
        self.worker.status_changed.connect(self.update_status)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.fps_update.connect(self.update_fps)
        self.worker.stopped_listening.connect(self.worker_thread.quit)
        self.worker.stopped_listening.connect(self._mark_stopped_from_worker)
        self.worker_thread.finished.connect(self._cleanup_worker_thread)
        self.worker.set_detection_enabled(self.detection_enabled)
        self._set_running(True)
        self.worker_thread.start()

    @Slot()
    def stop_stream(self) -> None:
        if self.worker is not None:
            self.update_status("Stopping...")
            self.stop_button.setEnabled(False)
            self.worker.stop()
            return
        self._set_running(False)
        self.update_status("Stopped")

    @Slot(object)
    def update_camera_frame(self, image: object) -> None:
        if not isinstance(image, QImage):
            return
        self.last_frame = image
        self.camera_canvas.set_image(image)

    @Slot(object)
    def update_analytics(self, snapshot: object) -> None:
        if not isinstance(snapshot, AnalyticsSnapshot):
            return
        self.last_snapshot = snapshot
        self.map_view.set_snapshot(snapshot)
        self.statistics_window.set_live_snapshot(snapshot, self.venue_map)
        self.statistics_window.set_current_session_id(self.statistics_service.current_session_id)
        self._refresh_runtime_summary_labels()

    @Slot(float)
    def update_fps(self, fps: float) -> None:
        self.fps_label.setText(f"FPS: {fps:.1f}")

    @Slot(str)
    def update_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    @Slot(str)
    def show_error(self, message: str) -> None:
        if message != self.last_error:
            print(f"Worker error: {message}", file=sys.stderr)
        self.last_error = message
        self.statusBar().showMessage(f"Error: {message}")

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
        self.venue_map.map_image_path = path
        self.map_view.set_venue_map(self.venue_map)
        self._push_runtime_config()

    @Slot()
    def save_configuration(self) -> None:
        self.config_repo.save_venue(self.venue_map)
        self.config_repo.save_camera(self.camera_config)
        self.statusBar().showMessage("Configuration saved")
        self._push_runtime_config()

    @Slot()
    def open_calibration_dialog(self) -> None:
        if self.last_frame is None:
            QMessageBox.warning(self, "No frame", "Start the stream and capture at least one frame first.")
            return
        dialog = CalibrationDialog(
            frame=self.last_frame,
            venue_map=self.venue_map,
            existing_pairs=self.camera_config.calibration_pairs,
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted or dialog.homography is None:
            return
        self.camera_config.homography_image_to_world = dialog.homography
        self.camera_config.calibration_pairs = dialog.calibration_pairs
        self._push_runtime_config()
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
        self.statistics_window.set_live_snapshot(self.last_snapshot, self.venue_map)
        self.statistics_window.set_current_session_id(self.statistics_service.current_session_id)
        self.statistics_window.show()
        self.statistics_window.raise_()
        self.statistics_window.activateWindow()

    @Slot()
    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    @Slot()
    def _cleanup_worker_thread(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
            self.worker_thread = None
        self._set_running(False)

    @Slot()
    def _mark_stopped_from_worker(self) -> None:
        self._set_running(False)
        self.fps_label.setText("FPS: 0.0")

    @Slot(object)
    def _handle_zone_created(self, zone: object) -> None:
        if not isinstance(zone, ZoneDefinition):
            return
        zones = [item for item in self.venue_map.zones if item.zone_id != zone.zone_id]
        zones.append(zone)
        self._replace_zone_list(zones)
        self.statusBar().showMessage(f"Added zone {zone.name}")

    @Slot(object)
    def _handle_zone_updated(self, zone: object) -> None:
        if not isinstance(zone, ZoneDefinition):
            return
        zones: list[ZoneDefinition] = []
        replaced = False
        for existing in self.venue_map.zones:
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
        self.venue_map.zones = zones
        self.map_view.set_venue_map(self.venue_map)
        self.map_view.set_snapshot(self.last_snapshot)
        self._push_runtime_config()
        self._refresh_runtime_summary_labels()

    def _push_runtime_config(self) -> None:
        if self.worker is not None:
            self.worker.update_configs(self.camera_config, self.venue_map)

    def _set_running(self, running: bool) -> None:
        self.settings_button.setEnabled(True)
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _refresh_runtime_summary_labels(self) -> None:
        occupancy = ", ".join(
            f"{zone.name}: {self.last_snapshot.active_zone_counts.get(zone.zone_id, 0)}"
            for zone in self.venue_map.zones
        )
        total_returns = sum(self.last_snapshot.return_counts.values())
        self.zone_stats_label.setText(f"Zone occupancy: {occupancy or 'no zones configured'}")
        self.tracks_stats_label.setText(
            f"Tracks: {len(self.last_snapshot.active_global_tracks)} | Returns: {total_returns}"
        )

    @Slot()
    def _exit_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self.stop_stream()
        event.accept()


def run_application(project_root: Path) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(project_root=project_root)
    window.showMaximized()
    return app.exec()
