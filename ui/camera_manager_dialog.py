from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.camera_overlap import build_camera_overlap_graph
from core.model_catalog import infer_detector_family, infer_detector_variant
from core.models import CameraConfig, OverlapDedupConfig
from ui.sample_catalog_dialog import SampleCatalogDialog


class CameraEditorDialog(QDialog):
    def __init__(
        self,
        camera_config: CameraConfig,
        detector_models: list[tuple[str, str]],
        all_camera_ids: list[str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Camera settings")
        self.resize(720, 560)
        self._camera = CameraConfig.from_dict(camera_config.to_dict())
        self._detector_models = detector_models
        self._all_camera_ids = [camera_id for camera_id in all_camera_ids if camera_id != self._camera.camera_id]
        self._build_ui()
        self._load_values()
        self._update_source_controls()
        self._update_tracker_controls()

    @property
    def camera_config(self) -> CameraConfig:
        return CameraConfig.from_dict(self._camera.to_dict())

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        form = QFormLayout()

        self.enabled_checkbox = QCheckBox("Enabled")
        self.camera_id_input = QLineEdit()
        self.camera_name_input = QLineEdit()
        self.display_order_spin = QSpinBox()
        self.display_order_spin.setRange(0, 99)
        self.panel_color_input = QLineEdit()

        self.source_type_combo = QComboBox()
        self.source_type_combo.addItem("UDP stream", "udp")
        self.source_type_combo.addItem("Local MP4", "file")
        self.source_value_input = QLineEdit()
        self.browse_source_button = QPushButton("Browse...")
        self.sample_clips_button = QPushButton("Sample clips...")
        self.loop_file_checkbox = QCheckBox("Loop playback")

        self.detector_model_combo = QComboBox()
        for label, model_path in self._detector_models:
            self.detector_model_combo.addItem(label, model_path)
        self.enable_detection_checkbox = QCheckBox("Enable person detection")
        self.robust_detection_checkbox = QCheckBox("Robust detection")

        self.tracker_backend_combo = QComboBox()
        self.tracker_backend_combo.addItem("BoT-SORT", "botsort")
        self.tracker_reid_checkbox = QCheckBox("Tracker ReID")
        self.track_buffer_spin = QSpinBox()
        self.track_buffer_spin.setRange(1, 300)
        self.match_threshold_spin = QDoubleSpinBox()
        self.match_threshold_spin.setRange(0.05, 0.99)
        self.match_threshold_spin.setSingleStep(0.05)
        self.new_track_threshold_spin = QDoubleSpinBox()
        self.new_track_threshold_spin.setRange(0.05, 0.99)
        self.new_track_threshold_spin.setSingleStep(0.05)
        self.proximity_threshold_spin = QDoubleSpinBox()
        self.proximity_threshold_spin.setRange(0.05, 0.99)
        self.proximity_threshold_spin.setSingleStep(0.05)
        self.appearance_threshold_spin = QDoubleSpinBox()
        self.appearance_threshold_spin.setRange(0.05, 0.99)
        self.appearance_threshold_spin.setSingleStep(0.05)

        self.overlap_checks: dict[str, QCheckBox] = {}
        overlap_widget = QWidget()
        overlap_layout = QGridLayout(overlap_widget)
        overlap_layout.setContentsMargins(0, 0, 0, 0)
        overlap_layout.setSpacing(6)
        for index, camera_id in enumerate(self._all_camera_ids):
            checkbox = QCheckBox(camera_id)
            self.overlap_checks[camera_id] = checkbox
            overlap_layout.addWidget(checkbox, index // 2, index % 2)

        source_row = QWidget()
        source_layout = QHBoxLayout(source_row)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(self.source_value_input, 1)
        source_layout.addWidget(self.browse_source_button)
        source_layout.addWidget(self.sample_clips_button)

        form.addRow("", self.enabled_checkbox)
        form.addRow("Camera ID", self.camera_id_input)
        form.addRow("Camera name", self.camera_name_input)
        form.addRow("Display order", self.display_order_spin)
        form.addRow("Panel color", self.panel_color_input)
        form.addRow("Source type", self.source_type_combo)
        form.addRow("Source", source_row)
        form.addRow("", self.loop_file_checkbox)
        form.addRow("Detector model", self.detector_model_combo)
        form.addRow("", self.enable_detection_checkbox)
        form.addRow("", self.robust_detection_checkbox)
        form.addRow("Tracker", self.tracker_backend_combo)
        form.addRow("", self.tracker_reid_checkbox)
        form.addRow("Track buffer", self.track_buffer_spin)
        form.addRow("Match threshold", self.match_threshold_spin)
        form.addRow("New track threshold", self.new_track_threshold_spin)
        form.addRow("Proximity threshold", self.proximity_threshold_spin)
        form.addRow("Appearance threshold", self.appearance_threshold_spin)
        form.addRow("Overlap cameras", overlap_widget)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        root.addLayout(form)
        root.addWidget(self.button_box)

        self.button_box.accepted.connect(self._accept_if_valid)
        self.button_box.rejected.connect(self.reject)
        self.source_type_combo.currentIndexChanged.connect(self._update_source_controls)
        self.tracker_backend_combo.currentIndexChanged.connect(self._update_tracker_controls)
        self.browse_source_button.clicked.connect(self._browse_source)
        self.sample_clips_button.clicked.connect(self._open_samples)

    def _load_values(self) -> None:
        self.enabled_checkbox.setChecked(self._camera.enabled)
        self.camera_id_input.setText(self._camera.camera_id)
        self.camera_name_input.setText(self._camera.name)
        self.display_order_spin.setValue(self._camera.display_order)
        self.panel_color_input.setText(self._camera.panel_color)
        self.source_type_combo.setCurrentIndex(0 if self._camera.source_type == "udp" else 1)
        self.source_value_input.setText(self._camera.source_value or self._camera.udp_url)
        self.loop_file_checkbox.setChecked(self._camera.loop_file)
        index = self.detector_model_combo.findData(self._camera.detector_model_path)
        if index >= 0:
            self.detector_model_combo.setCurrentIndex(index)
        self.enable_detection_checkbox.setChecked(self._camera.enabled)
        self.robust_detection_checkbox.setChecked(self._camera.detector_use_augmentation)
        tracker_index = self.tracker_backend_combo.findData(self._camera.tracker_backend)
        self.tracker_backend_combo.setCurrentIndex(max(0, tracker_index))
        self.tracker_reid_checkbox.setChecked(self._camera.tracker_reid_enabled)
        self.track_buffer_spin.setValue(self._camera.tracker_track_buffer)
        self.match_threshold_spin.setValue(self._camera.tracker_match_thresh)
        self.new_track_threshold_spin.setValue(self._camera.tracker_new_track_thresh)
        self.proximity_threshold_spin.setValue(self._camera.tracker_proximity_thresh)
        self.appearance_threshold_spin.setValue(self._camera.tracker_appearance_thresh)
        for camera_id, checkbox in self.overlap_checks.items():
            checkbox.setChecked(camera_id in self._camera.overlap_camera_ids)

    @Slot()
    def _update_source_controls(self) -> None:
        file_mode = self.source_type_combo.currentData() == "file"
        self.browse_source_button.setVisible(file_mode)
        self.sample_clips_button.setVisible(file_mode)
        self.loop_file_checkbox.setVisible(file_mode)

    @Slot()
    def _update_tracker_controls(self) -> None:
        uses_reid = self.tracker_backend_combo.currentData() == "botsort"
        self.tracker_reid_checkbox.setEnabled(uses_reid)
        self.proximity_threshold_spin.setEnabled(uses_reid)
        self.appearance_threshold_spin.setEnabled(uses_reid)

    @Slot()
    def _browse_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video file",
            str(Path.cwd()),
            "Video files (*.mp4 *.mov *.mkv *.avi)",
        )
        if path:
            self.source_value_input.setText(path)

    @Slot()
    def _open_samples(self) -> None:
        SampleCatalogDialog(self).exec()

    @Slot()
    def _accept_if_valid(self) -> None:
        camera_id = self.camera_id_input.text().strip()
        if not camera_id:
            QMessageBox.warning(self, "Invalid camera", "Camera ID must not be empty.")
            return
        source_value = self.source_value_input.text().strip()
        if not source_value:
            QMessageBox.warning(self, "Invalid source", "Source must not be empty.")
            return
        if self.source_type_combo.currentData() == "file" and not Path(source_value).expanduser().exists():
            QMessageBox.warning(self, "Missing file", "Selected video file does not exist.")
            return

        self._camera.enabled = self.enabled_checkbox.isChecked()
        self._camera.camera_id = camera_id
        self._camera.name = self.camera_name_input.text().strip() or camera_id
        self._camera.display_order = self.display_order_spin.value()
        self._camera.panel_color = self.panel_color_input.text().strip() or "#2563eb"
        self._camera.source_type = str(self.source_type_combo.currentData())
        self._camera.source_value = source_value
        self._camera.loop_file = self.loop_file_checkbox.isChecked()
        if self._camera.source_type == "udp":
            self._camera.udp_url = source_value
        self._camera.detector_model_path = str(self.detector_model_combo.currentData())
        self._camera.detector_family = infer_detector_family(self._camera.detector_model_path)
        self._camera.detector_variant = infer_detector_variant(self._camera.detector_model_path)
        self._camera.enabled = self.enable_detection_checkbox.isChecked()
        self._camera.detector_use_augmentation = self.robust_detection_checkbox.isChecked()
        self._camera.tracker_backend = str(self.tracker_backend_combo.currentData())
        self._camera.tracker_family = self._camera.tracker_backend
        self._camera.tracker_with_reid = self.tracker_reid_checkbox.isChecked()
        self._camera.tracker_reid_enabled = self.tracker_reid_checkbox.isChecked()
        self._camera.tracker_track_buffer = self.track_buffer_spin.value()
        self._camera.tracker_match_thresh = self.match_threshold_spin.value()
        self._camera.tracker_new_track_thresh = self.new_track_threshold_spin.value()
        self._camera.tracker_proximity_thresh = self.proximity_threshold_spin.value()
        self._camera.tracker_appearance_thresh = self.appearance_threshold_spin.value()
        self._camera.tracker_config_path = "config/trackers/botsort.yaml"
        self._camera.overlap_camera_ids = [
            camera_id for camera_id, checkbox in self.overlap_checks.items() if checkbox.isChecked()
        ]
        self.accept()


class CameraManagerDialog(QDialog):
    cameras_applied = Signal(object)

    def __init__(
        self,
        cameras: list[CameraConfig],
        detector_models: list[tuple[str, str]],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage cameras")
        self.resize(820, 520)
        self._cameras = [CameraConfig.from_dict(camera.to_dict()) for camera in cameras]
        self._detector_models = detector_models
        self._build_ui()
        self._refresh_list()

    @property
    def cameras(self) -> list[CameraConfig]:
        return [CameraConfig.from_dict(camera.to_dict()) for camera in self._cameras]

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        content = QHBoxLayout()
        self.camera_list = QListWidget()
        content.addWidget(self.camera_list, 1)

        buttons = QVBoxLayout()
        self.add_button = QPushButton("Add camera")
        self.edit_button = QPushButton("Edit camera")
        self.remove_button = QPushButton("Remove")
        self.move_up_button = QPushButton("Move up")
        self.move_down_button = QPushButton("Move down")
        for button in (
            self.add_button,
            self.edit_button,
            self.remove_button,
            self.move_up_button,
            self.move_down_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        content.addLayout(buttons)

        self.helper_label = QLabel(
            "Overlap cameras determine where overlap deduplication may suppress double-counting. Calibrate each camera later against shared anchors."
        )
        self.helper_label.setWordWrap(True)
        self.overlap_status_label = QLabel("Auto overlap: no calibrated adjacent cameras yet")
        self.overlap_status_label.setWordWrap(True)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply)

        root.addLayout(content, 1)
        root.addWidget(self.helper_label)
        root.addWidget(self.overlap_status_label)
        root.addWidget(self.button_box)

        self.add_button.clicked.connect(self._add_camera)
        self.edit_button.clicked.connect(self._edit_selected_camera)
        self.remove_button.clicked.connect(self._remove_selected_camera)
        self.move_up_button.clicked.connect(lambda: self._move_selected(-1))
        self.move_down_button.clicked.connect(lambda: self._move_selected(1))
        self.button_box.button(QDialogButtonBox.Apply).clicked.connect(self._emit_apply)
        self.button_box.accepted.connect(self._accept_and_emit)
        self.button_box.rejected.connect(self.reject)
        self.camera_list.currentRowChanged.connect(self._refresh_overlap_status)

    def _refresh_list(self) -> None:
        self.camera_list.clear()
        for camera in sorted(self._cameras, key=lambda item: (item.display_order, item.camera_id)):
            item = QListWidgetItem(
                f"{camera.name} [{camera.camera_id}] {'(disabled)' if not camera.enabled else ''}".strip()
            )
            item.setData(Qt.UserRole, camera.camera_id)
            self.camera_list.addItem(item)
        if self.camera_list.count():
            self.camera_list.setCurrentRow(0)
        self._refresh_overlap_status()

    def _selected_index(self) -> int | None:
        current = self.camera_list.currentItem()
        if current is None:
            return None
        camera_id = current.data(Qt.UserRole)
        for index, camera in enumerate(self._cameras):
            if camera.camera_id == camera_id:
                return index
        return None

    @Slot()
    def _add_camera(self) -> None:
        next_index = len(self._cameras) + 1
        camera = CameraConfig(
            camera_id=f"camera-{next_index}",
            name=f"Camera {next_index}",
            display_order=next_index - 1,
        )
        dialog = CameraEditorDialog(camera, self._detector_models, [item.camera_id for item in self._cameras], self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        self._cameras.append(dialog.camera_config)
        self._refresh_list()

    @Slot()
    def _edit_selected_camera(self) -> None:
        index = self._selected_index()
        if index is None:
            return
        current_id = self._cameras[index].camera_id
        other_ids = [item.camera_id for item in self._cameras if item.camera_id != current_id]
        dialog = CameraEditorDialog(self._cameras[index], self._detector_models, other_ids, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        updated = dialog.camera_config
        if updated.camera_id != current_id and any(
            camera.camera_id == updated.camera_id for camera in self._cameras
        ):
            QMessageBox.warning(self, "Duplicate camera", "Camera ID must be unique.")
            return
        self._cameras[index] = updated
        for camera in self._cameras:
            if camera.camera_id == updated.camera_id:
                continue
            camera.overlap_camera_ids = [
                updated.camera_id if overlap_id == current_id else overlap_id
                for overlap_id in camera.overlap_camera_ids
            ]
        self._refresh_list()

    @Slot()
    def _remove_selected_camera(self) -> None:
        index = self._selected_index()
        if index is None:
            return
        removed = self._cameras.pop(index)
        for camera in self._cameras:
            camera.overlap_camera_ids = [
                overlap_id for overlap_id in camera.overlap_camera_ids if overlap_id != removed.camera_id
            ]
        self._refresh_list()

    def _move_selected(self, delta: int) -> None:
        index = self._selected_index()
        if index is None:
            return
        new_index = index + delta
        if new_index < 0 or new_index >= len(self._cameras):
            return
        self._cameras[index], self._cameras[new_index] = self._cameras[new_index], self._cameras[index]
        for display_order, camera in enumerate(self._cameras):
            camera.display_order = display_order
        self._refresh_list()
        self.camera_list.setCurrentRow(new_index)

    @Slot()
    def _emit_apply(self) -> None:
        self.cameras_applied.emit(self.cameras)

    @Slot()
    def _accept_and_emit(self) -> None:
        self.cameras_applied.emit(self.cameras)
        self.accept()

    @Slot()
    def _refresh_overlap_status(self) -> None:
        index = self._selected_index()
        if index is None:
            self.overlap_status_label.setText("Auto overlap: no camera selected")
            return
        camera = self._cameras[index]
        overlap_graph = build_camera_overlap_graph(self._cameras, OverlapDedupConfig())
        neighbors: list[str] = []
        details: list[str] = []
        for neighbor_id in sorted(overlap_graph.neighbors_of(camera.camera_id)):
            relation = overlap_graph.relation_for(camera.camera_id, neighbor_id)
            if relation is None:
                continue
            neighbors.append(neighbor_id)
            details.append(
                f"{neighbor_id} (overlap {relation.overlap_area_m2:.2f} m², gap {relation.min_boundary_distance_m:.2f} m)"
            )
        if not neighbors:
            self.overlap_status_label.setText(
                f"Auto overlap for {camera.camera_id}: no adjacent calibrated cameras detected yet"
            )
            return
        self.overlap_status_label.setText(
            f"Auto overlap for {camera.camera_id}: " + ", ".join(details)
        )
