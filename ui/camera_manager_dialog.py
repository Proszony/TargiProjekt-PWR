from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
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
from core.model_catalog import available_detection_models
from core.models import CameraConfig, OverlapDedupConfig
from ui.style_system import apply_chrome


class CameraEditorDialog(QDialog):
    def __init__(
        self,
        camera_config: CameraConfig,
        all_camera_ids: list[str],
        detector_models: list[tuple[str, str]],
        project_root: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Camera settings")
        self.resize(680, 460)
        self._camera = CameraConfig.from_dict(camera_config.to_dict())
        self._all_camera_ids = [camera_id for camera_id in all_camera_ids if camera_id != self._camera.camera_id]
        self._detector_models = list(detector_models)
        self._project_root = project_root
        self._build_ui()
        self._load_values()
        self._update_source_controls()
        apply_chrome(self)

    @property
    def camera_config(self) -> CameraConfig:
        return CameraConfig.from_dict(self._camera.to_dict())

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        title = QLabel("Camera setup")
        title.setObjectName("SectionTitle")
        subtitle = QLabel(
            "Define runtime mode, source, and possible overlap neighbors for one camera at a time."
        )
        subtitle.setObjectName("SectionSubtitle")
        subtitle.setWordWrap(True)
        form = QFormLayout()
        form.setSpacing(12)

        self.enabled_checkbox = QCheckBox("Enabled")
        self.camera_id_input = QLineEdit()
        self.camera_name_input = QLineEdit()
        self.display_order_spin = QSpinBox()
        self.display_order_spin.setRange(0, 999)

        self.runtime_mode_combo = QComboBox()
        self.runtime_mode_combo.addItem("Local runtime", "local")
        self.runtime_mode_combo.addItem("Remote worker", "remote")
        self.remote_worker_id_input = QLineEdit()

        self.source_type_combo = QComboBox()
        self.source_type_combo.addItem("UDP stream", "udp")
        self.source_type_combo.addItem("Local MP4", "file")
        self.project_mp4_combo = QComboBox()
        self.project_mp4_combo.addItem("Use external path or stream value", "")
        for path in sorted(self._project_root.glob("*.mp4")):
            self.project_mp4_combo.addItem(path.name, path.name)
        self.source_value_input = QLineEdit()
        self.source_value_input.setPlaceholderText("UDP URL, project MP4 filename, or external file path")
        self.browse_source_button = QPushButton("Browse external...")
        self.loop_file_checkbox = QCheckBox("Loop playback")
        self.detector_model_combo = QComboBox()
        for label, model_path in self._detector_models:
            self.detector_model_combo.addItem(label, model_path)

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

        form.addRow("", self.enabled_checkbox)
        form.addRow("Camera ID", self.camera_id_input)
        form.addRow("Camera name", self.camera_name_input)
        form.addRow("Display order", self.display_order_spin)
        form.addRow("Runtime mode", self.runtime_mode_combo)
        form.addRow("Remote worker ID", self.remote_worker_id_input)
        form.addRow("Source type", self.source_type_combo)
        form.addRow("Project MP4", self.project_mp4_combo)
        form.addRow("Source", source_row)
        form.addRow("", self.loop_file_checkbox)
        form.addRow("Detector model", self.detector_model_combo)
        form.addRow("Overlap cameras", overlap_widget)

        helper = QLabel(
            "Overlap cameras only define where double-count suppression may happen. Calibration and booth mapping are handled elsewhere."
        )
        helper.setObjectName("HintText")
        helper.setWordWrap(True)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.button(QDialogButtonBox.Ok).setProperty("kind", "primary")
        self.button_box.button(QDialogButtonBox.Cancel).setProperty("kind", "danger")
        root.addWidget(title)
        root.addWidget(subtitle)
        root.addLayout(form)
        root.addWidget(helper)
        root.addWidget(self.button_box)

        self.button_box.accepted.connect(self._accept_if_valid)
        self.button_box.rejected.connect(self.reject)
        self.runtime_mode_combo.currentIndexChanged.connect(self._update_source_controls)
        self.source_type_combo.currentIndexChanged.connect(self._update_source_controls)
        self.project_mp4_combo.currentIndexChanged.connect(self._select_project_mp4)
        self.browse_source_button.clicked.connect(self._browse_source)

    def _load_values(self) -> None:
        self.enabled_checkbox.setChecked(self._camera.enabled)
        self.camera_id_input.setText(self._camera.camera_id)
        self.camera_name_input.setText(self._camera.name)
        self.display_order_spin.setValue(self._camera.display_order)
        runtime_index = self.runtime_mode_combo.findData(self._camera.runtime_mode)
        self.runtime_mode_combo.setCurrentIndex(max(0, runtime_index))
        self.remote_worker_id_input.setText(self._camera.remote_worker_id)
        self.source_type_combo.setCurrentIndex(0 if self._camera.source_type == "udp" else 1)
        self.source_value_input.setText(self._camera.source_value)
        project_mp4_index = self.project_mp4_combo.findData(self._camera.source_value)
        if project_mp4_index >= 0:
            self.project_mp4_combo.setCurrentIndex(project_mp4_index)
        self.loop_file_checkbox.setChecked(self._camera.loop_file)
        detector_index = self.detector_model_combo.findData(self._camera.detector_model_path)
        if detector_index >= 0:
            self.detector_model_combo.setCurrentIndex(detector_index)
        for camera_id, checkbox in self.overlap_checks.items():
            checkbox.setChecked(camera_id in self._camera.overlap_camera_ids)

    @Slot()
    def _update_source_controls(self) -> None:
        file_mode = self.source_type_combo.currentData() == "file"
        remote_mode = self.runtime_mode_combo.currentData() == "remote"
        self.remote_worker_id_input.setEnabled(remote_mode)
        self.project_mp4_combo.setVisible(file_mode)
        self.browse_source_button.setVisible(file_mode)
        self.loop_file_checkbox.setVisible(file_mode)

    @Slot()
    def _select_project_mp4(self) -> None:
        relative_path = self.project_mp4_combo.currentData()
        if isinstance(relative_path, str) and relative_path:
            self.source_value_input.setText(relative_path)

    @Slot()
    def _browse_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video file",
            str(self._project_root),
            "Video files (*.mp4 *.mov *.mkv *.avi)",
        )
        if path:
            self.source_value_input.setText(path)
            self.project_mp4_combo.setCurrentIndex(0)

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
        if (
            self.runtime_mode_combo.currentData() == "local"
            and self.source_type_combo.currentData() == "file"
            and not self._source_file_exists(source_value)
        ):
            QMessageBox.warning(self, "Missing file", "Selected video file does not exist.")
            return

        self._camera.enabled = self.enabled_checkbox.isChecked()
        self._camera.camera_id = camera_id
        self._camera.name = self.camera_name_input.text().strip() or camera_id
        self._camera.display_order = self.display_order_spin.value()
        self._camera.runtime_mode = str(self.runtime_mode_combo.currentData())
        self._camera.remote_worker_id = (
            self.remote_worker_id_input.text().strip() if self._camera.runtime_mode == "remote" else ""
        )
        self._camera.detector_model_path = str(self.detector_model_combo.currentData())
        self._camera.source_type = str(self.source_type_combo.currentData())
        self._camera.source_value = source_value
        self._camera.loop_file = self.loop_file_checkbox.isChecked() if self._camera.source_type == "file" else False
        self._camera.overlap_camera_ids = [
            camera_id for camera_id, checkbox in self.overlap_checks.items() if checkbox.isChecked()
        ]
        self.accept()

    def _source_file_exists(self, source_value: str) -> bool:
        path = Path(source_value).expanduser()
        if path.is_absolute():
            return path.exists()
        return (self._project_root / path).exists()


class CameraManagerDialog(QDialog):
    cameras_applied = Signal(object)

    def __init__(
        self,
        cameras: list[CameraConfig],
        project_root: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage cameras")
        self.resize(820, 520)
        self._cameras = [CameraConfig.from_dict(camera.to_dict()) for camera in cameras]
        self._detector_models = available_detection_models(project_root)
        self._project_root = project_root
        self._build_ui()
        self._refresh_list()
        apply_chrome(self)

    @property
    def cameras(self) -> list[CameraConfig]:
        return [CameraConfig.from_dict(camera.to_dict()) for camera in self._cameras]

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        title = QLabel("Camera manager")
        title.setObjectName("SectionTitle")
        subtitle = QLabel(
            "Build the camera lineup, adjust ordering, and inspect overlap adjacency before you return to the live workspace."
        )
        subtitle.setObjectName("SectionSubtitle")
        subtitle.setWordWrap(True)
        content = QHBoxLayout()
        content.setSpacing(14)
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
            "Configure only source assignment and overlap topology here. Detection, tracking, and ReID run with internal product defaults."
        )
        self.helper_label.setObjectName("HintText")
        self.helper_label.setWordWrap(True)
        self.overlap_status_label = QLabel("Auto overlap: no calibrated adjacent cameras yet")
        self.overlap_status_label.setObjectName("MutedText")
        self.overlap_status_label.setWordWrap(True)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply)
        self.button_box.button(QDialogButtonBox.Ok).setProperty("kind", "primary")
        self.button_box.button(QDialogButtonBox.Apply).setProperty("kind", "primary")
        self.button_box.button(QDialogButtonBox.Cancel).setProperty("kind", "danger")

        root.addWidget(title)
        root.addWidget(subtitle)
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
            mode_suffix = "[remote]" if camera.runtime_mode == "remote" else "[local]"
            state_suffix = "(disabled)" if not camera.enabled else ""
            item = QListWidgetItem(f"{camera.name} [{camera.camera_id}] {mode_suffix} {state_suffix}".strip())
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
        dialog = CameraEditorDialog(
            camera,
            [item.camera_id for item in self._cameras],
            self._detector_models,
            self._project_root,
            self,
        )
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
        dialog = CameraEditorDialog(
            self._cameras[index],
            other_ids,
            self._detector_models,
            self._project_root,
            self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        updated = dialog.camera_config
        if updated.camera_id != current_id and any(camera.camera_id == updated.camera_id for camera in self._cameras):
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
        details: list[str] = []
        for neighbor_id in sorted(overlap_graph.neighbors_of(camera.camera_id)):
            relation = overlap_graph.relation_for(camera.camera_id, neighbor_id)
            if relation is None:
                continue
            details.append(
                f"{neighbor_id} (overlap {relation.overlap_area_m2:.2f} m², gap {relation.min_boundary_distance_m:.2f} m)"
            )
        if not details:
            self.overlap_status_label.setText(
                f"Auto overlap for {camera.camera_id}: no adjacent calibrated cameras detected yet"
            )
            return
        self.overlap_status_label.setText(f"Auto overlap for {camera.camera_id}: " + ", ".join(details))
