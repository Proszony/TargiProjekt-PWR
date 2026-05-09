from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.models import CameraConfig, VenueMapConfig
from ui.sample_catalog_dialog import SampleCatalogDialog


class SettingsDialog(QDialog):
    settings_applied = Signal(object, object)

    def __init__(
        self,
        camera_config: CameraConfig,
        venue_map: VenueMapConfig,
        detector_models: list[tuple[str, str]],
        is_running: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(760, 520)
        self.setModal(True)
        self._camera_config = CameraConfig.from_dict(camera_config.to_dict())
        self._venue_map = VenueMapConfig.from_dict(venue_map.to_dict())
        self._detector_models = detector_models
        self._is_running = is_running

        self._build_ui()
        self._connect_signals()
        self._load_values()
        self._update_source_controls()
        self._update_tracker_controls()
        self._apply_running_state()

    @property
    def camera_config(self) -> CameraConfig:
        return CameraConfig.from_dict(self._camera_config.to_dict())

    @property
    def venue_map(self) -> VenueMapConfig:
        return VenueMapConfig.from_dict(self._venue_map.to_dict())

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self.tabs = QTabWidget()

        self.general_tab = QWidget()
        self.source_tab = QWidget()
        self.detection_tab = QWidget()
        self.tracking_tab = QWidget()
        self.venue_tab = QWidget()

        self.camera_id_input = QLineEdit()
        self.camera_name_input = QLineEdit()

        self.source_type_combo = QComboBox()
        self.source_type_combo.addItem("UDP stream", "udp")
        self.source_type_combo.addItem("Local MP4", "file")
        self.source_value_input = QLineEdit()
        self.browse_source_button = QPushButton("Browse...")
        self.sample_clips_button = QPushButton("Sample clips...")
        self.loop_file_checkbox = QCheckBox("Loop playback")

        self.enable_detection_checkbox = QCheckBox("Enable person detection")
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.01, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.detector_model_combo = QComboBox()
        self.detector_model_combo.setToolTip(
            "Built dynamically from files in the local models folder. Only plain detection weights are listed."
        )
        self.robust_detection_checkbox = QCheckBox("Robust detection")
        self.robust_detection_checkbox.setToolTip(
            "Runs augmented inference on each frame, such as extra scales and flips. It is slower, but it can help with side/back views, small people, and harder surveillance shots."
        )
        self.inference_size_combo = QComboBox()
        self.inference_size_combo.addItems(["160", "224", "320", "416", "640", "768", "960"])
        self.inference_size_combo.setToolTip(
            "YOLO internal input size. Smaller values are faster but less stable; larger values are slower but usually detect people more reliably."
        )

        self.tracker_backend_combo = QComboBox()
        self.tracker_backend_combo.addItem("BoT-SORT", "botsort")
        self.tracker_backend_combo.addItem("ByteTrack", "bytetrack")
        self.tracker_reid_checkbox = QCheckBox("Tracker ReID")
        self.tracker_buffer_spin = QSpinBox()
        self.tracker_buffer_spin.setRange(1, 300)
        self.tracker_match_thresh_spin = QDoubleSpinBox()
        self.tracker_match_thresh_spin.setRange(0.05, 0.99)
        self.tracker_match_thresh_spin.setSingleStep(0.05)
        self.tracker_new_track_thresh_spin = QDoubleSpinBox()
        self.tracker_new_track_thresh_spin.setRange(0.05, 0.99)
        self.tracker_new_track_thresh_spin.setSingleStep(0.05)
        self.tracker_proximity_thresh_spin = QDoubleSpinBox()
        self.tracker_proximity_thresh_spin.setRange(0.05, 0.99)
        self.tracker_proximity_thresh_spin.setSingleStep(0.05)
        self.tracker_appearance_thresh_spin = QDoubleSpinBox()
        self.tracker_appearance_thresh_spin.setRange(0.05, 0.99)
        self.tracker_appearance_thresh_spin.setSingleStep(0.05)

        self.world_width_input = QDoubleSpinBox()
        self.world_width_input.setRange(1.0, 500.0)
        self.world_height_input = QDoubleSpinBox()
        self.world_height_input.setRange(1.0, 500.0)

        for label, model_path in self._detector_models:
            self.detector_model_combo.addItem(label, model_path)

        general_form = QFormLayout(self.general_tab)
        general_form.addRow("Camera ID", self.camera_id_input)
        general_form.addRow("Camera name", self.camera_name_input)

        source_form = QFormLayout(self.source_tab)
        source_row = QWidget()
        source_layout = QHBoxLayout(source_row)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(self.source_value_input, 1)
        source_layout.addWidget(self.browse_source_button)
        source_layout.addWidget(self.sample_clips_button)
        source_form.addRow("Source type", self.source_type_combo)
        source_form.addRow("Source", source_row)
        source_form.addRow("", self.loop_file_checkbox)

        detection_form = QFormLayout(self.detection_tab)
        detection_form.addRow("", self.enable_detection_checkbox)
        detection_form.addRow("Confidence", self.confidence_spin)
        detection_form.addRow("Detector model", self.detector_model_combo)
        detection_form.addRow("", self.robust_detection_checkbox)
        detection_form.addRow("Inference size", self.inference_size_combo)

        tracking_form = QFormLayout(self.tracking_tab)
        tracking_form.addRow("Tracker", self.tracker_backend_combo)
        tracking_form.addRow("", self.tracker_reid_checkbox)
        tracking_form.addRow("Track buffer", self.tracker_buffer_spin)
        tracking_form.addRow("Match threshold", self.tracker_match_thresh_spin)
        tracking_form.addRow("New track threshold", self.tracker_new_track_thresh_spin)
        tracking_form.addRow("Proximity threshold", self.tracker_proximity_thresh_spin)
        tracking_form.addRow("Appearance threshold", self.tracker_appearance_thresh_spin)

        venue_form = QFormLayout(self.venue_tab)
        venue_form.addRow("Venue width [m]", self.world_width_input)
        venue_form.addRow("Venue height [m]", self.world_height_input)

        self.tabs.addTab(self.general_tab, "General")
        self.tabs.addTab(self.source_tab, "Source")
        self.tabs.addTab(self.detection_tab, "Detection")
        self.tabs.addTab(self.tracking_tab, "Tracking")
        self.tabs.addTab(self.venue_tab, "Venue")

        helper = QLabel(
            "Changes from OK and Apply update the live session when safe. Save config from the main window writes them to disk."
        )
        helper.setWordWrap(True)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )

        root.addWidget(self.tabs)
        root.addWidget(helper)
        root.addWidget(self.button_box)

    def _connect_signals(self) -> None:
        self.source_type_combo.currentIndexChanged.connect(self._on_source_type_changed)
        self.tracker_backend_combo.currentIndexChanged.connect(self._on_tracker_backend_changed)
        self.browse_source_button.clicked.connect(self._browse_source_file)
        self.sample_clips_button.clicked.connect(self._open_sample_catalog)
        self.button_box.accepted.connect(self._accept_and_apply)
        self.button_box.button(QDialogButtonBox.Apply).clicked.connect(self._apply_clicked)
        self.button_box.rejected.connect(self.reject)

    def _load_values(self) -> None:
        self.camera_id_input.setText(self._camera_config.camera_id)
        self.camera_name_input.setText(self._camera_config.name)
        source_type = self._camera_config.source_type if self._camera_config.source_type in {"udp", "file"} else "udp"
        self.source_type_combo.setCurrentIndex(0 if source_type == "udp" else 1)
        self.source_value_input.setText(self._camera_config.source_value or self._camera_config.udp_url)
        self.loop_file_checkbox.setChecked(self._camera_config.loop_file)
        model_index = self.detector_model_combo.findData(self._camera_config.detector_model_path)
        if model_index < 0 and self._camera_config.detector_model_path:
            self.detector_model_combo.addItem(
                f"{Path(self._camera_config.detector_model_path).name} (configured)",
                self._camera_config.detector_model_path,
            )
            model_index = self.detector_model_combo.findData(self._camera_config.detector_model_path)
        self.detector_model_combo.setCurrentIndex(max(0, model_index))
        self.enable_detection_checkbox.setChecked(self._camera_config.enabled)
        self.confidence_spin.setValue(0.25)
        self.robust_detection_checkbox.setChecked(self._camera_config.detector_use_augmentation)
        self.inference_size_combo.setCurrentText("640")
        tracker_index = self.tracker_backend_combo.findData(self._camera_config.tracker_backend)
        self.tracker_backend_combo.setCurrentIndex(max(0, tracker_index))
        self.tracker_reid_checkbox.setChecked(self._camera_config.tracker_reid_enabled)
        self.tracker_buffer_spin.setValue(self._camera_config.tracker_track_buffer)
        self.tracker_match_thresh_spin.setValue(self._camera_config.tracker_match_thresh)
        self.tracker_new_track_thresh_spin.setValue(self._camera_config.tracker_new_track_thresh)
        self.tracker_proximity_thresh_spin.setValue(self._camera_config.tracker_proximity_thresh)
        self.tracker_appearance_thresh_spin.setValue(self._camera_config.tracker_appearance_thresh)
        self.world_width_input.setValue(self._venue_map.world_width_m)
        self.world_height_input.setValue(self._venue_map.world_height_m)

    def _apply_running_state(self) -> None:
        if not self._is_running:
            return
        for widget in (
            self.camera_id_input,
            self.camera_name_input,
            self.source_type_combo,
            self.source_value_input,
            self.world_width_input,
            self.world_height_input,
        ):
            widget.setEnabled(False)
        self.loop_file_checkbox.setEnabled(False)
        self.browse_source_button.setEnabled(False)

    @Slot()
    def _on_source_type_changed(self) -> None:
        self._update_source_controls()

    @Slot()
    def _on_tracker_backend_changed(self) -> None:
        self._update_tracker_controls()

    def _update_source_controls(self) -> None:
        file_mode = self.source_type_combo.currentData() == "file"
        self.browse_source_button.setVisible(file_mode)
        self.loop_file_checkbox.setVisible(file_mode)
        self.sample_clips_button.setVisible(file_mode)
        if file_mode:
            self.source_value_input.setPlaceholderText("/path/to/test-video.mp4")
        else:
            self.source_value_input.setPlaceholderText("udp://0.0.0.0:5000")

    def _update_tracker_controls(self) -> None:
        uses_reid = self.tracker_backend_combo.currentData() == "botsort"
        self.tracker_reid_checkbox.setEnabled(uses_reid)
        self.tracker_proximity_thresh_spin.setEnabled(uses_reid)
        self.tracker_appearance_thresh_spin.setEnabled(uses_reid)

    @Slot()
    def _browse_source_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select MP4 source",
            str(Path.cwd()),
            "Video files (*.mp4 *.mov *.mkv *.avi)",
        )
        if path:
            self.source_value_input.setText(path)

    @Slot()
    def _open_sample_catalog(self) -> None:
        dialog = SampleCatalogDialog(self)
        dialog.exec()

    @Slot()
    def _apply_clicked(self) -> None:
        self._validate_and_emit()

    @Slot()
    def _accept_and_apply(self) -> None:
        if self._validate_and_emit():
            self.accept()

    def _validate_and_emit(self) -> bool:
        camera_id = self.camera_id_input.text().strip()
        if not camera_id:
            QMessageBox.warning(self, "Invalid camera", "Camera ID must not be empty.")
            self.tabs.setCurrentWidget(self.general_tab)
            return False

        source_value = self.source_value_input.text().strip()
        source_type = str(self.source_type_combo.currentData())
        if not source_value:
            QMessageBox.warning(self, "Invalid source", "Source must not be empty.")
            self.tabs.setCurrentWidget(self.source_tab)
            return False
        if source_type == "file" and not Path(source_value).expanduser().exists():
            QMessageBox.warning(self, "Missing file", "Selected video file does not exist.")
            self.tabs.setCurrentWidget(self.source_tab)
            return False
        if not self.detector_model_combo.currentData():
            QMessageBox.warning(self, "Invalid model", "Detector model must be selected.")
            self.tabs.setCurrentWidget(self.detection_tab)
            return False

        self._camera_config.camera_id = camera_id
        self._camera_config.name = self.camera_name_input.text().strip() or camera_id
        self._camera_config.source_type = source_type
        self._camera_config.source_value = source_value
        self._camera_config.loop_file = self.loop_file_checkbox.isChecked()
        self._camera_config.detector_model_path = str(self.detector_model_combo.currentData())
        self._camera_config.detector_use_augmentation = self.robust_detection_checkbox.isChecked()
        self._camera_config.enabled = self.enable_detection_checkbox.isChecked()
        self._camera_config.tracker_backend = str(self.tracker_backend_combo.currentData())
        self._camera_config.tracker_reid_enabled = self.tracker_reid_checkbox.isChecked()
        self._camera_config.tracker_track_buffer = self.tracker_buffer_spin.value()
        self._camera_config.tracker_match_thresh = self.tracker_match_thresh_spin.value()
        self._camera_config.tracker_new_track_thresh = self.tracker_new_track_thresh_spin.value()
        self._camera_config.tracker_proximity_thresh = self.tracker_proximity_thresh_spin.value()
        self._camera_config.tracker_appearance_thresh = self.tracker_appearance_thresh_spin.value()
        self._camera_config.tracker_config_path = (
            "config/trackers/botsort.yaml"
            if self._camera_config.tracker_backend == "botsort"
            else "config/trackers/bytetrack.yaml"
        )
        if self._camera_config.source_type == "udp":
            self._camera_config.udp_url = self._camera_config.source_value

        self._venue_map.world_width_m = self.world_width_input.value()
        self._venue_map.world_height_m = self.world_height_input.value()

        self.settings_applied.emit(self.camera_config, self.venue_map)
        return True
