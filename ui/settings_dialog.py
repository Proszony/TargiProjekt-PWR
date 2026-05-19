from __future__ import annotations

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.models import PlaybackSyncConfig, ProjectConfig


class SettingsDialog(QDialog):
    settings_applied = Signal(object)

    def __init__(self, project_config: ProjectConfig, is_running: bool, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project settings")
        self.resize(560, 360)
        self.setModal(True)
        self._project = ProjectConfig.from_dict(project_config.to_dict())
        self._is_running = is_running
        self._build_ui()
        self._load_values()
        self._apply_running_state()

    @property
    def project_config(self) -> ProjectConfig:
        return ProjectConfig.from_dict(self._project.to_dict())

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.analytics_tab = QWidget()
        self.playback_tab = QWidget()

        analytics_form = QFormLayout(self.analytics_tab)
        self.analytics_info = QLabel(
            "This product is optimized for booth analytics: dwell time, occupancy, visits, and overlap-only deduplication."
        )
        self.analytics_info.setWordWrap(True)
        self.zone_entry_spin = QDoubleSpinBox()
        self.zone_entry_spin.setRange(0.1, 10.0)
        self.zone_entry_spin.setSingleStep(0.05)
        self.zone_exit_grace_spin = QDoubleSpinBox()
        self.zone_exit_grace_spin.setRange(0.1, 10.0)
        self.zone_exit_grace_spin.setSingleStep(0.05)
        self.dedup_enabled_checkbox = QCheckBox("Enable overlap deduplication")
        self.dedup_confirmation_spin = QDoubleSpinBox()
        self.dedup_confirmation_spin.setRange(1.0, 10.0)
        self.dedup_confirmation_spin.setSingleStep(1.0)
        self.dedup_similarity_spin = QDoubleSpinBox()
        self.dedup_similarity_spin.setRange(0.0, 1.0)
        self.dedup_similarity_spin.setSingleStep(0.01)
        self.dedup_margin_spin = QDoubleSpinBox()
        self.dedup_margin_spin.setRange(0.0, 1.0)
        self.dedup_margin_spin.setSingleStep(0.01)
        self.live_rate_spin = QDoubleSpinBox()
        self.live_rate_spin.setRange(1.0, 10.0)
        self.live_rate_spin.setSingleStep(0.5)

        analytics_form.addRow(self.analytics_info)
        analytics_form.addRow("Zone entry confirmation [s]", self.zone_entry_spin)
        analytics_form.addRow("Zone exit grace [s]", self.zone_exit_grace_spin)
        analytics_form.addRow("", self.dedup_enabled_checkbox)
        analytics_form.addRow("Dedup confirmation frames", self.dedup_confirmation_spin)
        analytics_form.addRow("Dedup similarity threshold", self.dedup_similarity_spin)
        analytics_form.addRow("Dedup score margin", self.dedup_margin_spin)
        analytics_form.addRow("Live UI refresh rate [Hz]", self.live_rate_spin)

        playback_form = QFormLayout(self.playback_tab)
        self.file_sync_enabled_checkbox = QCheckBox("Enable strict sync for file sources")
        self.target_fps_spin = QDoubleSpinBox()
        self.target_fps_spin.setRange(1.0, 120.0)
        self.target_fps_spin.setSingleStep(1.0)
        self.sync_tolerance_spin = QDoubleSpinBox()
        self.sync_tolerance_spin.setRange(0.001, 1.0)
        self.sync_tolerance_spin.setSingleStep(0.005)
        self.late_drop_spin = QDoubleSpinBox()
        self.late_drop_spin.setRange(0.001, 2.0)
        self.late_drop_spin.setSingleStep(0.01)
        self.stale_packet_spin = QDoubleSpinBox()
        self.stale_packet_spin.setRange(0.001, 2.0)
        self.stale_packet_spin.setSingleStep(0.01)
        self.camera_missing_spin = QDoubleSpinBox()
        self.camera_missing_spin.setRange(0.001, 5.0)
        self.camera_missing_spin.setSingleStep(0.05)

        playback_form.addRow("", self.file_sync_enabled_checkbox)
        playback_form.addRow("Target FPS", self.target_fps_spin)
        playback_form.addRow("Sync tolerance [s]", self.sync_tolerance_spin)
        playback_form.addRow("Late frame drop [s]", self.late_drop_spin)
        playback_form.addRow("Stale packet [s]", self.stale_packet_spin)
        playback_form.addRow("Missing camera [s]", self.camera_missing_spin)

        self.tabs.addTab(self.analytics_tab, "Analytics")
        self.tabs.addTab(self.playback_tab, "Playback")

        helper = QLabel(
            "Use Manage cameras for source, detector, tracker, and overlap topology. Calibration and booths are edited outside this dialog."
        )
        helper.setWordWrap(True)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply)
        self.button_box.accepted.connect(self._accept_and_apply)
        self.button_box.button(QDialogButtonBox.Apply).clicked.connect(self._apply_clicked)
        self.button_box.rejected.connect(self.reject)

        root.addWidget(self.tabs)
        root.addWidget(helper)
        root.addWidget(self.button_box)

    def _load_values(self) -> None:
        analytics = self._project.analytics
        playback_sync: PlaybackSyncConfig = self._project.playback_sync
        self.zone_entry_spin.setValue(analytics.zone_entry_min_duration_s)
        self.zone_exit_grace_spin.setValue(analytics.zone_exit_grace_s)
        self.dedup_enabled_checkbox.setChecked(analytics.dedup_overlap_enabled)
        self.dedup_confirmation_spin.setValue(float(analytics.dedup_confirmation_frames))
        self.dedup_similarity_spin.setValue(analytics.dedup_similarity_threshold)
        self.dedup_margin_spin.setValue(analytics.dedup_margin_min)
        self.live_rate_spin.setValue(analytics.live_snapshot_rate_hz)
        self.file_sync_enabled_checkbox.setChecked(playback_sync.enabled_for_file_sources)
        self.target_fps_spin.setValue(playback_sync.target_fps)
        self.sync_tolerance_spin.setValue(playback_sync.sync_tolerance_s)
        self.late_drop_spin.setValue(playback_sync.late_frame_drop_threshold_s)
        self.stale_packet_spin.setValue(playback_sync.stale_packet_threshold_s)
        self.camera_missing_spin.setValue(playback_sync.camera_missing_timeout_s)

    def _apply_running_state(self) -> None:
        if not self._is_running:
            return
        self.file_sync_enabled_checkbox.setEnabled(False)

    @Slot()
    def _apply_clicked(self) -> None:
        self._emit()

    @Slot()
    def _accept_and_apply(self) -> None:
        self._emit()
        self.accept()

    def _emit(self) -> None:
        self._project.analytics.zone_entry_min_duration_s = self.zone_entry_spin.value()
        self._project.analytics.zone_exit_grace_s = self.zone_exit_grace_spin.value()
        self._project.analytics.dedup_overlap_enabled = self.dedup_enabled_checkbox.isChecked()
        self._project.analytics.dedup_confirmation_frames = int(round(self.dedup_confirmation_spin.value()))
        self._project.analytics.dedup_similarity_threshold = self.dedup_similarity_spin.value()
        self._project.analytics.dedup_margin_min = self.dedup_margin_spin.value()
        self._project.analytics.live_snapshot_rate_hz = self.live_rate_spin.value()

        self._project.overlap_dedup.enabled = self.dedup_enabled_checkbox.isChecked()
        self._project.overlap_dedup.confirmation_frames = int(round(self.dedup_confirmation_spin.value()))
        self._project.overlap_dedup.similarity_threshold = self.dedup_similarity_spin.value()
        self._project.overlap_dedup.margin_min = self.dedup_margin_spin.value()

        self._project.playback_sync.enabled_for_file_sources = self.file_sync_enabled_checkbox.isChecked()
        self._project.playback_sync.target_fps = self.target_fps_spin.value()
        self._project.playback_sync.sync_tolerance_s = self.sync_tolerance_spin.value()
        self._project.playback_sync.late_frame_drop_threshold_s = self.late_drop_spin.value()
        self._project.playback_sync.stale_packet_threshold_s = self.stale_packet_spin.value()
        self._project.playback_sync.camera_missing_timeout_s = self.camera_missing_spin.value()
        self.settings_applied.emit(self.project_config)
