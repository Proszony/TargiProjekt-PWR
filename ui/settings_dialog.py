from __future__ import annotations

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QVBoxLayout

from core.models import ProjectConfig


class SettingsDialog(QDialog):
    settings_applied = Signal(object)

    def __init__(self, project_config: ProjectConfig, is_running: bool, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project settings")
        self.resize(420, 220)
        self.setModal(True)
        self._project = ProjectConfig.from_dict(project_config.to_dict())
        self._is_running = is_running
        self._build_ui()
        self._load_values()

    @property
    def project_config(self) -> ProjectConfig:
        return ProjectConfig.from_dict(self._project.to_dict())

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        form = QFormLayout()

        helper = QLabel(
            "Only booth dwell timing is configurable here. Detection, sync, deduplication, and distributed transport run with product defaults."
        )
        helper.setWordWrap(True)

        self.zone_entry_spin = QDoubleSpinBox()
        self.zone_entry_spin.setRange(0.1, 10.0)
        self.zone_entry_spin.setSingleStep(0.05)

        self.zone_exit_grace_spin = QDoubleSpinBox()
        self.zone_exit_grace_spin.setRange(0.1, 10.0)
        self.zone_exit_grace_spin.setSingleStep(0.05)

        form.addRow("Zone entry confirmation [s]", self.zone_entry_spin)
        form.addRow("Zone exit grace [s]", self.zone_exit_grace_spin)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply)
        self.button_box.accepted.connect(self._accept_and_apply)
        self.button_box.button(QDialogButtonBox.Apply).clicked.connect(self._apply_clicked)
        self.button_box.rejected.connect(self.reject)

        root.addWidget(helper)
        root.addLayout(form)
        root.addWidget(self.button_box)

    def _load_values(self) -> None:
        self.zone_entry_spin.setValue(self._project.analytics.zone_entry_min_duration_s)
        self.zone_exit_grace_spin.setValue(self._project.analytics.zone_exit_grace_s)

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
        self.settings_applied.emit(self.project_config)
