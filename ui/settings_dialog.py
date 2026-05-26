from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from core.models import ProjectConfig
from ui.style_system import apply_chrome


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
        apply_chrome(self)
        self._apply_local_styles()

    @property
    def project_config(self) -> ProjectConfig:
        return ProjectConfig.from_dict(self._project.to_dict())

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 18)
        root.setSpacing(16)

        hero = QFrame()
        hero.setObjectName("SettingsHero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(0, 0, 0, 0)
        hero_layout.setSpacing(6)

        eyebrow = QLabel("Booth timing")
        eyebrow.setObjectName("HeroEyebrow")
        title = QLabel("Time tracking controls")
        title.setObjectName("SectionTitle")
        subtitle = QLabel(
            "Small timing changes decide when a person becomes counted inside a booth and when they are released."
        )
        subtitle.setObjectName("SectionSubtitle")
        subtitle.setWordWrap(True)

        hero_layout.addWidget(eyebrow)
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)

        self.zone_entry_spin = QDoubleSpinBox()
        self._configure_seconds_spin(self.zone_entry_spin)

        self.zone_exit_grace_spin = QDoubleSpinBox()
        self._configure_seconds_spin(self.zone_exit_grace_spin)

        controls = QVBoxLayout()
        controls.setSpacing(10)
        controls.addWidget(
            self._build_setting_row(
                "Entry confirmation",
                "Person must remain inside a booth this long before a visit starts.",
                self.zone_entry_spin,
            )
        )
        controls.addWidget(
            self._build_setting_row(
                "Exit grace",
                "Brief tracking gaps shorter than this keep the booth visit alive.",
                self.zone_exit_grace_spin,
            )
        )

        runtime_note = QLabel(
            "Changes apply to booth time metrics only. Detection, sync, counting, and distributed transport stay unchanged."
        )
        runtime_note.setObjectName("SettingsNote")
        runtime_note.setWordWrap(True)
        if self._is_running:
            runtime_note.setText(
                "Runtime is active. Applied values affect new booth timing decisions without restarting streams."
            )

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply)
        self.button_box.button(QDialogButtonBox.Ok).setText("Save and close")
        self.button_box.button(QDialogButtonBox.Apply).setText("Apply")
        self.button_box.button(QDialogButtonBox.Cancel).setText("Cancel")
        self.button_box.button(QDialogButtonBox.Ok).setProperty("kind", "primary")
        self.button_box.accepted.connect(self._accept_and_apply)
        self.button_box.button(QDialogButtonBox.Apply).clicked.connect(self._apply_clicked)
        self.button_box.rejected.connect(self.reject)

        root.addWidget(hero)
        root.addLayout(controls)
        root.addWidget(runtime_note)
        root.addStretch(1)
        root.addWidget(self.button_box)

    @staticmethod
    def _configure_seconds_spin(spin: QDoubleSpinBox) -> None:
        spin.setRange(0.1, 10.0)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setSuffix(" s")
        spin.setFixedSize(108, 34)
        spin.setAlignment(Qt.AlignRight)

    @staticmethod
    def _build_setting_row(title: str, description: str, control: QDoubleSpinBox) -> QFrame:
        frame = QFrame()
        frame.setObjectName("SettingRow")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(14)

        text_stack = QVBoxLayout()
        text_stack.setSpacing(4)
        title_label = QLabel(title)
        title_label.setObjectName("SettingTitle")
        description_label = QLabel(description)
        description_label.setObjectName("FieldHint")
        description_label.setWordWrap(True)
        text_stack.addWidget(title_label)
        text_stack.addWidget(description_label)

        layout.addLayout(text_stack, 1)
        layout.addWidget(control, 0, Qt.AlignRight | Qt.AlignVCenter)
        return frame

    def _apply_local_styles(self) -> None:
        self.setStyleSheet(
            self.styleSheet()
            + """
            QFrame#SettingsHero {
                background: transparent;
                border: 0;
            }
            QFrame#SettingRow {
                background: #0d1721;
                border: 1px solid #1f3141;
                border-radius: 12px;
            }
            QLabel#SettingTitle {
                color: #f2f7fb;
                font-size: 14px;
                font-weight: 700;
            }
            QLabel#SettingsNote {
                background: #0a131b;
                border: 1px solid #1f3141;
                border-radius: 10px;
                color: #9fb1c0;
                padding: 10px 12px;
            }
            QDoubleSpinBox {
                font-size: 13px;
                font-weight: 600;
            }
            """
        )

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
