from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.models import CameraConfig
from ui.camera_colors import camera_color


@dataclass(slots=True)
class CameraPanelState:
    camera_id: str
    name: str
    color: str
    runtime_mode: str = "local"
    status_text: str = "Idle"
    fps: float = 0.0
    media_time_s: float | None = None
    sync_drift_s: float = 0.0
    dropped_frames: int = 0
    sync_waiting: bool = False
    source_fps: float | None = None
    processing_latency_s: float | None = None
    image: QImage | None = None
    enabled: bool = True


class CameraSelectorButton(QPushButton):
    def __init__(self, state: CameraPanelState) -> None:
        super().__init__()
        self.state = state
        self.setObjectName("cameraSelector")
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(62)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setProperty("cameraColor", state.color)
        self.setProperty("cameraEnabled", state.enabled)
        self.refresh()

    def refresh(self) -> None:
        status = self.state.status_text
        if self.state.sync_waiting:
            status = "Waiting for sync"
        compact_status = self._compact_status(status)
        mode = self.state.runtime_mode.lower()
        if not self.state.enabled:
            compact_status = "disabled"
        self.setText(f"{self.state.name} [{mode}]\n{compact_status}")
        self.setToolTip(
            f"{self.state.name}\n"
            f"Mode: {self.state.runtime_mode}\n"
            f"Status: {self.state.status_text}\n"
            f"FPS: {self.state.fps:.1f}\n"
            f"Drift: {self.state.sync_drift_s:+.3f}s"
        )

    @staticmethod
    def _compact_status(status: str) -> str:
        normalized = status.strip().lower()
        if "waiting" in normalized:
            return "waiting"
        if normalized.startswith("error"):
            return "error"
        return status.strip() or "idle"


class CameraStageView(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self._state: CameraPanelState | None = None
        self.setObjectName("cameraStage")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        heading_row = QHBoxLayout()
        heading_row.setSpacing(12)

        title_wrap = QVBoxLayout()
        title_wrap.setSpacing(4)
        self.title_label = QLabel("No camera selected")
        self.title_label.setObjectName("cameraStageTitle")
        self.subtitle_label = QLabel("Add and enable cameras to begin monitoring.")
        self.subtitle_label.setObjectName("cameraStageSubtitle")
        title_wrap.addWidget(self.title_label)
        title_wrap.addWidget(self.subtitle_label)

        self.badge_label = QLabel("Idle")
        self.badge_label.setObjectName("cameraStageBadge")
        self.badge_label.setAlignment(Qt.AlignCenter)
        self.badge_label.setMinimumWidth(130)

        heading_row.addLayout(title_wrap, 1)
        heading_row.addWidget(self.badge_label, 0, Qt.AlignTop)

        self.image_label = QLabel("No live frame yet")
        self.image_label.setObjectName("cameraStageImage")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(480, 320)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.meta_label = QLabel("Camera telemetry appears here once a feed is selected.")
        self.meta_label.setObjectName("cameraStageMeta")
        self.meta_label.setWordWrap(True)

        root.addLayout(heading_row)
        root.addWidget(self.image_label, 1)
        root.addWidget(self.meta_label)

    def set_state(self, state: CameraPanelState | None) -> None:
        self._state = state
        self.refresh()

    def refresh(self) -> None:
        state = self._state
        if state is None:
            self.title_label.setText("No camera selected")
            self.subtitle_label.setText("Add and enable cameras to begin monitoring.")
            self.badge_label.setText("Idle")
            self.badge_label.setStyleSheet("")
            self.meta_label.setText("Camera telemetry appears here once a feed is selected.")
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("No live frame yet")
            return

        self.title_label.setText(state.name)
        subtitle = f"{state.runtime_mode.upper()} source"
        if state.media_time_s is not None:
            subtitle += f" | media {state.media_time_s:.2f}s"
        self.subtitle_label.setText(subtitle)
        self.badge_label.setText("Disabled" if not state.enabled else state.status_text or "Live")
        self.badge_label.setStyleSheet(
            f"background-color: {state.color}; color: #031017; border-radius: 12px; padding: 6px 12px; font-weight: 700;"
        )

        meta_bits = [
            f"Pipeline FPS {state.fps:.1f}",
            f"Sync drift {state.sync_drift_s:+.3f}s",
            f"Dropped {state.dropped_frames}",
            f"Sync {'waiting' if state.sync_waiting else 'stable'}",
        ]
        if state.source_fps is not None:
            meta_bits.append(f"Source FPS {state.source_fps:.1f}")
        if state.processing_latency_s is not None:
            meta_bits.append(f"Latency {state.processing_latency_s:.2f}s")
        self.meta_label.setText(" | ".join(meta_bits))

        if state.image is None:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Waiting for stream")
            return
        scaled = QPixmap.fromImage(state.image).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setText("")

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._state is not None and self._state.image is not None:
            self.refresh()


class CameraGridView(QWidget):
    camera_selected = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._buttons: dict[str, CameraSelectorButton] = {}
        self._states: dict[str, CameraPanelState] = {}
        self._selected_camera_id: str | None = None

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        rail = QFrame()
        rail.setObjectName("cameraRail")
        rail.setFixedWidth(190)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(8, 8, 8, 8)
        rail_layout.setSpacing(6)

        rail_title = QLabel("Cameras")
        rail_title.setObjectName("cameraRailTitle")

        self.selector_host = QWidget()
        self.selector_layout = QVBoxLayout(self.selector_host)
        self.selector_layout.setContentsMargins(0, 0, 0, 0)
        self.selector_layout.setSpacing(6)
        self.selector_layout.addStretch(1)

        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroller.setFrameShape(QFrame.NoFrame)
        scroller.setWidget(self.selector_host)

        rail_layout.addWidget(rail_title)
        rail_layout.addWidget(scroller, 1)

        self.stage = CameraStageView()

        root.addWidget(rail)
        root.addWidget(self.stage, 1)

    def set_cameras(self, cameras: list[CameraConfig]) -> None:
        states: dict[str, CameraPanelState] = {}
        for camera in sorted(cameras, key=lambda item: (item.display_order, item.camera_id)):
            existing = self._states.get(camera.camera_id)
            states[camera.camera_id] = CameraPanelState(
                camera_id=camera.camera_id,
                name=camera.name,
                color=camera_color(camera.display_order, camera.camera_id),
                runtime_mode=camera.runtime_mode,
                status_text=existing.status_text if existing else ("Disabled" if not camera.enabled else "Idle"),
                fps=existing.fps if existing else 0.0,
                media_time_s=existing.media_time_s if existing else None,
                sync_drift_s=existing.sync_drift_s if existing else 0.0,
                dropped_frames=existing.dropped_frames if existing else 0,
                sync_waiting=existing.sync_waiting if existing else False,
                source_fps=existing.source_fps if existing else None,
                processing_latency_s=existing.processing_latency_s if existing else None,
                image=existing.image if existing else None,
                enabled=camera.enabled,
            )
        self._states = states
        self._selected_camera_id = self._pick_selected_camera()
        self._rebuild_selector_list()
        self._refresh_stage()

    def selected_camera_id(self) -> str | None:
        return self._selected_camera_id

    def selected_camera_state(self) -> CameraPanelState | None:
        if self._selected_camera_id is None:
            return None
        return self._states.get(self._selected_camera_id)

    def set_selected_camera(self, camera_id: str) -> None:
        if camera_id not in self._states:
            return
        if camera_id == self._selected_camera_id:
            return
        self._selected_camera_id = camera_id
        self._refresh_selector_states()
        self._refresh_stage()
        self.camera_selected.emit(camera_id)

    def update_frame(self, camera_id: str, image: QImage) -> None:
        state = self._states.get(camera_id)
        button = self._buttons.get(camera_id)
        if state is None or button is None:
            return
        state.image = image
        button.refresh()
        if camera_id == self._selected_camera_id:
            self.stage.refresh()

    def update_status(self, camera_id: str, status_text: str) -> None:
        state = self._states.get(camera_id)
        button = self._buttons.get(camera_id)
        if state is None or button is None:
            return
        state.status_text = status_text
        button.refresh()
        if camera_id == self._selected_camera_id:
            self.stage.refresh()

    def update_fps(self, camera_id: str, fps: float) -> None:
        state = self._states.get(camera_id)
        button = self._buttons.get(camera_id)
        if state is None or button is None:
            return
        state.fps = fps
        button.refresh()
        if camera_id == self._selected_camera_id:
            self.stage.refresh()

    def update_sync_state(
        self,
        camera_id: str,
        *,
        media_time_s: float | None,
        sync_drift_s: float,
        dropped_frames: int,
        missing: bool,
        source_fps: float | None = None,
        processing_latency_s: float | None = None,
    ) -> None:
        state = self._states.get(camera_id)
        button = self._buttons.get(camera_id)
        if state is None or button is None:
            return
        state.media_time_s = media_time_s
        state.sync_drift_s = sync_drift_s
        state.dropped_frames = dropped_frames
        state.sync_waiting = missing
        state.source_fps = source_fps
        state.processing_latency_s = processing_latency_s
        button.refresh()
        if camera_id == self._selected_camera_id:
            self.stage.refresh()

    def _pick_selected_camera(self) -> str | None:
        if self._selected_camera_id in self._states:
            return self._selected_camera_id
        for state in self._states.values():
            if state.enabled:
                return state.camera_id
        return next(iter(self._states), None)

    def _rebuild_selector_list(self) -> None:
        while self.selector_layout.count():
            item = self.selector_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._buttons.clear()

        for state in self._states.values():
            button = CameraSelectorButton(state)
            button.clicked.connect(lambda checked=False, camera_id=state.camera_id: self.set_selected_camera(camera_id))
            self.selector_layout.addWidget(button)
            self._buttons[state.camera_id] = button
        self.selector_layout.addStretch(1)
        self._refresh_selector_states()

    def _refresh_selector_states(self) -> None:
        for camera_id, button in self._buttons.items():
            button.setChecked(camera_id == self._selected_camera_id)
            button.refresh()

    def _refresh_stage(self) -> None:
        self.stage.set_state(self.selected_camera_state())
