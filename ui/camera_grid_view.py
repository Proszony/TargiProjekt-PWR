from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

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


class CameraPanel(QFrame):
    def __init__(self, state: CameraPanelState) -> None:
        super().__init__()
        self.state = state
        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("cameraPanel")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            "#cameraPanel { border: 1px solid #334155; border-radius: 6px; background: #0f1720; }"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self.header_label = QLabel()
        self.header_label.setStyleSheet("font-weight: 600; color: #e2e8f0;")
        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: #94a3b8;")
        self.image_label = QLabel("Waiting for stream")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 220)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background: #020617; color: #cbd5e1;")

        root.addWidget(self.header_label)
        root.addWidget(self.info_label)
        root.addWidget(self.image_label, 1)
        self.refresh()

    def refresh(self) -> None:
        self.header_label.setText(self.state.name)
        self.header_label.setStyleSheet(
            f"font-weight: 600; color: {self.state.color};"
        )
        sync_bits = [self.state.runtime_mode, f"FPS: {self.state.fps:.1f}"]
        if self.state.media_time_s is not None:
            sync_bits.append(f"media t: {self.state.media_time_s:.2f}s")
            sync_bits.append(f"drift: {self.state.sync_drift_s:+.3f}s")
        if self.state.dropped_frames:
            sync_bits.append(f"drops: {self.state.dropped_frames}")
        sync_bits.append(f"sync: {'waiting' if self.state.sync_waiting else 'ok'}")
        if self.state.source_fps is not None:
            sync_bits.append(f"src fps: {self.state.source_fps:.1f}")
        if self.state.processing_latency_s is not None:
            sync_bits.append(f"proc: {self.state.processing_latency_s:.2f}s")
        self.info_label.setText(f"{self.state.status_text} | " + " | ".join(sync_bits))
        if self.state.image is None:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Waiting for stream")
            return
        scaled = QPixmap.fromImage(self.state.image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setText("")

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self.state.image is not None:
            self.refresh()


class CameraGridView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._panels: dict[str, CameraPanel] = {}
        self._states: dict[str, CameraPanelState] = {}
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._last_row_count = 0
        self._last_column_count = 0

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
            )
        self._states = states
        self._rebuild_grid()

    def update_frame(self, camera_id: str, image: QImage) -> None:
        state = self._states.get(camera_id)
        panel = self._panels.get(camera_id)
        if state is None or panel is None:
            return
        state.image = image
        panel.refresh()

    def update_status(self, camera_id: str, status_text: str) -> None:
        state = self._states.get(camera_id)
        panel = self._panels.get(camera_id)
        if state is None or panel is None:
            return
        state.status_text = status_text
        panel.refresh()

    def update_fps(self, camera_id: str, fps: float) -> None:
        state = self._states.get(camera_id)
        panel = self._panels.get(camera_id)
        if state is None or panel is None:
            return
        state.fps = fps
        panel.refresh()

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
        panel = self._panels.get(camera_id)
        if state is None or panel is None:
            return
        state.media_time_s = media_time_s
        state.sync_drift_s = sync_drift_s
        state.dropped_frames = dropped_frames
        state.sync_waiting = missing
        state.source_fps = source_fps
        state.processing_latency_s = processing_latency_s
        panel.refresh()

    def _rebuild_grid(self) -> None:
        for row in range(max(self._last_row_count, self._layout.rowCount())):
            self._layout.setRowStretch(row, 0)
            self._layout.setRowMinimumHeight(row, 0)
        for column in range(max(self._last_column_count, self._layout.columnCount())):
            self._layout.setColumnStretch(column, 0)
            self._layout.setColumnMinimumWidth(column, 0)
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._panels.clear()

        ordered_states = list(self._states.values())
        if len(ordered_states) == 1:
            panel = CameraPanel(ordered_states[0])
            self._layout.addWidget(panel, 0, 0)
            self._layout.setRowStretch(0, 1)
            self._layout.setColumnStretch(0, 1)
            self._panels[ordered_states[0].camera_id] = panel
            self._last_row_count = 1
            self._last_column_count = 1
            return
        columns = self._grid_columns_for_count(len(ordered_states))
        for index, state in enumerate(ordered_states):
            panel = CameraPanel(state)
            row = index // columns
            column = index % columns
            self._layout.addWidget(panel, row, column)
            self._panels[state.camera_id] = panel

        rows = max(1, (len(ordered_states) + columns - 1) // columns)
        for row in range(rows):
            self._layout.setRowStretch(row, 1)
        for column in range(columns):
            self._layout.setColumnStretch(column, 1)
        self._last_row_count = rows
        self._last_column_count = columns

    @staticmethod
    def _grid_columns_for_count(count: int) -> int:
        if count <= 2:
            return 1
        return 2
