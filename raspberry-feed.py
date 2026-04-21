import sys
import time
from typing import Optional

import av
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

STREAM_OPTIONS = {
    "fflags": "nobuffer",
    "flags": "low_delay",
    "strict": "experimental",
    "analyzeduration": "0",
}

STREAM_OPEN_TIMEOUT_S = 1.0
STREAM_READ_TIMEOUT_S = 1.0
DEFAULT_BIND_ADDRESS = "0.0.0.0"
DEFAULT_PORT = 5000
PLACEHOLDER_TEXT = f"Waiting for stream on UDP port {DEFAULT_PORT}"


def build_stream_url(bind_address: str, port: int) -> str:
    return f"udp://{bind_address}:{port}"


class VideoReceiverWorker(QObject):
    frame_ready = Signal(QImage)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    started_listening = Signal()
    stopped_listening = Signal()

    def __init__(self, bind_address: str, port: int) -> None:
        super().__init__()
        self.bind_address = bind_address
        self.port = port
        self._running = False
        self._container: Optional[av.container.InputContainer] = None

    @Slot()
    def run(self) -> None:
        url = build_stream_url(self.bind_address, self.port)
        self._running = True
        self.status_changed.emit(f"Listening on {url}")
        self.started_listening.emit()

        while self._running:
            try:
                self._container = av.open(
                    url,
                    options=STREAM_OPTIONS,
                    timeout=(STREAM_OPEN_TIMEOUT_S, STREAM_READ_TIMEOUT_S),
                )
                self.status_changed.emit("Connected")

                for frame in self._container.decode(video=0):
                    if not self._running:
                        break

                    rgb_frame = frame.to_ndarray(format="rgb24")
                    height, width, channels = rgb_frame.shape
                    bytes_per_line = channels * width
                    image = QImage(
                        rgb_frame.data,
                        width,
                        height,
                        bytes_per_line,
                        QImage.Format_RGB888,
                    ).copy()
                    self.frame_ready.emit(image)

                if self._running:
                    self.status_changed.emit("Stream lost, reconnecting...")
            except Exception as exc:
                if not self._running:
                    break
                self.status_changed.emit("Error opening/decoding stream; retrying...")
                self.error_occurred.emit(str(exc))
            finally:
                if self._container is not None:
                    try:
                        self._container.close()
                    except Exception:
                        pass
                    self._container = None

            if self._running:
                time.sleep(1.0)

        self.status_changed.emit("Stopped")
        self.stopped_listening.emit()

    @Slot()
    def stop(self) -> None:
        self._running = False
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            finally:
                self._container = None


class VideoDisplayLabel(QLabel):
    def __init__(self) -> None:
        super().__init__(PLACEHOLDER_TEXT)
        self._current_pixmap: Optional[QPixmap] = None
        self._fit_to_window = True
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet(
            "QLabel { background-color: #101418; color: #d8dee9; border: 1px solid #28313b; }"
        )

    def set_fit_to_window(self, enabled: bool) -> None:
        self._fit_to_window = enabled
        self._refresh_pixmap()

    def set_placeholder_text(self, text: str) -> None:
        if self._current_pixmap is None:
            self.setText(text)

    def clear_frame(self, text: str = PLACEHOLDER_TEXT) -> None:
        self._current_pixmap = None
        self.clear()
        self.setText(text)

    def has_frame(self) -> bool:
        return self._current_pixmap is not None

    def set_frame(self, image: QImage) -> None:
        self._current_pixmap = QPixmap.fromImage(image)
        self._refresh_pixmap()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._current_pixmap is None:
            return

        self.clear()
        if self._fit_to_window:
            scaled = self._current_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.setPixmap(scaled)
        else:
            self.setPixmap(self._current_pixmap)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.worker: Optional[VideoReceiverWorker] = None
        self.worker_thread: Optional[QThread] = None
        self.last_error: Optional[str] = None
        self._build_ui()
        self._connect_signals()
        self.update_status("Idle")

    def _build_ui(self) -> None:
        self.setWindowTitle("UDP Video Receiver")
        self.resize(960, 640)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        form_layout = QFormLayout()
        button_layout = QHBoxLayout()

        self.bind_address_input = QLineEdit(DEFAULT_BIND_ADDRESS)
        self.port_input = QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(DEFAULT_PORT)

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.fullscreen_button = QPushButton("Fullscreen")
        self.fit_to_window_checkbox = QCheckBox("Fit to window")
        self.fit_to_window_checkbox.setChecked(True)

        form_layout.addRow("Bind address", self.bind_address_input)
        form_layout.addRow("Port", self.port_input)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.fullscreen_button)
        button_layout.addWidget(self.fit_to_window_checkbox)
        button_layout.addStretch(1)

        form_container = QWidget()
        form_container.setLayout(form_layout)
        button_container = QWidget()
        button_container.setLayout(button_layout)

        control_layout.addWidget(form_container, 1)
        control_layout.addWidget(button_container, 2)

        self.video_label = VideoDisplayLabel()

        root_layout.addLayout(control_layout)
        root_layout.addWidget(self.video_label, 1)
        central_widget.setLayout(root_layout)

        exit_fullscreen_action = QAction(self)
        exit_fullscreen_action.setShortcut(Qt.Key_Escape)
        exit_fullscreen_action.triggered.connect(self._exit_fullscreen)
        self.addAction(exit_fullscreen_action)

        toggle_fullscreen_action = QAction(self)
        toggle_fullscreen_action.setShortcut(Qt.Key_F11)
        toggle_fullscreen_action.triggered.connect(self.toggle_fullscreen)
        self.addAction(toggle_fullscreen_action)

        self.statusBar().showMessage("Idle")

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.fit_to_window_checkbox.toggled.connect(self.video_label.set_fit_to_window)

    @Slot()
    def start_stream(self) -> None:
        if self.worker is not None or self.worker_thread is not None:
            return

        bind_address = self.bind_address_input.text().strip()
        if not bind_address:
            QMessageBox.warning(self, "Invalid Address", "Bind address cannot be empty.")
            return

        port = self.port_input.value()

        self.worker_thread = QThread(self)
        self.worker = VideoReceiverWorker(bind_address, port)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.update_video_frame)
        self.worker.status_changed.connect(self.update_status)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.stopped_listening.connect(self.worker_thread.quit)
        self.worker.stopped_listening.connect(self._mark_stopped_from_worker)
        self.worker_thread.finished.connect(self._cleanup_worker_thread)

        self.bind_address_input.setEnabled(False)
        self.port_input.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.video_label.set_placeholder_text(
            f"Waiting for stream on UDP port {port}"
        )
        self.last_error = None

        self.worker_thread.start()

    @Slot()
    def stop_stream(self) -> None:
        if self.worker is not None:
            self.worker.stop()

        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(3000)

        if self.worker is None and self.worker_thread is None:
            self._set_controls_running(False)
            if not self.video_label.has_frame():
                self.video_label.clear_frame(
                    f"Waiting for stream on UDP port {self.port_input.value()}"
                )
            self.update_status("Stopped")

    @Slot()
    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    @Slot()
    def update_video_frame(self, image: QImage) -> None:
        self.video_label.set_frame(image)
        self.update_status("Connected")

    @Slot(str)
    def update_status(self, text: str) -> None:
        self.statusBar().showMessage(text)
        if text in {"Idle", "Stopped"}:
            self.video_label.set_placeholder_text(
                f"Waiting for stream on UDP port {self.port_input.value()}"
            )

    @Slot(str)
    def show_error(self, message: str) -> None:
        if message != self.last_error:
            print(f"Receiver error: {message}", file=sys.stderr)
        self.last_error = message
        self.statusBar().showMessage(f"Error: {message}")

    @Slot()
    def _cleanup_worker_thread(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
            self.worker_thread = None

        self._set_controls_running(False)

    @Slot()
    def _mark_stopped_from_worker(self) -> None:
        if not self.video_label.has_frame():
            self.video_label.clear_frame(
                f"Waiting for stream on UDP port {self.port_input.value()}"
            )

    @Slot()
    def _exit_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()

    def _set_controls_running(self, running: bool) -> None:
        self.bind_address_input.setEnabled(not running)
        self.port_input.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self.stop_stream()
        event.accept()


def main() -> int:
    try:
        app = QApplication(sys.argv)
    except Exception as exc:
        print(f"Failed to start Qt application: {exc}", file=sys.stderr)
        return 1

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
