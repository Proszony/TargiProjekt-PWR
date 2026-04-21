#!/usr/bin/env python3
"""
UDP Video Receiver with Real‑Time YOLOv8 Head Detection (Optimized)
===================================================================
- Receives UDP H.264 stream (e.g., from Raspberry Pi camera)
- Runs YOLOv8n inference in a separate thread
- Resizes frames for faster inference, scales boxes back
- Displays FPS and detection statistics
- Non‑blocking: video remains fluid even when inference is slow

Dependencies:
    PySide6, av, opencv-python-headless, ultralytics, numpy
"""

import sys
import time
import queue
import threading
from pathlib import Path
from typing import Optional, Tuple

import av
import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Stream configuration
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

# ----------------------------------------------------------------------
# YOLO detection defaults
DEFAULT_MODEL_PATH = "models/yolov8n.pt"          # Auto‑downloads on first run
                                           # Models: yolov8n.pt or yolov8s.pt (nano 6,5MB or small 22MB)
DEFAULT_CONFIDENCE = 0.25
HEAD_RATIO = 0.30                          # Top 30% of person box
BOX_COLOR = (0, 255, 0)                    # Green in BGR
BOX_THICKNESS = 2
DEFAULT_INFERENCE_SIZE = 320               # Width/height for inference (square resize)


def build_stream_url(bind_address: str, port: int) -> str:
    return f"udp://{bind_address}:{port}"


class DetectionThread(threading.Thread):
    """
    Background thread that continuously processes frames from a queue,
    runs YOLO detection, and calls a callback with annotated frame + stats.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float,
        inference_size: int,
        callback,
    ):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.confidence = confidence
        self.inference_size = inference_size
        self.callback = callback  # function(frame, heads_count, inference_ms)

        self._running = True
        self._queue = queue.Queue(maxsize=2)  # Keep only latest frame
        self._model: Optional[YOLO] = None

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        """Put a new frame into the queue (non‑blocking, drops old if full)."""
        try:
            # Replace the latest frame if queue is full
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put_nowait(frame_bgr)
        except queue.Full:
            pass

    def stop(self) -> None:
        self._running = False
        # Put a sentinel to unblock the queue
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def run(self) -> None:
        # Load model (once)
        self._model = YOLO(self.model_path)
        # Warm‑up
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self._model(dummy, classes=[0], conf=self.confidence, verbose=False)

        while self._running:
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if frame is None or not self._running:
                break

            # Resize for inference (maintain aspect ratio, then pad to square)
            h, w = frame.shape[:2]
            scale = self.inference_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Create square canvas
            square = np.zeros(
                (self.inference_size, self.inference_size, 3), dtype=np.uint8
            )
            y_off = (self.inference_size - new_h) // 2
            x_off = (self.inference_size - new_w) // 2
            square[y_off:y_off+new_h, x_off:x_off+new_w] = resized

            # Run YOLO inference
            start_t = time.perf_counter()
            results = self._model(
                square,
                classes=[0],
                conf=self.confidence,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - start_t) * 1000.0

            # Draw head boxes on original frame
            annotated = frame.copy()
            heads_count = 0
            for box in results[0].boxes:
                # Box coordinates are relative to the square image
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                # Adjust for padding and scale back to original dimensions
                bx1 = (bx1 - x_off) / scale
                by1 = (by1 - y_off) / scale
                bx2 = (bx2 - x_off) / scale
                by2 = (by2 - y_off) / scale

                # Clamp to image bounds
                x1 = max(0, int(bx1))
                y1 = max(0, int(by1))
                x2 = min(w, int(bx2))
                y2 = min(h, int(by2))

                head_height = max(1, int((y2 - y1) * HEAD_RATIO))
                head_box = (x1, y1, x2, y1 + head_height)
                cv2.rectangle(
                    annotated,
                    (head_box[0], head_box[1]),
                    (head_box[2], head_box[3]),
                    BOX_COLOR,
                    BOX_THICKNESS,
                )
                heads_count += 1

            # Send annotated frame back via callback
            self.callback(annotated, heads_count, inference_ms)

        self._model = None


class VideoReceiverWorker(QObject):
    """Worker that receives UDP stream and manages detection thread."""

    frame_ready = Signal(QImage)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    started_listening = Signal()
    stopped_listening = Signal()
    detection_stats = Signal(int, float)   # heads_count, inference_time_ms
    fps_update = Signal(float)             # current display FPS

    def __init__(self, bind_address: str, port: int) -> None:
        super().__init__()
        self.bind_address = bind_address
        self.port = port
        self._running = False
        self._container: Optional[av.container.InputContainer] = None

        # Detection settings
        self.detection_enabled = True
        self.confidence = DEFAULT_CONFIDENCE
        self.inference_size = DEFAULT_INFERENCE_SIZE
        self.model_path = DEFAULT_MODEL_PATH

        self._detection_thread: Optional[DetectionThread] = None
        self._latest_annotated_frame: Optional[np.ndarray] = None
        self._last_heads_count = 0
        self._last_inference_ms = 0.0
        self._frame_lock = threading.Lock()

        # FPS calculation
        self._fps_alpha = 0.1  # smoothing factor
        self._smoothed_fps = 0.0

    def _start_detection_thread(self) -> None:
        if self._detection_thread is not None:
            return
        self._detection_thread = DetectionThread(
            model_path=self.model_path,
            confidence=self.confidence,
            inference_size=self.inference_size,
            callback=self._on_detection_done,
        )
        self._detection_thread.start()
        self.status_changed.emit("Detection thread started")

    def _stop_detection_thread(self) -> None:
        if self._detection_thread is not None:
            self._detection_thread.stop()
            self._detection_thread.join(timeout=2.0)
            self._detection_thread = None

    def _on_detection_done(self, annotated_bgr: np.ndarray, heads: int, ms: float) -> None:
        """Called from detection thread with results."""
        with self._frame_lock:
            self._latest_annotated_frame = annotated_bgr
            self._last_heads_count = heads
            self._last_inference_ms = ms

    @Slot()
    def run(self) -> None:
        url = build_stream_url(self.bind_address, self.port)
        self._running = True
        self.status_changed.emit(f"Listening on {url}")
        self.started_listening.emit()

        if self.detection_enabled:
            self._start_detection_thread()

        last_frame_time = time.perf_counter()
        frame_times = []

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

                    # Calculate FPS
                    now = time.perf_counter()
                    delta = now - last_frame_time
                    last_frame_time = now
                    instant_fps = 1.0 / delta if delta > 0 else 0.0
                    self._smoothed_fps = (
                        self._fps_alpha * instant_fps +
                        (1 - self._fps_alpha) * self._smoothed_fps
                    )
                    self.fps_update.emit(self._smoothed_fps)

                    # Convert to BGR for OpenCV
                    rgb_frame = frame.to_ndarray(format="rgb24")
                    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                    # Send to detection thread if enabled
                    if self.detection_enabled and self._detection_thread is not None:
                        self._detection_thread.push_frame(bgr_frame)

                    # Choose which frame to display
                    with self._frame_lock:
                        if self._latest_annotated_frame is not None:
                            display_bgr = self._latest_annotated_frame.copy()
                            heads = self._last_heads_count
                            inf_ms = self._last_inference_ms
                        else:
                            display_bgr = bgr_frame
                            heads = 0
                            inf_ms = 0.0

                    # Emit detection stats
                    self.detection_stats.emit(heads, inf_ms)

                    # Convert back to RGB for QImage
                    display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
                    height, width, channels = display_rgb.shape
                    bytes_per_line = channels * width
                    image = QImage(
                        display_rgb.data,
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

        self._stop_detection_thread()
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

    @Slot(bool)
    def set_detection_enabled(self, enabled: bool) -> None:
        self.detection_enabled = enabled
        if enabled and self._running:
            self._start_detection_thread()
        elif not enabled:
            self._stop_detection_thread()
            with self._frame_lock:
                self._latest_annotated_frame = None

    @Slot(float)
    def set_confidence(self, conf: float) -> None:
        self.confidence = conf
        if self._detection_thread is not None:
            self._detection_thread.confidence = conf

    @Slot(int)
    def set_inference_size(self, size: int) -> None:
        self.inference_size = size
        # Requires restart of detection thread to apply new size
        if self._detection_thread is not None:
            self._stop_detection_thread()
            if self._running:
                self._start_detection_thread()


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

    def resizeEvent(self, event) -> None:
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
        self.setWindowTitle("UDP Video Receiver with Real‑Time Head Detection")
        self.resize(960, 760)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        form_layout = QFormLayout()
        button_layout = QHBoxLayout()

        # Connection settings
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

        # Detection controls
        self.enable_detection_checkbox = QCheckBox("Enable Head Detection")
        self.enable_detection_checkbox.setChecked(True)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.01, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(DEFAULT_CONFIDENCE)

        self.inference_size_combo = QComboBox()
        self.inference_size_combo.addItems(["160", "224", "320", "416", "640"])
        self.inference_size_combo.setCurrentText(str(DEFAULT_INFERENCE_SIZE))
        self.inference_size_combo.setToolTip(
            "Smaller = faster inference, lower accuracy"
        )

        # Statistics labels
        self.stats_label = QLabel("Detected heads: 0 | Inference: 0 ms")
        self.stats_label.setStyleSheet("QLabel { font-weight: bold; }")
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("QLabel { font-weight: bold; }")

        # Layout assembly
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

        # Detection controls row
        detection_layout = QHBoxLayout()
        detection_layout.addWidget(self.enable_detection_checkbox)
        detection_layout.addWidget(QLabel("Confidence:"))
        detection_layout.addWidget(self.confidence_spin)
        detection_layout.addWidget(QLabel("Inference size:"))
        detection_layout.addWidget(self.inference_size_combo)
        detection_layout.addStretch(1)
        detection_layout.addWidget(self.stats_label)
        detection_layout.addSpacing(20)
        detection_layout.addWidget(self.fps_label)

        self.video_label = VideoDisplayLabel()

        root_layout.addLayout(control_layout)
        root_layout.addLayout(detection_layout)
        root_layout.addWidget(self.video_label, 1)
        central_widget.setLayout(root_layout)

        # Shortcuts
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

        # Detection control signals will be connected when worker is created
        self.enable_detection_checkbox.toggled.connect(self._on_detection_toggled)
        self.confidence_spin.valueChanged.connect(self._on_confidence_changed)
        self.inference_size_combo.currentTextChanged.connect(self._on_inference_size_changed)

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

        # Connect worker signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.update_video_frame)
        self.worker.status_changed.connect(self.update_status)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.stopped_listening.connect(self.worker_thread.quit)
        self.worker.stopped_listening.connect(self._mark_stopped_from_worker)
        self.worker.detection_stats.connect(self.update_stats)
        self.worker.fps_update.connect(self.update_fps)
        self.worker_thread.finished.connect(self._cleanup_worker_thread)

        # Apply current detection settings
        self.worker.set_detection_enabled(self.enable_detection_checkbox.isChecked())
        self.worker.set_confidence(self.confidence_spin.value())
        self.worker.set_inference_size(int(self.inference_size_combo.currentText()))

        # Connect live settings updates (cross‑thread slots)
        self.enable_detection_checkbox.toggled.connect(self.worker.set_detection_enabled)
        self.confidence_spin.valueChanged.connect(self.worker.set_confidence)
        # inference size change handled via _on_inference_size_changed

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

    @Slot(int, float)
    def update_stats(self, heads: int, inference_ms: float) -> None:
        self.stats_label.setText(f"Detected heads: {heads} | Inference: {inference_ms:.1f} ms")

    @Slot(float)
    def update_fps(self, fps: float) -> None:
        self.fps_label.setText(f"FPS: {fps:.1f}")

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
        self.stats_label.setText("Detected heads: 0 | Inference: 0 ms")
        self.fps_label.setText("FPS: 0.0")

    @Slot()
    def _exit_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()

    def _set_controls_running(self, running: bool) -> None:
        self.bind_address_input.setEnabled(not running)
        self.port_input.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    # Slots for detection settings
    @Slot(bool)
    def _on_detection_toggled(self, enabled: bool) -> None:
        if self.worker is not None:
            self.worker.set_detection_enabled(enabled)

    @Slot(float)
    def _on_confidence_changed(self, value: float) -> None:
        if self.worker is not None:
            self.worker.set_confidence(value)

    @Slot(str)
    def _on_inference_size_changed(self, text: str) -> None:
        if self.worker is not None:
            size = int(text)
            self.worker.set_inference_size(size)

    def closeEvent(self, event: QCloseEvent) -> None:
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