from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeyEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.calibration import compute_homography, homography_to_list
from core.models import CalibrationPair, VenueMapConfig
from ui.canvas import ImageCanvas
from ui.map_view import MapView


class CalibrationDialog(QDialog):
    def __init__(
        self,
        frame: QImage,
        venue_map: VenueMapConfig,
        existing_pairs: list[CalibrationPair] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Camera calibration")
        self.resize(1550, 900)
        self.setModal(True)
        self._pending_image_point: tuple[float, float] | None = None
        self._pairs: list[CalibrationPair] = list(existing_pairs or [])
        self._homography: list[list[float]] | None = None
        self._selected_pair_index: int | None = None

        self.camera_canvas = ImageCanvas("Reference frame required")
        self.camera_canvas.set_image(frame)
        self.camera_canvas.set_drag_enabled(True)
        self.camera_canvas.set_delete_enabled(True)
        self.camera_canvas.set_add_enabled(True)

        self.map_view = MapView()
        self.map_view.set_venue_map(venue_map)
        self.map_view.set_mode("pick_points")

        self.instructions_label = QLabel(
            "1. Click a point on the camera image.\n"
            "2. Click the matching point on the map.\n"
            "3. Drag points to correct misclicks.\n"
            "Right click deletes a point. Esc cancels the pending pair."
        )
        self.instructions_label.setWordWrap(True)
        self.status_label = QLabel()
        self.pairs_table = QTableWidget(0, 3)
        self.compute_button = QPushButton("Compute homography")
        self.undo_button = QPushButton("Undo last pair")
        self.clear_button = QPushButton("Clear all")
        self.close_button = QPushButton("Close")

        self._build_ui()
        self._connect_signals()
        self._refresh_all()

    @property
    def calibration_pairs(self) -> list[CalibrationPair]:
        return list(self._pairs)

    @property
    def homography(self) -> list[list[float]] | None:
        return self._homography

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)

        self.pairs_table.setHorizontalHeaderLabels(["#", "Image", "World"])
        self.pairs_table.horizontalHeader().setStretchLastSection(True)
        self.pairs_table.verticalHeader().setVisible(False)
        self.pairs_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.pairs_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.pairs_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.pairs_table.setColumnWidth(0, 40)
        self.pairs_table.setColumnWidth(1, 120)

        sidebar_layout.addWidget(self.instructions_label)
        sidebar_layout.addWidget(self.status_label)
        sidebar_layout.addWidget(self.pairs_table, 1)
        sidebar_layout.addWidget(self.compute_button)
        sidebar_layout.addWidget(self.undo_button)
        sidebar_layout.addWidget(self.clear_button)
        sidebar_layout.addWidget(self.close_button)

        splitter.addWidget(self.camera_canvas)
        splitter.addWidget(self.map_view)
        splitter.addWidget(sidebar)
        splitter.setSizes([650, 650, 280])
        sidebar.setMinimumWidth(280)
        sidebar.setMaximumWidth(360)

        root = QHBoxLayout()
        root.addWidget(splitter)
        self.setLayout(root)

    def _connect_signals(self) -> None:
        self.camera_canvas.point_added.connect(self._on_image_point_added)
        self.camera_canvas.point_moved.connect(self._on_image_point_moved)
        self.camera_canvas.point_deleted.connect(self._on_image_point_deleted)
        self.camera_canvas.selected_point_changed.connect(self._on_pair_selected)
        self.camera_canvas.escape_pressed.connect(self._clear_pending_pair)

        self.map_view.world_point_clicked.connect(self._on_world_point_added)
        self.map_view.world_point_moved.connect(self._on_world_point_moved)
        self.map_view.world_point_deleted.connect(self._on_world_point_deleted)
        self.map_view.calibration_point_selected.connect(self._on_pair_selected)
        self.map_view.escape_pressed.connect(self._clear_pending_pair)

        self.pairs_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.compute_button.clicked.connect(self._compute_homography)
        self.undo_button.clicked.connect(self._undo_last_pair)
        self.clear_button.clicked.connect(self._clear_pairs)
        self.close_button.clicked.connect(self.reject)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == Qt.Key_Escape:
            self._clear_pending_pair()
            event.accept()
            return
        super().keyPressEvent(event)

    def _on_image_point_added(self, x: float, y: float) -> None:
        self._pending_image_point = (x, y)
        self._selected_pair_index = None
        self._refresh_all()

    def _on_world_point_added(self, x: float, y: float) -> None:
        if self._pending_image_point is None:
            self._set_status("Select a point on the camera image first.")
            return
        self._pairs.append(CalibrationPair(image_point=self._pending_image_point, world_point=(x, y)))
        self._pending_image_point = None
        self._selected_pair_index = len(self._pairs) - 1
        self._refresh_all()

    def _on_image_point_moved(self, index: int, x: float, y: float) -> None:
        self._pairs[index].image_point = (x, y)
        self._selected_pair_index = index
        self._refresh_all()

    def _on_world_point_moved(self, index: int, x: float, y: float) -> None:
        self._pairs[index].world_point = (x, y)
        self._selected_pair_index = index
        self._refresh_all()

    def _on_image_point_deleted(self, index: int) -> None:
        del self._pairs[index]
        self._selected_pair_index = None
        self._refresh_all()

    def _on_world_point_deleted(self, index: int) -> None:
        del self._pairs[index]
        self._selected_pair_index = None
        self._refresh_all()

    def _on_pair_selected(self, index: object) -> None:
        self._selected_pair_index = index if isinstance(index, int) else None
        self._refresh_selection()

    def _on_table_selection_changed(self) -> None:
        selected_ranges = self.pairs_table.selectedRanges()
        if not selected_ranges:
            self._selected_pair_index = None
        else:
            self._selected_pair_index = selected_ranges[0].topRow()
        self._refresh_selection()

    def _undo_last_pair(self) -> None:
        if not self._pairs:
            return
        self._pairs.pop()
        self._selected_pair_index = None
        self._refresh_all()

    def _clear_pairs(self) -> None:
        self._pairs.clear()
        self._pending_image_point = None
        self._selected_pair_index = None
        self._homography = None
        self._refresh_all()

    def _clear_pending_pair(self) -> None:
        if self._pending_image_point is None:
            self._set_status("Selection cleared.")
        else:
            self._pending_image_point = None
            self._set_status("Pending image point cleared.")
        self.camera_canvas.clear_pending_point()
        self.camera_canvas.set_selected_point(None)
        self.map_view.set_selected_calibration_point(None)

    def _refresh_all(self) -> None:
        self.camera_canvas.set_editable_points([pair.image_point for pair in self._pairs])
        self.camera_canvas.set_pending_point(self._pending_image_point)
        self.map_view.set_calibration_points([pair.world_point for pair in self._pairs])
        self._refresh_table()
        self._refresh_selection()
        if self._pending_image_point is not None:
            self._set_status("Image point selected. Click the matching map point.")
        else:
            self._set_status(f"{len(self._pairs)} calibration pairs ready.")

    def _refresh_table(self) -> None:
        self.pairs_table.setRowCount(len(self._pairs))
        for row, pair in enumerate(self._pairs):
            self.pairs_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.pairs_table.setItem(
                row,
                1,
                QTableWidgetItem(f"({pair.image_point[0]:.1f}, {pair.image_point[1]:.1f})"),
            )
            self.pairs_table.setItem(
                row,
                2,
                QTableWidgetItem(f"({pair.world_point[0]:.2f}, {pair.world_point[1]:.2f})"),
            )

    def _refresh_selection(self) -> None:
        if self._selected_pair_index is None:
            self.camera_canvas.set_selected_point(None)
            self.map_view.set_selected_calibration_point(None)
            self.pairs_table.clearSelection()
            return
        if self._selected_pair_index >= len(self._pairs):
            self._selected_pair_index = None
            self._refresh_selection()
            return

        self.camera_canvas.set_selected_point(self._selected_pair_index)
        self.map_view.set_selected_calibration_point(self._selected_pair_index)
        self.pairs_table.blockSignals(True)
        self.pairs_table.selectRow(self._selected_pair_index)
        self.pairs_table.blockSignals(False)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _compute_homography(self) -> None:
        matrix = compute_homography(
            [pair.image_point for pair in self._pairs],
            [pair.world_point for pair in self._pairs],
        )
        if matrix is None:
            QMessageBox.warning(self, "Not enough points", "At least 4 valid pairs are required.")
            return
        self._homography = homography_to_list(matrix)
        self.accept()
