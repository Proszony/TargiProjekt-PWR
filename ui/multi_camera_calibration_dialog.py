from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from core.calibration import (
    compute_world_viewport,
    compute_homography_result,
    recompute_camera_coverage,
)
from core.coverage_mapping import default_coverage_polygon_image, propose_coverage_polygon_image
from core.models import CameraAnchorObservation, CameraConfig, CameraCoverageOverlay, Point, ProjectConfig, SharedAnchor
from ui.canvas import ImageCanvas
from ui.map_view import MapView


class MultiCameraCalibrationDialog(QDialog):
    calibration_applied = Signal(object)

    def __init__(
        self,
        project_config: ProjectConfig,
        camera_frames: dict[str, QImage],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Multi-camera calibration")
        self.resize(1400, 820)
        self._project = ProjectConfig.from_dict(project_config.to_dict())
        self._camera_frames = dict(camera_frames)
        self._pending_image_point: Point | None = None
        self._current_anchor_order: list[str] = []
        self._selected_anchor_id: str | None = None
        self._edit_mode = "anchors"
        self._build_ui()
        self._connect_signals()
        self._load_cameras()
        self._refresh_views()

    @property
    def project_config(self) -> ProjectConfig:
        return ProjectConfig.from_dict(self._project.to_dict())

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Anchors", "anchors")
        self.mode_combo.addItem("Coverage", "coverage")
        self.camera_status_label = QLabel()
        top_row.addWidget(QLabel("Camera"))
        top_row.addWidget(self.camera_combo)
        top_row.addWidget(QLabel("Mode"))
        top_row.addWidget(self.mode_combo)
        top_row.addWidget(self.camera_status_label, 1)

        body = QHBoxLayout()
        self.image_canvas = ImageCanvas("No frame captured for this camera")
        self.image_canvas.set_add_enabled(True)
        self.image_canvas.set_drag_enabled(True)
        self.image_canvas.set_delete_enabled(True)
        self.map_view = MapView()
        self.map_view.set_pick_points_mode(True)

        side = QVBoxLayout()
        self.instructions = QLabel(
            "1. Click a point in the camera image.\n"
            "2. Click an existing anchor on the map or empty map space to create one.\n"
            "3. Drag anchors or observations to refine them.\n"
            "4. Compute calibration for the selected camera."
        )
        self.instructions.setWordWrap(True)
        self.quality_label = QLabel("Calibration: not computed")
        self.quality_label.setWordWrap(True)
        self.anchor_list = QListWidget()
        self.compute_button = QPushButton("Compute selected camera")
        self.remove_observation_button = QPushButton("Remove selected observation")
        self.auto_detect_coverage_button = QPushButton("Auto-detect coverage")
        self.reset_coverage_button = QPushButton("Reset to fallback")
        self.accept_coverage_button = QPushButton("Accept coverage")

        side.addWidget(self.instructions)
        side.addWidget(self.quality_label)
        side.addWidget(self.anchor_list, 1)
        side.addWidget(self.compute_button)
        side.addWidget(self.remove_observation_button)
        side.addWidget(self.auto_detect_coverage_button)
        side.addWidget(self.reset_coverage_button)
        side.addWidget(self.accept_coverage_button)

        body.addWidget(self.image_canvas, 5)
        body.addWidget(self.map_view, 5)
        body.addLayout(side, 3)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        root.addLayout(top_row)
        root.addLayout(body, 1)
        root.addWidget(self.button_box)

    def _connect_signals(self) -> None:
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.image_canvas.point_added.connect(self._on_image_point_added)
        self.image_canvas.point_moved.connect(self._on_image_point_moved)
        self.image_canvas.point_deleted.connect(self._on_image_point_deleted)
        self.image_canvas.selected_point_changed.connect(self._on_image_selection_changed)
        self.image_canvas.escape_pressed.connect(self._clear_pending_point)
        self.map_view.world_point_clicked.connect(self._on_map_empty_clicked)
        self.map_view.calibration_point_selected.connect(self._on_anchor_selected)
        self.map_view.world_point_moved.connect(self._on_anchor_moved)
        self.map_view.world_point_deleted.connect(self._on_anchor_deleted)
        self.map_view.escape_pressed.connect(self._clear_pending_point)
        self.anchor_list.currentRowChanged.connect(self._on_anchor_list_row_changed)
        self.compute_button.clicked.connect(self._compute_selected_camera)
        self.remove_observation_button.clicked.connect(self._remove_selected_observation)
        self.auto_detect_coverage_button.clicked.connect(self._auto_detect_coverage)
        self.reset_coverage_button.clicked.connect(self._reset_coverage_to_fallback)
        self.accept_coverage_button.clicked.connect(self._accept_coverage)
        self.button_box.accepted.connect(self._accept)
        self.button_box.rejected.connect(self.reject)

    def _load_cameras(self) -> None:
        self.camera_combo.clear()
        for camera in sorted(self._project.cameras, key=lambda item: (item.display_order, item.camera_id)):
            self.camera_combo.addItem(camera.name, camera.camera_id)

    def _selected_camera(self) -> CameraConfig | None:
        camera_id = self.camera_combo.currentData()
        for camera in self._project.cameras:
            if camera.camera_id == camera_id:
                return camera
        return None

    def _refresh_views(self) -> None:
        camera = self._selected_camera()
        if camera is None:
            return
        frame = self._camera_frames.get(camera.camera_id)
        if (
            frame is not None
            and camera.homography_image_to_world is not None
            and not camera.coverage_polygon_image
        ):
            self._apply_coverage_proposal(camera, frame, auto_generated=True)
            self._recompute_selected_camera_coverage()
        self.image_canvas.set_image(frame)
        self.image_canvas.set_polygon_mode(self._edit_mode == "coverage")
        self.image_canvas.set_add_enabled(True)
        self.image_canvas.set_drag_enabled(True)
        self.image_canvas.set_delete_enabled(True)
        self.map_view.set_pick_points_mode(self._edit_mode == "anchors")

        anchor_points = [anchor.world_point for anchor in self._project.shared_anchors]
        self.map_view.set_venue_map(self._project.venue_map)
        self.map_view.set_world_viewport(
            compute_world_viewport(
                self._project.cameras,
                self._project.venue_map.zones,
                manual_override=self._project.venue_map.manual_viewport_override,
            )
        )
        self.map_view.set_calibration_points(anchor_points)
        self.map_view.set_camera_overlaps([])
        self.map_view.set_camera_coverages(self._coverage_overlays_for(camera))

        observation_map = {item.anchor_id: item for item in camera.anchor_observations}
        image_points: list[Point] = []
        self._current_anchor_order = []
        current_anchor_id = self._selected_anchor_id
        self.anchor_list.blockSignals(True)
        self.anchor_list.clear()
        for index, anchor in enumerate(self._project.shared_anchors, start=1):
            observation = observation_map.get(anchor.anchor_id)
            label = f"{index}. {anchor.name} world=({anchor.world_point[0]:.2f}, {anchor.world_point[1]:.2f})"
            if observation is not None:
                image_points.append(observation.image_point)
                self._current_anchor_order.append(anchor.anchor_id)
                label += f" image=({observation.image_point[0]:.1f}, {observation.image_point[1]:.1f})"
            else:
                label += " image=(not set)"
            item = QListWidgetItem(label)
            item.setData(256, anchor.anchor_id)
            self.anchor_list.addItem(item)
            if current_anchor_id == anchor.anchor_id:
                self.anchor_list.setCurrentItem(item)
        self.anchor_list.blockSignals(False)

        coverage_points = list(camera.coverage_polygon_image or [])
        if self._edit_mode == "coverage":
            self.image_canvas.set_editable_points(coverage_points)
        else:
            self.image_canvas.set_editable_points(image_points)
        self.image_canvas.set_pending_point(self._pending_image_point if self._edit_mode == "anchors" else None)
        image_index = next(
            (index for index, item in enumerate(self._current_anchor_order) if item == self._selected_anchor_id),
            None,
        )
        shared_index = next(
            (index for index, anchor in enumerate(self._project.shared_anchors) if anchor.anchor_id == self._selected_anchor_id),
            None,
        )
        self.image_canvas.set_selected_point(image_index if self._edit_mode == "anchors" else None)
        self.map_view.set_selected_calibration_point(shared_index)
        self.camera_status_label.setText(
            f"Anchors: {len(self._project.shared_anchors)} | Observations: {len(camera.anchor_observations)} | "
            f"Coverage points: {len(coverage_points)}"
        )
        is_anchor_mode = self._edit_mode == "anchors"
        self.anchor_list.setEnabled(is_anchor_mode)
        self.compute_button.setEnabled(is_anchor_mode)
        self.auto_detect_coverage_button.setEnabled(frame is not None)
        self.reset_coverage_button.setEnabled(frame is not None)
        self.accept_coverage_button.setEnabled(self._edit_mode == "coverage")
        self.remove_observation_button.setText(
            "Remove selected observation" if is_anchor_mode else "Delete selected coverage vertex"
        )
        self.quality_label.setText(self._calibration_status_text(camera))

    def _clear_pending_point(self) -> None:
        self._pending_image_point = None
        self.image_canvas.clear_pending_point()

    @Slot()
    def _on_camera_changed(self) -> None:
        self._clear_pending_point()
        self._selected_anchor_id = None
        self._refresh_views()

    @Slot()
    def _on_mode_changed(self) -> None:
        self._edit_mode = str(self.mode_combo.currentData() or "anchors")
        self._clear_pending_point()
        self._refresh_views()

    @Slot(float, float)
    def _on_image_point_added(self, x: float, y: float) -> None:
        if self._edit_mode == "coverage":
            camera = self._selected_camera()
            if camera is None:
                return
            polygon = list(camera.coverage_polygon_image or [])
            polygon.append((x, y))
            camera.coverage_polygon_image = polygon
            camera.coverage_auto_generated = False
            self._recompute_selected_camera_coverage()
            self._refresh_views()
            return
        self._pending_image_point = (x, y)
        self.image_canvas.set_pending_point(self._pending_image_point)

    @Slot(object)
    def _on_image_selection_changed(self, index: object) -> None:
        if self._edit_mode == "coverage":
            return
        if isinstance(index, int) and 0 <= index < len(self._current_anchor_order):
            anchor_id = self._current_anchor_order[index]
            self._selected_anchor_id = anchor_id
            row = self._anchor_row(anchor_id)
            if row is not None:
                self.anchor_list.setCurrentRow(row)

    @Slot(float, float)
    def _on_map_empty_clicked(self, world_x: float, world_y: float) -> None:
        if self._edit_mode != "anchors":
            return
        if self._pending_image_point is None:
            return
        anchor_id = f"anchor-{len(self._project.shared_anchors) + 1:03d}"
        anchor = SharedAnchor(anchor_id=anchor_id, name=anchor_id, world_point=(world_x, world_y))
        self._project.shared_anchors.append(anchor)
        self._set_observation(anchor_id, self._pending_image_point)
        self._clear_pending_point()
        self._refresh_views()

    @Slot(object)
    def _on_anchor_selected(self, index: object) -> None:
        if self._edit_mode != "anchors":
            return
        if not isinstance(index, int) or index < 0 or index >= len(self._project.shared_anchors):
            return
        anchor = self._project.shared_anchors[index]
        self._selected_anchor_id = anchor.anchor_id
        row = self._anchor_row(anchor.anchor_id)
        if row is not None:
            self.anchor_list.setCurrentRow(row)
        if self._pending_image_point is not None:
            self._set_observation(anchor.anchor_id, self._pending_image_point)
            self._clear_pending_point()
            self._refresh_views()
            return
        camera = self._selected_camera()
        if camera is None:
            return
        observation_map = {item.anchor_id: item for item in camera.anchor_observations}
        visible_index = None
        for position, anchor_id in enumerate(self._current_anchor_order):
            if anchor_id == anchor.anchor_id:
                visible_index = position
                break
        self.image_canvas.set_selected_point(visible_index)
        self.map_view.set_selected_calibration_point(index)

    @Slot(int, float, float)
    def _on_anchor_moved(self, index: int, x: float, y: float) -> None:
        if self._edit_mode != "anchors":
            return
        if index < 0 or index >= len(self._project.shared_anchors):
            return
        self._project.shared_anchors[index].world_point = (x, y)
        self._selected_anchor_id = self._project.shared_anchors[index].anchor_id
        self._refresh_views()

    @Slot(int)
    def _on_anchor_deleted(self, index: int) -> None:
        if self._edit_mode != "anchors":
            return
        if index < 0 or index >= len(self._project.shared_anchors):
            return
        anchor_id = self._project.shared_anchors[index].anchor_id
        del self._project.shared_anchors[index]
        for camera in self._project.cameras:
            camera.anchor_observations = [
                item for item in camera.anchor_observations if item.anchor_id != anchor_id
            ]
        self._refresh_views()

    @Slot(int, float, float)
    def _on_image_point_moved(self, index: int, x: float, y: float) -> None:
        if self._edit_mode == "coverage":
            camera = self._selected_camera()
            if camera is None:
                return
            polygon = list(camera.coverage_polygon_image or [])
            if index < 0 or index >= len(polygon):
                return
            polygon[index] = (x, y)
            camera.coverage_polygon_image = polygon
            camera.coverage_auto_generated = False
            self._recompute_selected_camera_coverage()
            self._refresh_views()
            return
        if index < 0 or index >= len(self._current_anchor_order):
            return
        anchor_id = self._current_anchor_order[index]
        self._selected_anchor_id = anchor_id
        self._set_observation(anchor_id, (x, y))
        self._refresh_views()

    @Slot(int)
    def _on_image_point_deleted(self, index: int) -> None:
        if self._edit_mode == "coverage":
            camera = self._selected_camera()
            if camera is None:
                return
            polygon = list(camera.coverage_polygon_image or [])
            if index < 0 or index >= len(polygon):
                return
            del polygon[index]
            camera.coverage_polygon_image = polygon or None
            camera.coverage_auto_generated = False
            self._recompute_selected_camera_coverage()
            self._refresh_views()
            return
        if index < 0 or index >= len(self._current_anchor_order):
            return
        self._remove_observation(self._current_anchor_order[index])
        self._refresh_views()

    @Slot(int)
    def _on_anchor_list_row_changed(self, row: int) -> None:
        if self._edit_mode != "anchors":
            return
        if row < 0 or row >= self.anchor_list.count():
            self._selected_anchor_id = None
            self.map_view.set_selected_calibration_point(None)
            self.image_canvas.set_selected_point(None)
            return
        anchor_id = self.anchor_list.item(row).data(256)
        self._selected_anchor_id = anchor_id
        shared_index = next(
            (index for index, item in enumerate(self._project.shared_anchors) if item.anchor_id == anchor_id),
            None,
        )
        self.map_view.set_selected_calibration_point(shared_index)
        image_index = next(
            (index for index, item in enumerate(self._current_anchor_order) if item == anchor_id),
            None,
        )
        self.image_canvas.set_selected_point(image_index)

    @Slot()
    def _remove_selected_observation(self) -> None:
        if self._edit_mode == "coverage":
            selected = self.image_canvas.selected_point()
            if selected is None:
                return
            self._on_image_point_deleted(selected)
            return
        row = self.anchor_list.currentRow()
        if row < 0:
            return
        anchor_id = self.anchor_list.item(row).data(256)
        self._remove_observation(anchor_id)
        self._refresh_views()

    def _set_observation(self, anchor_id: str, point: Point) -> None:
        camera = self._selected_camera()
        if camera is None:
            return
        for item in camera.anchor_observations:
            if item.anchor_id == anchor_id:
                item.image_point = point
                return
        camera.anchor_observations.append(CameraAnchorObservation(anchor_id=anchor_id, image_point=point))

    def _remove_observation(self, anchor_id: str) -> None:
        camera = self._selected_camera()
        if camera is None:
            return
        camera.anchor_observations = [
            item for item in camera.anchor_observations if item.anchor_id != anchor_id
        ]

    def _anchor_row(self, anchor_id: str) -> int | None:
        for row in range(self.anchor_list.count()):
            if self.anchor_list.item(row).data(256) == anchor_id:
                return row
        return None

    @Slot()
    def _compute_selected_camera(self) -> None:
        camera = self._selected_camera()
        frame = self._camera_frames.get(camera.camera_id) if camera is not None else None
        if camera is None or frame is None:
            QMessageBox.warning(self, "No frame", "This camera does not have a captured frame yet.")
            return
        anchor_lookup = {anchor.anchor_id: anchor for anchor in self._project.shared_anchors}
        image_points: list[Point] = []
        world_points: list[Point] = []
        for observation in camera.anchor_observations:
            anchor = anchor_lookup.get(observation.anchor_id)
            if anchor is None:
                continue
            image_points.append(observation.image_point)
            world_points.append(anchor.world_point)
        result = compute_homography_result(image_points, world_points, use_ransac=True)
        if result.homography_image_to_world is None:
            QMessageBox.warning(
                self,
                "Calibration failed",
                "At least 4 valid shared anchor observations are required for this camera.",
            )
            return
        warnings = list(result.warnings)
        warnings.extend(self._anchor_distribution_warnings(frame.width(), frame.height(), image_points, world_points))
        unique_warnings = list(dict.fromkeys(warnings))

        camera.homography_image_to_world = result.homography_image_to_world
        camera.frame_width = frame.width()
        camera.frame_height = frame.height()
        camera.calibration_rmse_px = result.reprojection_rmse_px
        camera.calibration_max_error_px = result.max_reprojection_error_px
        camera.calibration_valid = result.is_valid
        if not camera.coverage_polygon_image:
            self._apply_coverage_proposal(camera, frame, auto_generated=True)
        self._recompute_selected_camera_coverage(extra_warnings=unique_warnings)
        self.camera_status_label.setText(
            f"Calibrated {camera.camera_id} with {len(image_points)} anchors"
        )
        self._refresh_views()

    def _coverage_overlays_for(self, camera: CameraConfig) -> list[CameraCoverageOverlay]:
        if not camera.coverage_polygon_world and not camera.coverage_polygon_world_raw:
            return []
        return [
            CameraCoverageOverlay(
                camera_id=camera.camera_id,
                camera_name=camera.name,
                color=camera.panel_color,
                polygon_world=list(camera.coverage_polygon_world or []),
                raw_polygon_world=list(camera.coverage_polygon_world_raw or []),
                calibration_valid=camera.calibration_valid,
                calibration_warning_text=camera.calibration_warning_text,
            )
        ]

    def _calibration_status_text(self, camera: CameraConfig) -> str:
        projected = "yes" if camera.coverage_polygon_world else "no"
        coverage_confidence = (
            f"{camera.coverage_confidence:.2f}"
            if camera.coverage_confidence is not None
            else "n/a"
        )
        rmse_text = (
            f"{camera.calibration_rmse_px:.1f}px"
            if camera.calibration_rmse_px is not None
            else "n/a"
        )
        max_error = (
            f"{camera.calibration_max_error_px:.1f}px"
            if camera.calibration_max_error_px is not None
            else "n/a"
        )
        warnings = camera.calibration_warning_text or "none"
        validity = "valid" if camera.calibration_valid else "needs review"
        return (
            f"Calibration: {validity}\n"
            f"RMSE: {rmse_text}\n"
            f"Max error: {max_error}\n"
            f"Projected coverage valid: {projected}\n"
            f"Coverage confidence: {coverage_confidence}\n"
            f"Warnings: {warnings}"
        )

    @staticmethod
    def _anchor_distribution_warnings(
        frame_width: int,
        frame_height: int,
        image_points: list[Point],
        world_points: list[Point],
    ) -> list[str]:
        warnings: list[str] = []
        if len(image_points) < 4:
            return warnings
        image_x = [point[0] for point in image_points]
        image_y = [point[1] for point in image_points]
        world_x = [point[0] for point in world_points]
        world_y = [point[1] for point in world_points]
        if frame_width > 0 and (max(image_x) - min(image_x)) / frame_width < 0.35:
            warnings.append("Anchor image spread is narrow across width.")
        if frame_height > 0 and (max(image_y) - min(image_y)) / frame_height < 0.25:
            warnings.append("Anchor image spread is narrow across height.")
        if max(world_x) - min(world_x) < 1.5:
            warnings.append("Anchor world spread is narrow across X.")
        if max(world_y) - min(world_y) < 1.5:
            warnings.append("Anchor world spread is narrow across Y.")
        return warnings

    @Slot()
    def _accept(self) -> None:
        uncalibrated = [
            camera.name
            for camera in self._project.cameras
            if camera.enabled and len(camera.anchor_observations) >= 4 and camera.homography_image_to_world is None
        ]
        if uncalibrated:
            reply = QMessageBox.question(
                self,
                "Incomplete calibration",
                "Some cameras have enough anchor observations but no computed homography. Save anyway?",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.calibration_applied.emit(self.project_config)
        self.accept()

    @Slot()
    def _auto_detect_coverage(self) -> None:
        camera = self._selected_camera()
        frame = self._camera_frames.get(camera.camera_id) if camera is not None else None
        if camera is None or frame is None:
            return
        self._apply_coverage_proposal(camera, frame, auto_generated=True)
        self._recompute_selected_camera_coverage()
        self._refresh_views()

    @Slot()
    def _reset_coverage_to_fallback(self) -> None:
        camera = self._selected_camera()
        frame = self._camera_frames.get(camera.camera_id) if camera is not None else None
        if camera is None or frame is None:
            return
        camera.coverage_polygon_image = default_coverage_polygon_image(frame.width(), frame.height())
        camera.coverage_auto_generated = False
        camera.coverage_confidence = 0.25
        camera.coverage_warning_text = "Coverage reset to fallback polygon."
        self._recompute_selected_camera_coverage()
        self._refresh_views()

    @Slot()
    def _accept_coverage(self) -> None:
        camera = self._selected_camera()
        if camera is None:
            return
        camera.coverage_auto_generated = False
        self._recompute_selected_camera_coverage()
        self._refresh_views()

    def _apply_coverage_proposal(self, camera: CameraConfig, frame: QImage, *, auto_generated: bool) -> None:
        frame_bgr = self._qimage_to_bgr(frame)
        proposal = propose_coverage_polygon_image(
            frame_bgr,
            (frame.width(), frame.height()),
        )
        if proposal.polygon_image:
            camera.coverage_polygon_image = list(proposal.polygon_image)
            camera.coverage_auto_generated = auto_generated
            camera.coverage_confidence = proposal.confidence
            camera.coverage_warning_text = " | ".join(proposal.warnings)

    def _recompute_selected_camera_coverage(self, *, extra_warnings: list[str] | None = None) -> None:
        camera = self._selected_camera()
        if camera is None:
            return
        coverage_result = recompute_camera_coverage(camera)
        camera.coverage_polygon_world_raw = coverage_result.raw_polygon_world or None
        camera.coverage_polygon_world = coverage_result.sanitized_polygon_world or None
        merged_warnings = list(extra_warnings or [])
        if camera.coverage_warning_text:
            merged_warnings.append(camera.coverage_warning_text)
        merged_warnings.extend(coverage_result.warnings)
        merged_warnings = list(dict.fromkeys([warning for warning in merged_warnings if warning]))
        camera.coverage_warning_text = " | ".join(coverage_result.warnings)
        camera.calibration_warning_text = " | ".join(merged_warnings)
        camera.calibration_valid = bool(camera.homography_image_to_world) and coverage_result.is_valid

    @staticmethod
    def _qimage_to_bgr(image: QImage) -> np.ndarray:
        converted = image.convertToFormat(QImage.Format.Format_RGB888)
        width = converted.width()
        height = converted.height()
        ptr = converted.bits()
        buffer = np.frombuffer(ptr, dtype=np.uint8, count=height * width * 3)
        rgb = buffer.reshape((height, width, 3)).copy()
        return rgb[:, :, ::-1]
