from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    from core.streaming import CameraPipelineWorker
    from core.models import CameraAnchorObservation, CameraConfig, CameraCoverageComputationResult
except ModuleNotFoundError as exc:  # pragma: no cover - optional video/UI dependency
    CameraPipelineWorker = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(CameraPipelineWorker is None, f"Streaming dependencies unavailable: {_IMPORT_ERROR}")
class StreamingPreviewTests(unittest.TestCase):
    def test_preview_gate_emits_first_frame_then_respects_interval(self) -> None:
        worker = CameraPipelineWorker.__new__(CameraPipelineWorker)
        worker._preview_interval_s = 1.0 / 12.0
        worker._last_preview_frame_published_at = None

        self.assertTrue(worker._preview_frame_due())

        worker._last_preview_frame_published_at = 100.0
        with patch("core.streaming.time.perf_counter", return_value=100.01):
            self.assertFalse(worker._preview_frame_due())
        with patch("core.streaming.time.perf_counter", return_value=100.09):
            self.assertTrue(worker._preview_frame_due())

    def test_camera_coverage_recompute_is_cached_until_signature_changes(self) -> None:
        worker = CameraPipelineWorker.__new__(CameraPipelineWorker)
        worker._coverage_recompute_signature = None
        frame = _FrameStub(width=200, height=100)
        camera = CameraConfig(
            homography_image_to_world=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            coverage_polygon_image=[(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)],
            frame_width=200,
            frame_height=100,
        )
        coverage_result = CameraCoverageComputationResult(
            raw_polygon_world=[(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)],
            sanitized_polygon_world=[(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)],
            is_valid=True,
        )

        with patch("core.streaming.recompute_camera_coverage", return_value=coverage_result) as recompute:
            worker._refresh_camera_coverage(camera, frame)
            worker._refresh_camera_coverage(camera, frame)
            self.assertEqual(recompute.call_count, 1)

            camera.coverage_polygon_image = [(1.0, 0.0), (100.0, 0.0), (100.0, 50.0), (1.0, 50.0)]
            worker._refresh_camera_coverage(camera, frame)
            self.assertEqual(recompute.call_count, 2)

            camera.anchor_observations = [CameraAnchorObservation(anchor_id="a1", image_point=(5.0, 6.0))]
            worker._refresh_camera_coverage(camera, frame)
            self.assertEqual(recompute.call_count, 3)


class _FrameStub:
    def __init__(self, *, width: int, height: int) -> None:
        self.shape = (height, width, 3)


if __name__ == "__main__":
    unittest.main()
