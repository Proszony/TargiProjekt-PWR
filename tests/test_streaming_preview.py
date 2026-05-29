from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    from core.streaming import CameraPipelineWorker
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


if __name__ == "__main__":
    unittest.main()
