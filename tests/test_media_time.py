import unittest
from types import SimpleNamespace

from core.media_time import resolve_media_time


class MediaTimeTests(unittest.TestCase):
    def test_resolve_media_time_prefers_frame_time(self) -> None:
        frame = SimpleNamespace(time=1.25, pts=30, time_base=1 / 30)
        media_time = resolve_media_time(frame, nominal_fps=30.0, frame_index=10)
        self.assertAlmostEqual(media_time, 1.25)

    def test_resolve_media_time_falls_back_to_pts(self) -> None:
        frame = SimpleNamespace(time=None, pts=15, time_base=1 / 30)
        media_time = resolve_media_time(frame, nominal_fps=30.0, frame_index=10)
        self.assertAlmostEqual(media_time, 0.5)

    def test_resolve_media_time_falls_back_to_frame_index(self) -> None:
        frame = SimpleNamespace(time=None, pts=None, time_base=None)
        media_time = resolve_media_time(frame, nominal_fps=25.0, frame_index=50)
        self.assertAlmostEqual(media_time, 2.0)


if __name__ == "__main__":
    unittest.main()
