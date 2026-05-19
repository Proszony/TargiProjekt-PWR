import unittest

from core.zones import point_in_polygon


class ZoneTests(unittest.TestCase):
    def test_point_in_polygon(self) -> None:
        polygon = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]
        self.assertTrue(point_in_polygon((2.0, 2.0), polygon))
        self.assertFalse(point_in_polygon((6.0, 2.0), polygon))


if __name__ == "__main__":
    unittest.main()
