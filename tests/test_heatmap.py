import unittest

from core.heatmap import HeatmapAccumulator
from core.models import MapPresence, WorldViewport


class HeatmapAccumulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.viewport = WorldViewport(min_x=0.0, min_y=0.0, max_x=10.0, max_y=5.0)
        self.accumulator = HeatmapAccumulator(
            enabled=True,
            sample_interval_s=1.0,
            grid_columns=10,
            min_rows=4,
            max_rows=8,
        )
        self.accumulator.reset(self.viewport, timestamp=0.0)

    def test_accumulates_dwell_into_expected_cell(self) -> None:
        snapshot = self.accumulator.update(
            1.0,
            [MapPresence(presence_id="p1", world_point=(2.1, 1.1))],
            self.viewport,
        )

        self.assertIsNotNone(snapshot)
        self.assertEqual(len(snapshot.cells), 4)
        self.assertIn((2, 1), {(cell.x_index, cell.y_index) for cell in snapshot.cells})
        self.assertAlmostEqual(snapshot.total_dwell_s, 1.0)

    def test_does_not_update_before_sample_interval(self) -> None:
        first = self.accumulator.update(
            0.5,
            [MapPresence(presence_id="p1", world_point=(2.1, 1.1))],
            self.viewport,
        )

        self.assertEqual(first.cells, [])
        self.assertEqual(first.total_dwell_s, 0.0)

    def test_multiple_presences_add_independently(self) -> None:
        snapshot = self.accumulator.update(
            1.0,
            [
                MapPresence(presence_id="p1", world_point=(2.1, 1.1)),
                MapPresence(presence_id="p2", world_point=(8.0, 4.0)),
            ],
            self.viewport,
        )

        self.assertGreaterEqual(len(snapshot.cells), 4)
        self.assertAlmostEqual(snapshot.total_dwell_s, 2.0)

    def test_empty_presences_keep_prior_data_without_adding(self) -> None:
        self.accumulator.update(
            1.0,
            [MapPresence(presence_id="p1", world_point=(2.1, 1.1))],
            self.viewport,
        )
        snapshot = self.accumulator.update(2.0, [], self.viewport)

        self.assertEqual(len(snapshot.cells), 4)
        self.assertAlmostEqual(snapshot.total_dwell_s, 1.0)

    def test_viewport_change_resets_grid_deterministically(self) -> None:
        self.accumulator.update(
            1.0,
            [MapPresence(presence_id="p1", world_point=(2.1, 1.1))],
            self.viewport,
        )
        next_viewport = WorldViewport(min_x=0.0, min_y=0.0, max_x=5.0, max_y=5.0)
        snapshot = self.accumulator.update(
            2.0,
            [MapPresence(presence_id="p1", world_point=(2.1, 1.1))],
            next_viewport,
        )

        self.assertEqual(snapshot.cells, [])
        self.assertEqual(snapshot.rows, 8)
        self.assertEqual(snapshot.columns, 10)

    def test_sparse_snapshot_excludes_zero_cells_and_tracks_max(self) -> None:
        self.accumulator.update(
            1.0,
            [MapPresence(presence_id="p1", world_point=(2.1, 1.1))],
            self.viewport,
        )
        snapshot = self.accumulator.update(
            2.0,
            [MapPresence(presence_id="p1", world_point=(2.1, 1.1))],
            self.viewport,
        )

        self.assertEqual(len(snapshot.cells), 4)
        self.assertAlmostEqual(snapshot.max_dwell_s, 0.72)
        self.assertAlmostEqual(snapshot.total_dwell_s, 2.0)


if __name__ == "__main__":
    unittest.main()
