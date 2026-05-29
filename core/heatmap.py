from __future__ import annotations

from core.models import HeatmapCell, HeatmapSnapshot, MapPresence, WorldViewport


class HeatmapAccumulator:
    def __init__(
        self,
        *,
        enabled: bool = True,
        sample_interval_s: float = 0.5,
        grid_columns: int = 160,
        min_rows: int = 40,
        max_rows: int = 160,
    ) -> None:
        self.enabled = enabled
        self.sample_interval_s = max(float(sample_interval_s), 0.05)
        self.grid_columns = max(int(grid_columns), 1)
        self.min_rows = max(int(min_rows), 1)
        self.max_rows = max(int(max_rows), self.min_rows)
        self._viewport = WorldViewport()
        self._columns = self.grid_columns
        self._rows = self.min_rows
        self._cells: dict[int, float] = {}
        self._last_sample_ts: float | None = None

    def configure(
        self,
        *,
        enabled: bool,
        sample_interval_s: float,
        grid_columns: int,
        min_rows: int,
        max_rows: int,
    ) -> None:
        self.enabled = enabled
        self.sample_interval_s = max(float(sample_interval_s), 0.05)
        self.grid_columns = max(int(grid_columns), 1)
        self.min_rows = max(int(min_rows), 1)
        self.max_rows = max(int(max_rows), self.min_rows)

    def reset(self, viewport: WorldViewport | None = None, timestamp: float | None = None) -> None:
        if viewport is not None:
            self._viewport = viewport
        self._columns, self._rows = self._grid_shape(self._viewport)
        self._cells.clear()
        self._last_sample_ts = timestamp

    def update(
        self,
        timestamp: float,
        presences: list[MapPresence],
        viewport: WorldViewport,
    ) -> HeatmapSnapshot | None:
        if not self.enabled:
            self._last_sample_ts = timestamp
            return None
        if self._viewport_key(viewport) != self._viewport_key(self._viewport):
            self.reset(viewport, timestamp)
            return self.snapshot(timestamp)
        if self._last_sample_ts is None:
            self._last_sample_ts = timestamp
            return self.snapshot(timestamp)

        elapsed_s = max(timestamp - self._last_sample_ts, 0.0)
        if elapsed_s < self.sample_interval_s:
            return self.snapshot(timestamp)

        self._last_sample_ts = timestamp
        if presences:
            for presence in presences:
                for cell_index, weight in self._weighted_cells_for_point(presence.world_point):
                    self._cells[cell_index] = self._cells.get(cell_index, 0.0) + elapsed_s * weight
        return self.snapshot(timestamp)

    def snapshot(self, timestamp: float) -> HeatmapSnapshot:
        cells = [
            HeatmapCell(
                x_index=cell_index % self._columns,
                y_index=cell_index // self._columns,
                dwell_s=dwell_s,
            )
            for cell_index, dwell_s in sorted(self._cells.items())
            if dwell_s > 0.0
        ]
        total_dwell_s = sum(cell.dwell_s for cell in cells)
        max_dwell_s = max((cell.dwell_s for cell in cells), default=0.0)
        return HeatmapSnapshot(
            timestamp=timestamp,
            viewport=self._viewport,
            columns=self._columns,
            rows=self._rows,
            max_dwell_s=max_dwell_s,
            total_dwell_s=total_dwell_s,
            cells=cells,
        )

    def _weighted_cells_for_point(self, point: tuple[float, float]) -> list[tuple[int, float]]:
        width = max(self._viewport.max_x - self._viewport.min_x, 1e-6)
        height = max(self._viewport.max_y - self._viewport.min_y, 1e-6)
        x_ratio = (point[0] - self._viewport.min_x) / width
        y_ratio = (point[1] - self._viewport.min_y) / height
        if x_ratio < 0.0 or x_ratio > 1.0 or y_ratio < 0.0 or y_ratio > 1.0:
            return []

        grid_x = min(max(x_ratio * self._columns - 0.5, 0.0), self._columns - 1.0)
        grid_y = min(max(y_ratio * self._rows - 0.5, 0.0), self._rows - 1.0)
        x0 = int(grid_x)
        y0 = int(grid_y)
        x1 = min(x0 + 1, self._columns - 1)
        y1 = min(y0 + 1, self._rows - 1)
        x_fraction = grid_x - x0
        y_fraction = grid_y - y0
        weighted = [
            (x0, y0, (1.0 - x_fraction) * (1.0 - y_fraction)),
            (x1, y0, x_fraction * (1.0 - y_fraction)),
            (x0, y1, (1.0 - x_fraction) * y_fraction),
            (x1, y1, x_fraction * y_fraction),
        ]
        merged: dict[int, float] = {}
        for x_index, y_index, weight in weighted:
            if weight <= 0.0:
                continue
            cell_index = y_index * self._columns + x_index
            merged[cell_index] = merged.get(cell_index, 0.0) + weight
        return sorted(merged.items())

    def _grid_shape(self, viewport: WorldViewport) -> tuple[int, int]:
        width = max(viewport.max_x - viewport.min_x, 1e-6)
        height = max(viewport.max_y - viewport.min_y, 1e-6)
        rows = int(round(self.grid_columns * height / width))
        return self.grid_columns, min(max(rows, self.min_rows), self.max_rows)

    @staticmethod
    def _viewport_key(viewport: WorldViewport) -> tuple[float, float, float, float]:
        return (
            round(viewport.min_x, 6),
            round(viewport.min_y, 6),
            round(viewport.max_x, 6),
            round(viewport.max_y, 6),
        )
