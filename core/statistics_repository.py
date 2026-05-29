from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from core.models import AnalyticsSnapshot, BoothVisitSession, HeatmapCell, HeatmapSnapshot, VenueMapConfig, WorldViewport


class StatisticsRepository:
    _REQUIRED_TABLE_COLUMNS = {
        "sessions": {
            "id",
            "started_at",
            "ended_at",
            "source_type",
            "source_label",
            "camera_id",
        },
        "booth_visit_sessions": {
            "id",
            "session_id",
            "visit_id",
            "analytics_track_id",
            "zone_id",
            "entered_at",
            "left_at",
            "dwell_s",
            "source_camera_ids_json",
            "dedup_mode",
        },
        "zone_snapshots": {
            "id",
            "session_id",
            "ts",
            "zone_id",
            "zone_name",
            "zone_kind",
            "occupancy",
            "unique_visits",
            "total_dwell_s",
            "avg_dwell_s",
            "median_dwell_s",
            "peak_occupancy",
        },
        "session_heatmaps": {
            "session_id",
            "created_at",
            "viewport_json",
            "columns",
            "rows",
            "max_dwell_s",
            "total_dwell_s",
            "cells_json",
        },
    }

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _managed_connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._managed_connection() as connection:
            self._repair_incompatible_schema(connection)
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    source_type TEXT NOT NULL,
                    source_label TEXT NOT NULL,
                    camera_id TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS booth_visit_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    visit_id TEXT NOT NULL,
                    analytics_track_id TEXT NOT NULL,
                    zone_id TEXT NOT NULL,
                    entered_at REAL NOT NULL,
                    left_at REAL,
                    dwell_s REAL NOT NULL,
                    source_camera_ids_json TEXT NOT NULL,
                    dedup_mode TEXT NOT NULL,
                    UNIQUE(session_id, visit_id),
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS zone_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    ts REAL NOT NULL,
                    zone_id TEXT NOT NULL,
                    zone_name TEXT NOT NULL,
                    zone_kind TEXT NOT NULL,
                    occupancy INTEGER NOT NULL,
                    unique_visits INTEGER NOT NULL,
                    total_dwell_s REAL NOT NULL,
                    avg_dwell_s REAL NOT NULL,
                    median_dwell_s REAL NOT NULL,
                    peak_occupancy INTEGER NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS session_heatmaps (
                    session_id INTEGER PRIMARY KEY,
                    created_at REAL NOT NULL,
                    viewport_json TEXT NOT NULL,
                    columns INTEGER NOT NULL,
                    rows INTEGER NOT NULL,
                    max_dwell_s REAL NOT NULL,
                    total_dwell_s REAL NOT NULL,
                    cells_json TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                """
            )

    def _repair_incompatible_schema(self, connection: sqlite3.Connection) -> None:
        for table_name, required_columns in self._REQUIRED_TABLE_COLUMNS.items():
            existing_columns = self._table_columns(connection, table_name)
            if not existing_columns:
                continue
            if required_columns.issubset(existing_columns):
                continue
            connection.execute(f"DROP TABLE IF EXISTS {table_name}")

    @staticmethod
    def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(row[1]) for row in rows}

    def start_session(
        self,
        started_at: float,
        source_type: str,
        source_label: str,
        camera_id: str,
    ) -> int:
        with self._managed_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO sessions (started_at, source_type, source_label, camera_id)
                VALUES (?, ?, ?, ?)
                """,
                (started_at, source_type, source_label, camera_id),
            )
            return int(cursor.lastrowid)

    def finish_session(self, session_id: int, ended_at: float) -> None:
        with self._managed_connection() as connection:
            connection.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?",
                (ended_at, session_id),
            )

    def record_visit_sessions(self, session_id: int, sessions: list[BoothVisitSession]) -> None:
        if not sessions:
            return
        rows = [
            (
                session_id,
                visit.visit_id,
                visit.analytics_track_id,
                visit.zone_id,
                visit.entered_at,
                visit.left_at,
                visit.dwell_s,
                json.dumps(visit.source_camera_ids, ensure_ascii=True),
                visit.dedup_mode,
            )
            for visit in sessions
        ]
        with self._managed_connection() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO booth_visit_sessions (
                    session_id, visit_id, analytics_track_id, zone_id,
                    entered_at, left_at, dwell_s, source_camera_ids_json, dedup_mode
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def record_snapshot(self, session_id: int, snapshot: AnalyticsSnapshot, venue_map: VenueMapConfig) -> None:
        if not snapshot.zone_metrics:
            return
        rows = []
        for zone in venue_map.zones:
            metrics = snapshot.zone_metrics.get(zone.zone_id)
            if metrics is None:
                continue
            rows.append(
                (
                    session_id,
                    snapshot.timestamp,
                    zone.zone_id,
                    zone.name,
                    zone.kind,
                    metrics.current_occupancy,
                    metrics.unique_visits,
                    metrics.total_dwell_s,
                    metrics.avg_dwell_s,
                    metrics.median_dwell_s,
                    metrics.peak_occupancy,
                )
            )
        if not rows:
            return
        with self._managed_connection() as connection:
            connection.executemany(
                """
                INSERT INTO zone_snapshots (
                    session_id, ts, zone_id, zone_name, zone_kind, occupancy,
                    unique_visits, total_dwell_s, avg_dwell_s, median_dwell_s, peak_occupancy
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def record_session_heatmap(self, session_id: int, heatmap: HeatmapSnapshot) -> None:
        if heatmap.columns <= 0 or heatmap.rows <= 0:
            return
        viewport_json = json.dumps(heatmap.viewport.to_dict(), ensure_ascii=True)
        cells_json = json.dumps(
            [
                {
                    "x_index": cell.x_index,
                    "y_index": cell.y_index,
                    "dwell_s": cell.dwell_s,
                }
                for cell in heatmap.cells
                if cell.dwell_s > 0.0
            ],
            ensure_ascii=True,
        )
        with self._managed_connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO session_heatmaps (
                    session_id, created_at, viewport_json, columns, rows,
                    max_dwell_s, total_dwell_s, cells_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    heatmap.timestamp,
                    viewport_json,
                    heatmap.columns,
                    heatmap.rows,
                    heatmap.max_dwell_s,
                    heatmap.total_dwell_s,
                    cells_json,
                ),
            )

    def list_sessions(self) -> list[dict[str, object]]:
        with self._managed_connection() as connection:
            rows = connection.execute(
                """
                SELECT id, started_at, ended_at, source_type, source_label, camera_id
                FROM sessions
                ORDER BY started_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def load_session_zone_metrics(self, session_id: int) -> list[dict[str, object]]:
        with self._managed_connection() as connection:
            rows = connection.execute(
                """
                SELECT latest.zone_id, latest.zone_name, latest.zone_kind,
                       latest.occupancy AS current_occupancy,
                       latest.unique_visits, latest.total_dwell_s,
                       latest.avg_dwell_s, latest.median_dwell_s, latest.peak_occupancy
                FROM zone_snapshots latest
                INNER JOIN (
                    SELECT zone_id, MAX(ts) AS max_ts
                    FROM zone_snapshots
                    WHERE session_id = ?
                    GROUP BY zone_id
                ) grouped
                ON latest.zone_id = grouped.zone_id AND latest.ts = grouped.max_ts
                WHERE latest.session_id = ?
                ORDER BY latest.zone_name
                """,
                (session_id, session_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def load_zone_timeline(self, session_id: int, zone_id: str) -> list[dict[str, object]]:
        with self._managed_connection() as connection:
            rows = connection.execute(
                """
                SELECT ts, occupancy, unique_visits, total_dwell_s, avg_dwell_s, median_dwell_s, peak_occupancy
                FROM zone_snapshots
                WHERE session_id = ? AND zone_id = ?
                ORDER BY ts ASC
                """,
                (session_id, zone_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def load_booth_visit_sessions(self, session_id: int) -> list[dict[str, object]]:
        with self._managed_connection() as connection:
            rows = connection.execute(
                """
                SELECT visit_id, analytics_track_id, zone_id, entered_at, left_at, dwell_s, source_camera_ids_json, dedup_mode
                FROM booth_visit_sessions
                WHERE session_id = ?
                ORDER BY entered_at ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def load_session_heatmap(self, session_id: int) -> HeatmapSnapshot | None:
        with self._managed_connection() as connection:
            row = connection.execute(
                """
                SELECT created_at, viewport_json, columns, rows, max_dwell_s, total_dwell_s, cells_json
                FROM session_heatmaps
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        viewport = WorldViewport.from_dict(json.loads(str(row["viewport_json"])))
        cells_data = json.loads(str(row["cells_json"]))
        cells = [
            HeatmapCell(
                x_index=int(item["x_index"]),
                y_index=int(item["y_index"]),
                dwell_s=float(item["dwell_s"]),
            )
            for item in cells_data
        ]
        return HeatmapSnapshot(
            timestamp=float(row["created_at"]),
            viewport=viewport,
            columns=int(row["columns"]),
            rows=int(row["rows"]),
            max_dwell_s=float(row["max_dwell_s"]),
            total_dwell_s=float(row["total_dwell_s"]),
            cells=cells,
        )
