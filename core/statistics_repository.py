from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from core.models import AnalyticsEvent, AnalyticsSnapshot, GlobalTrack, VenueMapConfig


class StatisticsRepository:
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

                CREATE TABLE IF NOT EXISTS zone_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    ts REAL NOT NULL,
                    global_track_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    zone_id TEXT,
                    payload_json TEXT NOT NULL,
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
                    unique_entries INTEGER NOT NULL,
                    total_dwell_s REAL NOT NULL,
                    avg_dwell_s REAL NOT NULL,
                    return_count INTEGER NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS track_lifecycle (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    global_track_id TEXT NOT NULL,
                    first_seen_ts REAL NOT NULL,
                    last_seen_ts REAL NOT NULL,
                    UNIQUE(session_id, global_track_id),
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                """
            )

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

    def record_events(self, session_id: int, events: list[AnalyticsEvent]) -> None:
        if not events:
            return
        rows = [
            (
                session_id,
                event.timestamp,
                event.global_track_id,
                event.event_type,
                event.zone_id,
                json.dumps(event.payload, ensure_ascii=True),
            )
            for event in events
        ]
        with self._managed_connection() as connection:
            connection.executemany(
                """
                INSERT INTO zone_events (session_id, ts, global_track_id, event_type, zone_id, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
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
                    metrics.unique_entries,
                    metrics.total_dwell_s,
                    metrics.avg_dwell_s,
                    metrics.return_count,
                )
            )
        if not rows:
            return
        with self._managed_connection() as connection:
            connection.executemany(
                """
                INSERT INTO zone_snapshots (
                    session_id, ts, zone_id, zone_name, zone_kind, occupancy,
                    unique_entries, total_dwell_s, avg_dwell_s, return_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def upsert_track_lifecycle(self, session_id: int, tracks: dict[str, GlobalTrack]) -> None:
        if not tracks:
            return
        rows = [
            (
                session_id,
                track.global_track_id,
                track.first_seen_ts,
                track.last_seen_ts,
            )
            for track in tracks.values()
        ]
        with self._managed_connection() as connection:
            connection.executemany(
                """
                INSERT INTO track_lifecycle (session_id, global_track_id, first_seen_ts, last_seen_ts)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id, global_track_id) DO UPDATE SET
                    first_seen_ts = MIN(first_seen_ts, excluded.first_seen_ts),
                    last_seen_ts = MAX(last_seen_ts, excluded.last_seen_ts)
                """,
                rows,
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
                       latest.unique_entries, latest.total_dwell_s,
                       latest.avg_dwell_s, latest.return_count
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
                SELECT ts, occupancy, unique_entries, total_dwell_s, avg_dwell_s, return_count
                FROM zone_snapshots
                WHERE session_id = ? AND zone_id = ?
                ORDER BY ts ASC
                """,
                (session_id, zone_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def load_events(self, session_id: int) -> list[dict[str, object]]:
        with self._managed_connection() as connection:
            rows = connection.execute(
                """
                SELECT ts, global_track_id, event_type, zone_id, payload_json
                FROM zone_events
                WHERE session_id = ?
                ORDER BY ts ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]
