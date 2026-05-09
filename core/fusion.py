from __future__ import annotations

from core.models import GlobalTrack, LocalTrack


class GlobalFusionEngine:
    def __init__(self) -> None:
        self._global_tracks: dict[str, GlobalTrack] = {}

    def reset(self) -> None:
        self._global_tracks.clear()

    def update(
        self,
        timestamp: float,
        active_local_tracks: dict[int, LocalTrack],
        expired_local_tracks: list[LocalTrack],
    ) -> dict[str, GlobalTrack]:
        active_ids = set()

        for track in active_local_tracks.values():
            global_id = self._global_id(track)
            active_ids.add(global_id)
            global_track = self._global_tracks.get(global_id)
            if global_track is None:
                global_track = GlobalTrack(
                    global_track_id=global_id,
                    member_local_tracks=[f"{track.camera_id}:{track.local_track_id}"],
                    first_seen_ts=track.first_seen_ts,
                    last_seen_ts=track.last_seen_ts,
                    positions_world=[track.ground_anchor_world] if track.ground_anchor_world else [],
                    velocity=track.velocity,
                    current_bbox_xyxy=track.current_bbox_xyxy,
                    ground_anchor_world=track.ground_anchor_world,
                    smoothed_ground_anchor_world=track.smoothed_ground_anchor_world,
                    ground_anchor_image=track.ground_anchor_image,
                    camera_id=track.camera_id,
                    active=True,
                    appearance_descriptor=track.appearance_descriptor,
                )
                self._global_tracks[global_id] = global_track

            global_track.active = True
            global_track.last_seen_ts = track.last_seen_ts
            global_track.camera_id = track.camera_id
            global_track.current_bbox_xyxy = track.current_bbox_xyxy
            global_track.ground_anchor_world = track.ground_anchor_world
            global_track.smoothed_ground_anchor_world = track.smoothed_ground_anchor_world
            global_track.ground_anchor_image = track.ground_anchor_image
            global_track.velocity = track.velocity
            global_track.appearance_descriptor = track.appearance_descriptor
            global_track.inactive_since_ts = None
            global_track.reactivation_deadline_ts = None
            local_member = f"{track.camera_id}:{track.local_track_id}"
            if local_member not in global_track.member_local_tracks:
                global_track.member_local_tracks.append(local_member)
            if track.ground_anchor_world is not None:
                global_track.positions_world.append(track.ground_anchor_world)
                if len(global_track.positions_world) > 120:
                    global_track.positions_world = global_track.positions_world[-120:]

        for track in expired_local_tracks:
            global_id = self._global_id(track)
            if global_id in self._global_tracks:
                global_track = self._global_tracks[global_id]
                global_track.active = False
                global_track.last_seen_ts = track.last_seen_ts
                global_track.inactive_since_ts = track.last_seen_ts
                global_track.reactivation_deadline_ts = None
                global_track.camera_id = track.camera_id

        return dict(self._global_tracks)

    @staticmethod
    def _global_id(track: LocalTrack) -> str:
        return f"{track.camera_id}:T{track.local_track_id:04d}"
