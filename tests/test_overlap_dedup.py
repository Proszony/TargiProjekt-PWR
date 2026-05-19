import unittest

from core.camera_overlap import build_camera_overlap_graph
from core.models import CameraConfig, CameraIdentityTrack, MapDedupConfig, OverlapDedupConfig, ReIDConfig
from core.overlap_dedup import OverlapDedupEngine


def _track(camera_id: str, suffix: int, point: tuple[float, float], appearance: list[float]) -> CameraIdentityTrack:
    return CameraIdentityTrack(
        camera_person_id=f"{camera_id}:P{suffix:05d}",
        camera_id=camera_id,
        first_seen_ts=0.0,
        last_seen_ts=0.0,
        active=True,
        appearance_prototype=list(appearance),
        appearance_memory=[list(appearance)],
        ground_anchor_world=point,
        smoothed_ground_anchor_world=point,
    )


class OverlapDedupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cameras = [
            CameraConfig(
                camera_id="camera-a",
                calibration_valid=True,
                coverage_polygon_world=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
            ),
            CameraConfig(
                camera_id="camera-b",
                calibration_valid=True,
                coverage_polygon_world=[(2.0, 0.0), (6.0, 0.0), (6.0, 4.0), (2.0, 4.0)],
            ),
        ]
        self.graph = build_camera_overlap_graph(self.cameras, OverlapDedupConfig())

    def test_same_person_in_overlap_merges_after_confirmation(self) -> None:
        engine = OverlapDedupEngine(OverlapDedupConfig(confirmation_frames=2), ReIDConfig())
        left = _track("camera-a", 1, (2.5, 2.0), [1.0, 0.0, 0.0])
        right = _track("camera-b", 2, (2.7, 2.0), [0.99, 0.01, 0.0])

        first = engine.update(0.0, {"camera-a": {left.camera_person_id: left}, "camera-b": {right.camera_person_id: right}}, [], self.graph)
        second = engine.update(0.2, {"camera-a": {left.camera_person_id: left}, "camera-b": {right.camera_person_id: right}}, [], self.graph)

        self.assertEqual(sum(1 for track in first.values() if track.active), 2)
        self.assertEqual(sum(1 for track in second.values() if track.active), 1)

    def test_weak_similarity_does_not_merge(self) -> None:
        engine = OverlapDedupEngine(OverlapDedupConfig(confirmation_frames=1), ReIDConfig())
        left = _track("camera-a", 1, (2.5, 2.0), [1.0, 0.0, 0.0])
        right = _track("camera-b", 2, (2.7, 2.0), [0.0, 1.0, 0.0])

        tracks = engine.update(0.0, {"camera-a": {left.camera_person_id: left}, "camera-b": {right.camera_person_id: right}}, [], self.graph)
        self.assertEqual(sum(1 for track in tracks.values() if track.active), 2)

    def test_no_overlap_relation_means_no_cross_camera_dedup(self) -> None:
        engine = OverlapDedupEngine(OverlapDedupConfig(confirmation_frames=1), ReIDConfig())
        graph = build_camera_overlap_graph(
            [
                CameraConfig(camera_id="camera-a", calibration_valid=True, coverage_polygon_world=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
                CameraConfig(camera_id="camera-b", calibration_valid=True, coverage_polygon_world=[(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)]),
            ],
            OverlapDedupConfig(),
        )
        left = _track("camera-a", 1, (0.5, 0.5), [1.0, 0.0, 0.0])
        right = _track("camera-b", 2, (5.5, 5.5), [1.0, 0.0, 0.0])

        tracks = engine.update(0.0, {"camera-a": {left.camera_person_id: left}, "camera-b": {right.camera_person_id: right}}, [], graph)
        self.assertEqual(sum(1 for track in tracks.values() if track.active), 2)

    def test_two_people_seen_by_both_cameras_result_in_two_map_presences_not_four(self) -> None:
        engine = OverlapDedupEngine(OverlapDedupConfig(confirmation_frames=1), ReIDConfig())
        left_a = _track("camera-a", 1, (2.40, 2.0), [1.0, 0.0, 0.0])
        right_a = _track("camera-b", 1, (2.55, 2.0), [0.95, 0.05, 0.0])
        left_b = _track("camera-a", 2, (3.10, 2.2), [0.0, 1.0, 0.0])
        right_b = _track("camera-b", 2, (3.25, 2.2), [0.05, 0.95, 0.0])

        engine.resolve_map_presences(
            0.0,
            {
                "camera-a": {left_a.camera_person_id: left_a, left_b.camera_person_id: left_b},
                "camera-b": {right_a.camera_person_id: right_a, right_b.camera_person_id: right_b},
            },
            self.graph,
            MapDedupConfig(confirmation_frames=2, tracklet_min_points=2),
        )
        presences = engine.resolve_map_presences(
            0.1,
            {
                "camera-a": {left_a.camera_person_id: left_a, left_b.camera_person_id: left_b},
                "camera-b": {right_a.camera_person_id: right_a, right_b.camera_person_id: right_b},
            },
            self.graph,
            MapDedupConfig(confirmation_frames=2, tracklet_min_points=2),
        )
        presences = engine.resolve_map_presences(
            0.2,
            {
                "camera-a": {left_a.camera_person_id: left_a, left_b.camera_person_id: left_b},
                "camera-b": {right_a.camera_person_id: right_a, right_b.camera_person_id: right_b},
            },
            self.graph,
            MapDedupConfig(confirmation_frames=2, tracklet_min_points=2),
        )

        self.assertEqual(len(presences), 2)
        self.assertTrue(all(presence.merged_for_counting for presence in presences))

    def test_same_person_in_overlap_merges_even_without_strong_reid_when_geometry_is_strong(self) -> None:
        engine = OverlapDedupEngine(OverlapDedupConfig(confirmation_frames=1), ReIDConfig())
        left = _track("camera-a", 1, (2.5, 2.0), [1.0, 0.0, 0.0])
        right = _track("camera-b", 2, (2.55, 2.0), [])

        engine.resolve_map_presences(
            0.0,
            {"camera-a": {left.camera_person_id: left}, "camera-b": {right.camera_person_id: right}},
            self.graph,
            MapDedupConfig(confirmation_frames=2, tracklet_min_points=2),
        )
        presences = engine.resolve_map_presences(
            0.1,
            {"camera-a": {left.camera_person_id: left}, "camera-b": {right.camera_person_id: right}},
            self.graph,
            MapDedupConfig(confirmation_frames=2, tracklet_min_points=2),
        )
        presences = engine.resolve_map_presences(
            0.2,
            {"camera-a": {left.camera_person_id: left}, "camera-b": {right.camera_person_id: right}},
            self.graph,
            MapDedupConfig(confirmation_frames=2, tracklet_min_points=2),
        )

        self.assertEqual(len(presences), 1)
        self.assertTrue(presences[0].merged_for_counting)

    def test_two_close_people_do_not_cross_merge_when_one_to_one_assignment_is_used(self) -> None:
        engine = OverlapDedupEngine(OverlapDedupConfig(confirmation_frames=1), ReIDConfig())
        left_a = _track("camera-a", 1, (2.40, 2.0), [1.0, 0.0, 0.0])
        right_a = _track("camera-b", 1, (2.52, 2.0), [0.95, 0.05, 0.0])
        left_b = _track("camera-a", 2, (2.80, 2.0), [0.0, 1.0, 0.0])
        right_b = _track("camera-b", 2, (2.92, 2.0), [0.05, 0.95, 0.0])

        engine.resolve_map_presences(
            0.0,
            {
                "camera-a": {left_a.camera_person_id: left_a, left_b.camera_person_id: left_b},
                "camera-b": {right_a.camera_person_id: right_a, right_b.camera_person_id: right_b},
            },
            self.graph,
            MapDedupConfig(confirmation_frames=2, tracklet_min_points=2),
        )
        presences = engine.resolve_map_presences(
            0.1,
            {
                "camera-a": {left_a.camera_person_id: left_a, left_b.camera_person_id: left_b},
                "camera-b": {right_a.camera_person_id: right_a, right_b.camera_person_id: right_b},
            },
            self.graph,
            MapDedupConfig(confirmation_frames=2, tracklet_min_points=2),
        )
        presences = engine.resolve_map_presences(
            0.2,
            {
                "camera-a": {left_a.camera_person_id: left_a, left_b.camera_person_id: left_b},
                "camera-b": {right_a.camera_person_id: right_a, right_b.camera_person_id: right_b},
            },
            self.graph,
            MapDedupConfig(confirmation_frames=2, tracklet_min_points=2),
        )

        self.assertEqual(len(presences), 2)
        self.assertEqual(
            sorted(len(presence.source_camera_person_ids) for presence in presences),
            [2, 2],
        )

    def test_boundary_buffer_allows_merge_at_overlap_edge(self) -> None:
        engine = OverlapDedupEngine(OverlapDedupConfig(confirmation_frames=1), ReIDConfig())
        left = _track("camera-a", 1, (1.95, 2.0), [1.0, 0.0, 0.0])
        right = _track("camera-b", 2, (2.15, 2.0), [0.95, 0.05, 0.0])

        engine.resolve_map_presences(
            0.0,
            {"camera-a": {left.camera_person_id: left}, "camera-b": {right.camera_person_id: right}},
            self.graph,
            MapDedupConfig(confirmation_frames=2, tracklet_min_points=2),
        )
        presences = engine.resolve_map_presences(
            0.1,
            {"camera-a": {left.camera_person_id: left}, "camera-b": {right.camera_person_id: right}},
            self.graph,
            MapDedupConfig(confirmation_frames=2, tracklet_min_points=2),
        )
        presences = engine.resolve_map_presences(
            0.2,
            {"camera-a": {left.camera_person_id: left}, "camera-b": {right.camera_person_id: right}},
            self.graph,
            MapDedupConfig(confirmation_frames=2, tracklet_min_points=2),
        )

        self.assertEqual(len(presences), 1)
        self.assertTrue(presences[0].merged_for_counting)

    def test_presence_hold_keeps_single_presence_during_brief_dropout(self) -> None:
        engine = OverlapDedupEngine(OverlapDedupConfig(confirmation_frames=1), ReIDConfig())
        left = _track("camera-a", 1, (2.5, 2.0), [1.0, 0.0, 0.0])
        right = _track("camera-b", 2, (2.55, 2.0), [0.95, 0.05, 0.0])
        config = MapDedupConfig(confirmation_frames=2, tracklet_min_points=2, presence_hold_s=0.30)

        engine.resolve_map_presences(
            0.0,
            {"camera-a": {left.camera_person_id: left}, "camera-b": {right.camera_person_id: right}},
            self.graph,
            config,
        )
        presences = engine.resolve_map_presences(
            0.1,
            {"camera-a": {left.camera_person_id: left}, "camera-b": {right.camera_person_id: right}},
            self.graph,
            config,
        )
        presences = engine.resolve_map_presences(
            0.2,
            {"camera-a": {left.camera_person_id: left}, "camera-b": {right.camera_person_id: right}},
            self.graph,
            config,
        )
        self.assertEqual(len(presences), 1)

    def test_presence_publish_ttl_hides_stale_presence_before_hold_expires(self) -> None:
        engine = OverlapDedupEngine(OverlapDedupConfig(confirmation_frames=1), ReIDConfig())
        left = _track("camera-a", 1, (2.5, 2.0), [1.0, 0.0, 0.0])
        right = _track("camera-b", 2, (2.55, 2.0), [0.95, 0.05, 0.0])
        config = MapDedupConfig(
            confirmation_frames=2,
            tracklet_min_points=2,
            presence_hold_s=0.40,
            presence_publish_ttl_s=0.15,
        )

        engine.resolve_map_presences(
            0.0,
            {"camera-a": {left.camera_person_id: left}, "camera-b": {right.camera_person_id: right}},
            self.graph,
            config,
        )
        engine.resolve_map_presences(
            0.1,
            {"camera-a": {left.camera_person_id: left}, "camera-b": {right.camera_person_id: right}},
            self.graph,
            config,
        )
        presences = engine.resolve_map_presences(
            0.2,
            {"camera-a": {left.camera_person_id: left}, "camera-b": {right.camera_person_id: right}},
            self.graph,
            config,
        )
        self.assertEqual(len(presences), 1)

        presences = engine.resolve_map_presences(
            0.3,
            {},
            self.graph,
            config,
        )
        self.assertEqual(len(presences), 1)

        presences = engine.resolve_map_presences(
            0.36,
            {},
            self.graph,
            config,
        )
        self.assertEqual(len(presences), 0)


if __name__ == "__main__":
    unittest.main()
