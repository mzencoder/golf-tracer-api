import unittest
from pathlib import Path

from tracker import Point, build_prediction, select_launch_seed, select_track, track_motion_from_seed


class TrajectoryTests(unittest.TestCase):
    def test_static_markers_do_not_count_as_measured_motion(self):
        detections = [[Point(frame, 100, 500, 0.9)] for frame in range(80)]
        self.assertEqual(select_track(detections, 1280, 720), [])

    def test_moving_ball_is_measured(self):
        detections = [[Point(frame, 80 + 12 * frame, 550, 0.8)] for frame in range(20)]
        self.assertEqual(len(select_track(detections, 1280, 720)), 20)

    def test_late_disappearing_tee_ball_is_seed(self):
        detections = [[] for _ in range(180)]
        for frame in range(15, 75):
            detections[frame].append(Point(frame, 620, 505, 0.7))
        for frame in range(90, 150):
            detections[frame].append(Point(frame, 620, 496, 0.7))
        for frame in range(180):
            detections[frame].append(Point(frame, 1000, 500, 0.9))
        seed = select_launch_seed(detections, 1280, 720, 30)
        self.assertIsNotNone(seed)
        self.assertEqual(seed.frame, 149)

    def test_launch_seed_follows_nearby_ball_across_bins(self):
        detections = [[] for _ in range(190)]
        for frame in range(50, 151):
            detections[frame].append(Point(frame, 620, 504, 0.8))
        for frame in range(151, 171):
            detections[frame].append(Point(frame, 620, 521, 0.8))
        seed = select_launch_seed(detections, 1280, 720, 60)
        self.assertIsNotNone(seed)
        self.assertEqual(seed.frame, 170)

    def test_tee_ball_near_bottom_edge_can_be_launch_seed(self):
        detections = [[] for _ in range(180)]
        for frame in range(20, 61):
            detections[frame].append(Point(frame, 2057, 1968, 0.8))
        seed = select_launch_seed(detections, 4096, 2160, 24)
        self.assertIsNotNone(seed)
        self.assertEqual(seed.frame, 60)
        self.assertEqual(seed.x, 2057)

    def test_estimated_arc_rises_then_falls(self):
        seed = Point(100, 620, 500, 0.8)
        points = build_prediction([], seed, 250, 1280, 720, 60)
        self.assertGreater(len(points), 50)
        self.assertLess(min(p.y for p in points), 260)
        self.assertLess(points[-1].y, seed.y - 40)
        self.assertGreater(points[-1].x, seed.x + 20)
        self.assertLess(points[-1].x, seed.x + 50)
        self.assertAlmostEqual(points[1].x - points[0].x,
                               points[-1].x - points[-2].x, places=4)
        self.assertTrue(all(p.source == "predicted" for p in points))

    def test_short_clip_does_not_force_predicted_landing(self):
        seed = Point(100, 620, 500, 0.8)
        points = build_prediction([], seed, 150, 1280, 720, 60)
        self.assertTrue(points)
        self.assertAlmostEqual(points[-1].y, min(p.y for p in points), delta=2)

    def test_last_measured_velocity_continues_into_descent(self):
        measured = [Point(100 + step, 600 + step * 2, 500 - step * 10, 0.8)
                    for step in range(11)]
        points = build_prediction(measured, None, 240, 1280, 720, 60)
        self.assertTrue(points)
        self.assertLess(points[0].y, measured[-1].y)
        self.assertLess(min(p.y for p in points), points[0].y)
        self.assertGreater(points[-1].y, points[0].y)
        self.assertLess(points[-1].y, measured[0].y - 40)
        self.assertGreater(points[-1].x, measured[-1].x + 120)
        self.assertAlmostEqual(points[0].x - measured[-1].x, 2, places=4)
        self.assertAlmostEqual(points[-1].x - points[-2].x, 2, places=4)

    @unittest.skipUnless((Path(__file__).parent / "sample" / "golf-shot-60fps.mp4").exists(),
                         "Optional local video fixture is unavailable")
    def test_sample_shot_has_visible_post_impact_points(self):
        sample = Path(__file__).parent / "sample" / "golf-shot-60fps.mp4"
        points = track_motion_from_seed(sample, Point(535, 618, 496, 0.75),
                                        59.94, 1280, 720, 709)
        self.assertGreaterEqual(len(points), 20)
        self.assertLess(points[-1].y, 220)
        predicted = build_prediction(points, None, 709, 1280, 720, 59.94)
        self.assertTrue(predicted)
        self.assertAlmostEqual(predicted[1].x - predicted[0].x,
                               predicted[-1].x - predicted[-2].x, places=4)
        self.assertLess(abs(predicted[-1].x - points[-1].x), 35)
        self.assertLess(predicted[-1].y, 410)


if __name__ == "__main__":
    unittest.main()
