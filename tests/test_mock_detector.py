import unittest

from mock_detector import MockDetector


class MockDetectorTests(unittest.TestCase):
    def test_observation_is_not_oracle_truth(self):
        detector = MockDetector(
            trajectory="linear",
            measurement_noise_px=2.0,
            random_seed=7,
            duration_sec=0.1,
        )
        _, detection = detector.get_detection()

        self.assertIsNotNone(detection)
        self.assertNotEqual(detection["x"], detection["true_x"])
        self.assertNotEqual(detection["y"], detection["true_y"])

    def test_linear_object_leaves_frame_instead_of_sticking_to_edge(self):
        detector = MockDetector(
            trajectory="linear",
            width=300,
            height=200,
            radius=20,
            fps=30,
            duration_sec=2.0,
            measurement_noise_px=0.0,
        )
        positions = []
        missed_after_visibility = False
        while True:
            frame, detection = detector.get_detection()
            if frame is None:
                break
            if detection is None:
                missed_after_visibility = bool(positions)
            else:
                positions.append(detection["true_x"])

        self.assertTrue(missed_after_visibility)
        self.assertLess(positions[-1], detector.width - detector.radius)
        self.assertEqual(len(positions), len(set(positions)))

    def test_wall_impact_reverses_horizontal_velocity(self):
        detector = MockDetector(
            trajectory="wall_impact",
            measurement_noise_px=0.0,
            duration_sec=2.0,
        )
        for _ in range(detector.max_frames):
            detector.get_detection()

        self.assertGreaterEqual(detector.sim_bounce_count, 1)
        self.assertLess(detector.sim_vx, 0.0)

    def test_exact_wall_truth_changes_velocity_at_contact(self):
        detector = MockDetector(trajectory="wall_impact", measurement_noise_px=0.0)
        impact_time = (detector.width - detector.radius - 120.0) / 900.0
        before = detector.get_truth_at_timestamp(detector.start_time + impact_time - 0.001)
        at_contact = detector.get_truth_at_timestamp(detector.start_time + impact_time)
        after = detector.get_truth_at_timestamp(detector.start_time + impact_time + 0.001)

        self.assertLess(before[0], at_contact[0])
        self.assertAlmostEqual(at_contact[0], detector.width - detector.radius)
        self.assertLess(after[0], at_contact[0])

    def test_ballistic_object_leaves_frame_without_a_static_floor_tail(self):
        detector = MockDetector(
            trajectory="ballistic",
            measurement_noise_px=0.0,
            duration_sec=6.0,
        )
        y_positions = []
        missed_after_visibility = False
        while True:
            frame, detection = detector.get_detection()
            if frame is None:
                break
            if detection is None:
                missed_after_visibility = bool(y_positions)
            else:
                y_positions.append(detection["true_y"])

        self.assertTrue(missed_after_visibility)
        self.assertLess(y_positions[-1], detector.height - detector.radius)
        self.assertNotEqual(y_positions[-1], y_positions[-2])

    def test_pendulum_stays_on_its_declared_length(self):
        detector = MockDetector(
            trajectory="pendulum",
            measurement_noise_px=0.0,
            duration_sec=0.5,
        )
        pivot_x = detector.width * 0.5
        pivot_y = 90.0
        length = min(180.0, detector.height * 0.34)
        while True:
            frame, detection = detector.get_detection()
            if frame is None:
                break
            self.assertIsNotNone(detection)
            distance = ((detection["true_x"] - pivot_x) ** 2 + (detection["true_y"] - pivot_y) ** 2) ** 0.5
            self.assertAlmostEqual(distance, length, places=6)

    def test_resting_ball_does_not_bounce_every_frame(self):
        detector = MockDetector(
            trajectory="bounce",
            fps=60,
            duration_sec=12.0,
            measurement_noise_px=0.0,
        )
        for _ in range(detector.max_frames):
            detector.get_detection()

        self.assertTrue(detector.resting_on_floor)
        self.assertEqual(detector.sim_vy, 0.0)
        self.assertEqual(detector.sim_y, detector.height - detector.radius)
        self.assertLess(detector.sim_bounce_count, 20)


if __name__ == "__main__":
    unittest.main()
