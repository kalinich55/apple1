import math
import unittest

from mock_detector import MockDetector
from predictors.kalman_accel_predictor import KalmanAccelPredictor


class KalmanAccelPredictorTests(unittest.TestCase):
    def test_predict_future_stops_overshooting_after_sudden_stop(self):
        predictor = KalmanAccelPredictor()
        points = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.1),
            (20.0, 0.0, 0.2),
            (20.0, 0.0, 0.3),
        ]

        for x, y, timestamp in points:
            predictor.update(x, y, timestamp)

        x_pred, y_pred = predictor.predict_future(0.08)
        self.assertTrue(math.isfinite(x_pred))
        self.assertTrue(math.isfinite(y_pred))
        self.assertLess(abs(x_pred - 20.0), 3.0)
        self.assertLess(abs(y_pred), 1e-6)

    def test_duplicate_timestamp_does_not_create_extreme_velocity(self):
        predictor = KalmanAccelPredictor()
        predictor.update(0.0, 0.0, 1.0)
        self.assertFalse(predictor.update(10.0, 0.0, 1.0))
        self.assertTrue(predictor.update(10.0, 0.0, 1.1))

        _, _, vx, _, _, _ = predictor.get_current_state()
        self.assertLess(abs(vx), 1_000.0)

    def test_linear_motion_is_consistent_across_frame_rates(self):
        errors = []
        for fps in (15, 30, 60):
            predictor = KalmanAccelPredictor()
            dt = 1.0 / fps
            for index in range(1 + int(1.0 / dt)):
                timestamp = index * dt
                predictor.update(220.0 * timestamp, -40.0 * timestamp, timestamp)

            x_pred, y_pred = predictor.predict_future(0.08)
            errors.append(math.hypot(x_pred - 220.0 * 1.08, y_pred + 40.0 * 1.08))

        self.assertLess(max(errors), 1.0)

    def test_short_ballistic_forecast_is_accurate(self):
        predictor = KalmanAccelPredictor()
        dt = 1.0 / 30.0
        errors = []
        for index in range(60):
            timestamp = index * dt
            x = 210.0 * timestamp
            y = 300.0 - 455.0 * timestamp + 0.5 * 500.0 * timestamp**2
            predictor.update(x, y, timestamp)
            if predictor.is_ready():
                x_pred, y_pred = predictor.predict_future(0.08)
                target_time = timestamp + 0.08
                true_x = 210.0 * target_time
                true_y = 300.0 - 455.0 * target_time + 0.5 * 500.0 * target_time**2
                errors.append(math.hypot(x_pred - true_x, y_pred - true_y))

        self.assertLess(sum(errors) / len(errors), 3.0)

    def test_reacquires_after_a_sudden_direction_reversal(self):
        predictor = KalmanAccelPredictor()
        detector = MockDetector(
            trajectory="bounce",
            duration_sec=1.4,
            fps=30,
            measurement_noise_px=0.0,
        )
        saw_rejection = False
        recovered_upward = False

        while True:
            frame, detection = detector.get_detection()
            if frame is None:
                break

            accepted = predictor.update(
                detection["x"], detection["y"], detection["timestamp"]
            )
            if detection["sim_time"] >= 1.22 and not accepted:
                saw_rejection = True
            if saw_rejection and accepted and predictor.is_ready():
                _, _, _, vy, _, _ = predictor.get_current_state()
                recovered_upward = vy < 0.0

        self.assertTrue(saw_rejection)
        self.assertTrue(recovered_upward)

    def test_reacquire_keeps_the_estimated_ballistic_acceleration(self):
        predictor = KalmanAccelPredictor()
        dt = 1.0 / 30.0

        # Let the filter estimate a stable gravity-like vertical acceleration.
        for index in range(45):
            timestamp = index * dt
            predictor.update(200.0 * timestamp, 0.5 * 500.0 * timestamp**2, timestamp)

        _, _, _, _, _, ay_before = predictor.get_current_state()
        self.assertGreater(ay_before, 350.0)

        # Two far-away points trigger reacquisition, as can happen after a
        # collision or a sharp manoeuvre that crosses the innovation gate.
        self.assertFalse(predictor.update(10_000.0, 600.0, 45 * dt))
        self.assertTrue(predictor.update(10_100.0, 620.0, 46 * dt))

        _, _, _, _, _, ay_after = predictor.get_current_state()
        self.assertAlmostEqual(ay_after, ay_before, delta=1e-6)

    def test_small_noisy_motion_does_not_launch_the_forecast_sideways(self):
        predictor = KalmanAccelPredictor()
        dt = 1.0 / 30.0
        points = [
            (400.0, 300.0),
            (401.5, 299.0),
            (398.5, 301.0),
            (402.0, 300.5),
            (405.0, 299.5),
        ]

        for index, (x, y) in enumerate(points):
            predictor.update(x, y, index * dt)

        current_x, current_y, *_ = predictor.get_current_state()
        predicted_x, predicted_y = predictor.predict_future(0.08)
        lead = math.hypot(predicted_x - current_x, predicted_y - current_y)

        self.assertLess(lead, 30.0)

    def test_harmonic_motion_forecast_has_no_large_single_frame_spikes(self):
        predictor = KalmanAccelPredictor()
        dt = 1.0 / 30.0
        previous_prediction = None
        largest_step = 0.0

        for index in range(120):
            timestamp = index * dt
            x = 500.0 + 120.0 * math.sin(2.0 * math.pi * timestamp / 1.6)
            # Deterministic contour jitter similar to a moving segmentation mask.
            x += (index % 3 - 1) * 2.0
            predictor.update(x, 300.0, timestamp)
            if not predictor.is_ready():
                continue
            prediction = predictor.predict_future(0.08)
            if previous_prediction is not None:
                largest_step = max(
                    largest_step,
                    math.hypot(
                        prediction[0] - previous_prediction[0],
                        prediction[1] - previous_prediction[1],
                    ),
                )
            previous_prediction = prediction

        self.assertLess(largest_step, 45.0)


if __name__ == "__main__":
    unittest.main()
