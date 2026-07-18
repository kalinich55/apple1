import math
import unittest

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


if __name__ == "__main__":
    unittest.main()
