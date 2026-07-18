import math

from predictors.kalman_accel_predictor import KalmanAccelPredictor


def test_predict_future_stops_overshooting_after_sudden_stop():
    predictor = KalmanAccelPredictor()

    points = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.1),
        (20.0, 0.0, 0.2),
        (20.0, 0.0, 0.3),
    ]

    for x, y, t in points:
        predictor.update(x, y, t)

    x_pred, y_pred = predictor.predict_future(0.08)

    assert math.isfinite(x_pred)
    assert math.isfinite(y_pred)
    assert abs(x_pred - 20.0) < 3.0
    assert abs(y_pred - 0.0) < 1e-6
