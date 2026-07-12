import numpy as np
from collections import deque
from config import POLY2_HISTORY_SIZE, POLY2_MIN_POINTS
from .base_predictor import BasePredictor


class Poly2Predictor(BasePredictor):
    def __init__(self):
        self.history = deque(maxlen=POLY2_HISTORY_SIZE)

    def reset(self):
        self.history.clear()

    def update(self, x, y, t):
        self.history.append((float(x), float(y), float(t)))

    def is_ready(self):
        return len(self.history) >= POLY2_MIN_POINTS

    def _fit(self):
        if len(self.history) < POLY2_MIN_POINTS:
            return None

        data = list(self.history)
        t0 = data[0][2]

        ts = np.array([p[2] - t0 for p in data], dtype=np.float64)
        xs = np.array([p[0] for p in data], dtype=np.float64)
        ys = np.array([p[1] for p in data], dtype=np.float64)

        coef_x = np.polyfit(ts, xs, 2)
        coef_y = np.polyfit(ts, ys, 2)

        return t0, coef_x, coef_y

    def get_current_state(self):
        if len(self.history) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if len(self.history) < POLY2_MIN_POINTS:
            x, y, _ = self.history[-1]
            return x, y, 0.0, 0.0, 0.0, 0.0

        fitted = self._fit()
        if fitted is None:
            x, y, _ = self.history[-1]
            return x, y, 0.0, 0.0, 0.0, 0.0

        t0, coef_x, coef_y = fitted
        t_now = self.history[-1][2] - t0

        x = np.polyval(coef_x, t_now)
        y = np.polyval(coef_y, t_now)

        dcoef_x = np.polyder(coef_x, 1)
        dcoef_y = np.polyder(coef_y, 1)
        ddcoef_x = np.polyder(coef_x, 2)
        ddcoef_y = np.polyder(coef_y, 2)

        vx = np.polyval(dcoef_x, t_now)
        vy = np.polyval(dcoef_y, t_now)
        ax = np.polyval(ddcoef_x, t_now)
        ay = np.polyval(ddcoef_y, t_now)

        return float(x), float(y), float(vx), float(vy), float(ax), float(ay)

    def predict_future(self, delta_t):
        if len(self.history) == 0:
            return 0.0, 0.0

        if len(self.history) < POLY2_MIN_POINTS:
            x, y, _, _, _, _ = self.get_current_state()
            return x, y

        fitted = self._fit()
        if fitted is None:
            x, y, _, _, _, _ = self.get_current_state()
            return x, y

        t0, coef_x, coef_y = fitted
        future_t = self.history[-1][2] + delta_t
        tf = future_t - t0

        x_pred = np.polyval(coef_x, tf)
        y_pred = np.polyval(coef_y, tf)

        return float(x_pred), float(y_pred)