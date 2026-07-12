from collections import deque
from config import LINEAR_HISTORY_SIZE, LINEAR_MIN_POINTS
from .base_predictor import BasePredictor


class LinearPredictor(BasePredictor):
    def __init__(self):
        self.history = deque(maxlen=LINEAR_HISTORY_SIZE)

    def reset(self):
        self.history.clear()

    def update(self, x, y, t):
        self.history.append((float(x), float(y), float(t)))

    def is_ready(self):
        return len(self.history) >= LINEAR_MIN_POINTS

    def get_current_state(self):
        if len(self.history) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if len(self.history) == 1:
            x, y, _ = self.history[-1]
            return x, y, 0.0, 0.0, 0.0, 0.0

        x1, y1, t1 = self.history[-2]
        x2, y2, t2 = self.history[-1]

        dt = t2 - t1
        if dt <= 1e-6:
            return x2, y2, 0.0, 0.0, 0.0, 0.0

        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt

        return x2, y2, vx, vy, 0.0, 0.0

    def predict_future(self, delta_t):
        x, y, vx, vy, ax, ay = self.get_current_state()
        x_pred = x + vx * delta_t
        y_pred = y + vy * delta_t
        return x_pred, y_pred