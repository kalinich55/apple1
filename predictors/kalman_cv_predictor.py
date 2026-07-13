import numpy as np
from config import KALMAN_MIN_POINTS


class KalmanCVPredictor:
    """
    Фильтр Калмана для 2D-движения с моделью постоянной скорости.

    Состояние:
        [x, y, vx, vy]
    """

    def __init__(self):
        self.initialized = False
        self.measurement_count = 0

        # Вектор состояния:
        # x, y, vx, vy
        self.state = np.zeros((4, 1), dtype=float)

        # Ковариация ошибки состояния
        self.P = np.eye(4) * 500.0

        # Шум измерения
        self.R = np.array([
            [9.0, 0.0],
            [0.0, 9.0]
        ], dtype=float)

        # Матрица измерения:
        # наблюдаем только x и y
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        self.last_t = None

    def initialize(self, x: float, y: float, timestamp: float):
        """
        Инициализация фильтра первой точкой.
        """
        self.state = np.array([
            [x],
            [y],
            [0.0],
            [0.0]
        ], dtype=float)

        self.P = np.eye(4) * 100.0
        self.last_t = timestamp
        self.initialized = True
        self.measurement_count = 1

    def _build_F(self, dt: float):
        """
        Матрица перехода состояния для модели постоянной скорости.

        x_new  = x + vx*dt
        y_new  = y + vy*dt
        vx_new = vx
        vy_new = vy
        """
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

    def _build_Q(self, dt: float, q: float = 5.0):
        """
        Матрица шума процесса для модели постоянной скорости.
        """
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        return q * np.array([
            [dt4 / 4, 0,       dt3 / 2, 0],
            [0,       dt4 / 4, 0,       dt3 / 2],
            [dt3 / 2, 0,       dt2,     0],
            [0,       dt3 / 2, 0,       dt2]
        ], dtype=float)

    def update(self, x: float, y: float, timestamp: float):
        """
        Полный шаг Калмана:
        1) predict
        2) correct
        """
        if not self.initialized:
            self.initialize(x, y, timestamp)
            return

        dt = timestamp - self.last_t
        if dt <= 0:
            dt = 1e-3
        if dt > 0.1:
            dt = 0.1

        # ---------- Predict ----------
        F = self._build_F(dt)
        Q = self._build_Q(dt, q=8.0)

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

        # ---------- Correct ----------
        z = np.array([[x], [y]], dtype=float)

        y_res = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y_res

        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

        # Ограничение скорости от выбросов
        MAX_VEL = 2000.0
        self.state[2, 0] = np.clip(self.state[2, 0], -MAX_VEL, MAX_VEL)
        self.state[3, 0] = np.clip(self.state[3, 0], -MAX_VEL, MAX_VEL)

        self.last_t = timestamp
        self.measurement_count += 1

    def get_current_state(self):
        """
        Возвращает текущую оценку:
        x, y, vx, vy, ax, ay

        ax и ay возвращаем как 0.0 для совместимости с main.py
        """
        return (
            float(self.state[0, 0]),
            float(self.state[1, 0]),
            float(self.state[2, 0]),
            float(self.state[3, 0]),
            0.0,
            0.0,
        )

    def predict_future(self, delta_t: float):
        """
        Прогноз через delta_t секунд вперед
        без изменения внутреннего состояния фильтра.
        """
        if not self.initialized:
            return 0.0, 0.0

        if delta_t < 0:
            delta_t = 0.0
        if delta_t > 0.2:
            delta_t = 0.2

        F = self._build_F(delta_t)
        future_state = F @ self.state

        x_pred = float(future_state[0, 0])
        y_pred = float(future_state[1, 0])

        return x_pred, y_pred

    def is_ready(self):
        return self.measurement_count >= KALMAN_MIN_POINTS

    def reset(self):
        """
        Сброс фильтра.
        """
        self.initialized = False
        self.measurement_count = 0
        self.state = np.zeros((4, 1), dtype=float)
        self.P = np.eye(4) * 500.0
        self.last_t = None