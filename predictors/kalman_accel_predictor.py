import numpy as np
from config import KALMAN_MIN_POINTS


class KalmanAccelPredictor:
    """
    Фильтр Калмана для 2D-движения с моделью постоянного ускорения.

    Состояние:
        [x, y, vx, vy, ax, ay]
    """

    def __init__(self):
        self.initialized = False
        self.measurement_count = 0

        # Вектор состояния:
        # x, y, vx, vy, ax, ay
        self.state = np.zeros((6, 1), dtype=float)

        # Ковариация ошибки состояния
        self.P = np.eye(6) * 500.0

        # Шум измерения:
        # считаем, что x и y измеряются с шумом
        self.R = np.array([
            [9.0, 0.0],
            [0.0, 9.0]
        ])

        # Матрица измерения:
        # наблюдаем только x и y
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
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
            [0.0],
            [0.0],
            [0.0]
        ], dtype=float)

        self.P = np.eye(6) * 100.0
        self.last_t = timestamp
        self.initialized = True
        self.measurement_count = 1

    def _build_F(self, dt: float):
        """
        Матрица перехода состояния для модели постоянного ускорения.

        x_new  = x + vx*dt + 0.5*ax*dt^2
        y_new  = y + vy*dt + 0.5*ay*dt^2
        vx_new = vx + ax*dt
        vy_new = vy + ay*dt
        ax_new = ax
        ay_new = ay
        """
        dt2 = dt * dt

        return np.array([
            [1, 0, dt, 0,  0.5 * dt2, 0],
            [0, 1, 0, dt,  0, 0.5 * dt2],
            [0, 0, 1, 0,   dt, 0],
            [0, 0, 0, 1,   0, dt],
            [0, 0, 0, 0,   1, 0],
            [0, 0, 0, 0,   0, 1]
        ], dtype=float)

    def _build_Q(self, dt: float, q: float = 1.0):
        """
        Матрица шума процесса.
        q — интенсивность шума модели.

        Для начала берем простую диагональную форму,
        зависящую от dt.
        """
        dt2 = dt * dt

        return np.diag([
            q * dt2,
            q * dt2,
            q * dt,
            q * dt,
            q,
            q
        ])

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

        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

        self.last_t = timestamp
        self.measurement_count += 1

    def get_current_state(self):
        """
        Возвращает текущую оценку:
        x, y, vx, vy, ax, ay
        """
        return (
            float(self.state[0, 0]),
            float(self.state[1, 0]),
            float(self.state[2, 0]),
            float(self.state[3, 0]),
            float(self.state[4, 0]),
            float(self.state[5, 0]),
        )

    def predict_future(self, delta_t: float):
        """
        Прогноз через delta_t секунд вперед
        без изменения внутреннего состояния фильтра.
        """
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
        self.state = np.zeros((6, 1), dtype=float)
        self.P = np.eye(6) * 500.0
        self.last_t = None