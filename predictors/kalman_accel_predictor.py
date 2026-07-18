import numpy as np
from config import KALMAN_MIN_POINTS


class KalmanAccelPredictor:
    """
    Фильтр Калмана для 2D-движения с моделью постоянного ускорения.

    Состояние:
        [x, y, vx, vy, ax, ay]

    Реализация сделана более отзывчивой к новым измерениям и
    с ограничением скорости/ускорения, чтобы прогноз не уходил
    в бесконечный разгон.
    """

    def __init__(self):
        self.initialized = False
        self.measurement_count = 0

        self.state = np.zeros((6, 1), dtype=float)

        self.P = np.eye(6) * 250.0

        self.R = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=float)

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=float)

        self.last_t = None
        self.init_buffer = []
        self.last_measurement = None
        self.last_velocity = None

        self.max_velocity = 2500.0
        self.max_acceleration = 1800.0
        self.accel_damping = 0.65

    def initialize_from_two_points(self, x1, y1, t1, x2, y2, t2):
        dt = t2 - t1
        if dt <= 1e-6:
            dt = 1e-3

        vx0 = (x2 - x1) / dt
        vy0 = (y2 - y1) / dt

        self.state = np.array([
            [x2],
            [y2],
            [vx0],
            [vy0],
            [0.0],
            [0.0]
        ], dtype=float)

        self.P = np.diag([
            40.0,
            40.0,
            200.0,
            200.0,
            800.0,
            800.0,
        ])

        self.last_t = t2
        self.initialized = True
        self.measurement_count = 2

    def _build_F(self, dt: float):
        dt2 = dt * dt

        return np.array([
            [1, 0, dt, 0, 0.5 * dt2, 0],
            [0, 1, 0, dt, 0, 0.5 * dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)

    def _build_Q(self, dt: float):
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        accel_noise = 30.0

        return accel_noise * np.array([
            [dt4 / 4, 0,       dt3 / 2, 0,       dt2 / 2, 0],
            [0,       dt4 / 4, 0,       dt3 / 2, 0,       dt2 / 2],
            [dt3 / 2, 0,       dt2,     0,       dt,     0],
            [0,       dt3 / 2, 0,       dt2,     0,       dt],
            [dt2 / 2, 0,       dt,      0,       1.0,    0],
            [0,       dt2 / 2, 0,       dt,      0,       1.0],
        ], dtype=float)

    def _clamp_state(self):
        self.state[2, 0] = np.clip(self.state[2, 0], -self.max_velocity, self.max_velocity)
        self.state[3, 0] = np.clip(self.state[3, 0], -self.max_velocity, self.max_velocity)
        self.state[4, 0] = np.clip(self.state[4, 0], -self.max_acceleration, self.max_acceleration)
        self.state[5, 0] = np.clip(self.state[5, 0], -self.max_acceleration, self.max_acceleration)

        self.state[4, 0] *= self.accel_damping
        self.state[5, 0] *= self.accel_damping

        # Больше доверяем последней измеренной точке, чем старому инерционному движению
        self.state[2, 0] *= 0.92
        self.state[3, 0] *= 0.92

    def update(self, x: float, y: float, timestamp: float):
        if not self.initialized:
            self.init_buffer.append((float(x), float(y), float(timestamp)))

            if len(self.init_buffer) == 1:
                self.measurement_count = 1
                return

            if len(self.init_buffer) >= 2:
                x1, y1, t1 = self.init_buffer[0]
                x2, y2, t2 = self.init_buffer[1]
                self.initialize_from_two_points(x1, y1, t1, x2, y2, t2)
                return

        dt = timestamp - self.last_t
        if dt <= 0:
            dt = 1e-3

        F = self._build_F(dt)
        Q = self._build_Q(dt)

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

        z = np.array([[x], [y]], dtype=float)

        y_res = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        if self.last_measurement is not None:
            prev_x, prev_y, prev_t = self.last_measurement
            dt_meas = timestamp - prev_t
            if dt_meas > 1e-6:
                obs_vx = (x - prev_x) / dt_meas
                obs_vy = (y - prev_y) / dt_meas
                measured_speed = np.hypot(obs_vx, obs_vy)

                if measured_speed < 2.5:
                    self.state = np.array([[x], [y], [0.0], [0.0], [0.0], [0.0]], dtype=float)
                    self.P = np.diag([25.0, 25.0, 80.0, 80.0, 300.0, 300.0])
                    self.last_measurement = (float(x), float(y), float(timestamp))
                    self.last_t = timestamp
                    self.measurement_count += 1
                    return

        self.state = self.state + K @ y_res
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

        if self.last_measurement is not None:
            prev_x, prev_y, prev_t = self.last_measurement
            dt_meas = timestamp - prev_t
            if dt_meas > 1e-6:
                obs_vx = (x - prev_x) / dt_meas
                obs_vy = (y - prev_y) / dt_meas
                measured_speed = np.hypot(obs_vx, obs_vy)

                if measured_speed < 4.0:
                    self.state[2, 0] = 0.0
                    self.state[3, 0] = 0.0
                    self.state[4, 0] = 0.0
                    self.state[5, 0] = 0.0
                else:
                    if self.last_velocity is not None:
                        prev_vx, prev_vy = self.last_velocity
                        obs_ax = (obs_vx - prev_vx) / max(dt_meas, 1e-3)
                        obs_ay = (obs_vy - prev_vy) / max(dt_meas, 1e-3)
                    else:
                        obs_ax = 0.0
                        obs_ay = 0.0

                    self.state[2, 0] = 0.7 * self.state[2, 0] + 0.3 * obs_vx
                    self.state[3, 0] = 0.7 * self.state[3, 0] + 0.3 * obs_vy
                    self.state[4, 0] = 0.6 * self.state[4, 0] + 0.4 * obs_ax
                    self.state[5, 0] = 0.6 * self.state[5, 0] + 0.4 * obs_ay

                self.last_velocity = (obs_vx, obs_vy)
        else:
            self.last_velocity = None

        self.last_measurement = (float(x), float(y), float(timestamp))

        self._clamp_state()

        self.last_t = timestamp
        self.measurement_count += 1

    def get_current_state(self):
        if not self.initialized:
            if len(self.init_buffer) > 0:
                x, y, _ = self.init_buffer[-1]
                return x, y, 0.0, 0.0, 0.0, 0.0
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        return (
            float(self.state[0, 0]),
            float(self.state[1, 0]),
            float(self.state[2, 0]),
            float(self.state[3, 0]),
            float(self.state[4, 0]),
            float(self.state[5, 0]),
        )

    def predict_future(self, delta_t: float):
        if not self.initialized:
            x, y, _, _, _, _ = self.get_current_state()
            return x, y

        if delta_t < 0:
            delta_t = 0.0

        speed = np.hypot(float(self.state[2, 0]), float(self.state[3, 0]))
        if speed < 8.0:
            x, y, _, _, _, _ = self.get_current_state()
            return x, y

        F = self._build_F(delta_t)
        future_state = F @ self.state

        x_pred = float(future_state[0, 0])
        y_pred = float(future_state[1, 0])

        return x_pred, y_pred

    def is_ready(self):
        return self.measurement_count >= KALMAN_MIN_POINTS

    def reset(self):
        self.initialized = False
        self.measurement_count = 0
        self.state = np.zeros((6, 1), dtype=float)
        self.P = np.eye(6) * 250.0
        self.last_t = None
        self.init_buffer = []
        self.last_measurement = None
        self.last_velocity = None