"""Constant-acceleration Kalman predictor for image-plane coordinates."""

import math

import numpy as np

from config import (
    KALMAN_INNOVATION_GATE,
    KALMAN_JERK_NOISE,
    KALMAN_MEASUREMENT_STD_PX,
    KALMAN_MIN_POINTS,
)


class KalmanAccelPredictor:
    """A 2D constant-acceleration Kalman filter.

    State order is ``[x, y, vx, vy, ax, ay]``.  The implementation uses a
    white-jerk process model and a Joseph covariance update, so its behaviour
    is driven by elapsed time rather than by an arbitrary per-frame damping
    factor.  That matters for the same throw recorded at 30, 60, or 120 FPS.
    """

    def __init__(
        self,
        measurement_std_px: float = KALMAN_MEASUREMENT_STD_PX,
        jerk_noise: float = KALMAN_JERK_NOISE,
        innovation_gate: float = KALMAN_INNOVATION_GATE,
        max_velocity: float = 12_000.0,
        max_acceleration: float = 60_000.0,
    ):
        if measurement_std_px <= 0:
            raise ValueError("measurement_std_px must be positive")
        if jerk_noise <= 0:
            raise ValueError("jerk_noise must be positive")

        self.measurement_std_px = float(measurement_std_px)
        self.jerk_noise = float(jerk_noise)
        self.innovation_gate = float(innovation_gate)
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)

        self.H = np.array(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
        )
        self.R = np.eye(2, dtype=float) * self.measurement_std_px**2
        self.reset()

    def _build_F(self, dt: float) -> np.ndarray:
        dt2 = dt * dt
        return np.array(
            [
                [1.0, 0.0, dt, 0.0, 0.5 * dt2, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, 0.5 * dt2],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def _build_Q(self, dt: float) -> np.ndarray:
        """Discrete white-jerk covariance for each image axis."""
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt
        q_axis = self.jerk_noise * np.array(
            [
                [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
                [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
                [dt3 / 6.0, dt2 / 2.0, dt],
            ],
            dtype=float,
        )

        q = np.zeros((6, 6), dtype=float)
        x_indices = (0, 2, 4)
        y_indices = (1, 3, 5)
        q[np.ix_(x_indices, x_indices)] = q_axis
        q[np.ix_(y_indices, y_indices)] = q_axis
        return q

    def _clamp_state(self) -> None:
        self.state[2, 0] = np.clip(self.state[2, 0], -self.max_velocity, self.max_velocity)
        self.state[3, 0] = np.clip(self.state[3, 0], -self.max_velocity, self.max_velocity)
        self.state[4, 0] = np.clip(
            self.state[4, 0], -self.max_acceleration, self.max_acceleration
        )
        self.state[5, 0] = np.clip(
            self.state[5, 0], -self.max_acceleration, self.max_acceleration
        )

    def _initialize_from_second_measurement(
        self, x: float, y: float, timestamp: float
    ) -> None:
        first_x, first_y, first_t = self._first_measurement
        dt = timestamp - first_t
        if dt <= 0:
            # Keep the newest position but wait for a strictly newer timestamp
            # before deriving a velocity.  Treating dt=0 as 1 ms creates a
            # fictitious, enormous speed.
            self._first_measurement = (x, y, timestamp)
            return

        self.state = np.array(
            [
                [x],
                [y],
                [(x - first_x) / dt],
                [(y - first_y) / dt],
                [0.0],
                [0.0],
            ],
            dtype=float,
        )
        self._clamp_state()
        self.P = np.diag(
            [
                self.measurement_std_px**2,
                self.measurement_std_px**2,
                20_000.0,
                20_000.0,
                250_000.0,
                250_000.0,
            ]
        )
        self.last_t = timestamp
        self.initialized = True
        self.measurement_count = 2
        self._first_measurement = None

    def update(self, x: float, y: float, timestamp: float) -> bool:
        """Assimilate one detector point; return ``False`` if it is rejected."""
        x, y, timestamp = float(x), float(y), float(timestamp)
        if not all(math.isfinite(value) for value in (x, y, timestamp)):
            return False

        if not self.initialized:
            if self._first_measurement is None:
                self._first_measurement = (x, y, timestamp)
                self.state[0, 0] = x
                self.state[1, 0] = y
                self.last_t = timestamp
                self.measurement_count = 1
            else:
                self._initialize_from_second_measurement(x, y, timestamp)
            return self.initialized

        dt = timestamp - self.last_t
        if dt <= 0:
            return False

        f = self._build_F(dt)
        self.state = f @ self.state
        self.P = f @ self.P @ f.T + self._build_Q(dt)
        self._clamp_state()

        z = np.array([[x], [y]], dtype=float)
        innovation = z - self.H @ self.state
        s = self.H @ self.P @ self.H.T + self.R

        try:
            mahalanobis_sq = float((innovation.T @ np.linalg.solve(s, innovation)).item())
        except np.linalg.LinAlgError:
            return False

        self.last_t = timestamp
        if mahalanobis_sq > self.innovation_gate:
            # State and covariance are already propagated to the new time;
            # discard only the implausible measurement.
            return False

        try:
            gain = np.linalg.solve(s, self.H @ self.P).T
        except np.linalg.LinAlgError:
            return False

        self.state = self.state + gain @ innovation
        identity = np.eye(6, dtype=float)
        residual_projection = identity - gain @ self.H
        self.P = (
            residual_projection @ self.P @ residual_projection.T + gain @ self.R @ gain.T
        )
        self.P = 0.5 * (self.P + self.P.T)
        self._clamp_state()
        self.measurement_count += 1
        return True

    def get_current_state(self):
        return tuple(float(value) for value in self.state[:, 0])

    def predict_future(self, delta_t: float):
        if not self.initialized:
            return float(self.state[0, 0]), float(self.state[1, 0])

        delta_t = max(0.0, float(delta_t))
        future_state = self._build_F(delta_t) @ self.state
        return float(future_state[0, 0]), float(future_state[1, 0])

    def is_ready(self):
        return self.measurement_count >= KALMAN_MIN_POINTS and self.initialized

    def reset(self):
        self.initialized = False
        self.measurement_count = 0
        self.state = np.zeros((6, 1), dtype=float)
        self.P = np.eye(6, dtype=float) * 1_000.0
        self.last_t = None
        self._first_measurement = None
