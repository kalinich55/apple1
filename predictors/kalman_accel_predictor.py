"""Constant-acceleration Kalman predictor for image-plane coordinates."""

from collections import deque
import math

import numpy as np

from config import (
    KALMAN_INNOVATION_GATE,
    KALMAN_JERK_NOISE,
    KALMAN_FORECAST_NOISE_MARGIN_SIGMA,
    KALMAN_FORECAST_SPEED_FACTOR,
    KALMAN_MEASUREMENT_STD_PX,
    KALMAN_MOTION_HISTORY,
    KALMAN_REACQUIRE_AFTER_REJECTIONS,
    KALMAN_REACQUIRE_VELOCITY_BLEND,
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
        reacquire_after_rejections: int = KALMAN_REACQUIRE_AFTER_REJECTIONS,
        motion_history: int = KALMAN_MOTION_HISTORY,
        forecast_speed_factor: float = KALMAN_FORECAST_SPEED_FACTOR,
        forecast_noise_margin_sigma: float = KALMAN_FORECAST_NOISE_MARGIN_SIGMA,
        reacquire_velocity_blend: float = KALMAN_REACQUIRE_VELOCITY_BLEND,
        max_velocity: float = 12_000.0,
        max_acceleration: float = 60_000.0,
    ):
        if measurement_std_px <= 0:
            raise ValueError("measurement_std_px must be positive")
        if jerk_noise <= 0:
            raise ValueError("jerk_noise must be positive")
        if reacquire_after_rejections < 1:
            raise ValueError("reacquire_after_rejections must be positive")
        if motion_history < 2:
            raise ValueError("motion_history must be at least 2")
        if forecast_speed_factor <= 0 or forecast_noise_margin_sigma < 0:
            raise ValueError("forecast guard parameters must be non-negative")
        if not 0.0 <= reacquire_velocity_blend <= 1.0:
            raise ValueError("reacquire_velocity_blend must be between 0 and 1")

        self.measurement_std_px = float(measurement_std_px)
        self.jerk_noise = float(jerk_noise)
        self.innovation_gate = float(innovation_gate)
        self.reacquire_after_rejections = int(reacquire_after_rejections)
        self.motion_history = int(motion_history)
        self.forecast_speed_factor = float(forecast_speed_factor)
        self.forecast_noise_margin_sigma = float(forecast_noise_margin_sigma)
        self.reacquire_velocity_blend = float(reacquire_velocity_blend)
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
        self.last_accepted_measurement = (x, y, timestamp)
        self.recent_measurements.append((x, y, timestamp))
        self.consecutive_rejections = 0

    def _reacquire_from_measurement(self, x: float, y: float, timestamp: float) -> bool:
        """Recover after a sustained real manoeuvre or collision.

        Innovation gating is useful for one-off segmentation glitches, but a
        ball that reverses at a wall otherwise gets rejected forever because
        the stale state keeps moving away from it. Two consecutive rejected
        points are enough to estimate the direction of the new trajectory.
        """
        if len(self.rejected_measurements) < 2:
            return False

        previous_x, previous_y, previous_t = self.rejected_measurements[-2]
        latest_x, latest_y, latest_t = self.rejected_measurements[-1]
        dt = latest_t - previous_t
        if dt <= 1e-6:
            return False

        observed_vx = (latest_x - previous_x) / dt
        observed_vy = (latest_y - previous_y) / dt
        old_vx = float(self.state[2, 0])
        old_vy = float(self.state[3, 0])
        blend = self.reacquire_velocity_blend
        vx = (1.0 - blend) * old_vx + blend * observed_vx
        vy = (1.0 - blend) * old_vy + blend * observed_vy

        # A collision is an impulse: it can change velocity almost
        # instantaneously, but it normally does not cancel persistent forces.
        # For example, after a bounce the vertical velocity reverses while
        # image-plane gravity still accelerates the ball downwards.  Keep the
        # previous acceleration as a deliberately uncertain prior instead of
        # forcing the first forecast after reacquisition to be linear.
        acceleration_prior = np.nan_to_num(
            self.state[4:6, 0],
            nan=0.0,
            posinf=self.max_acceleration,
            neginf=-self.max_acceleration,
        )
        self.state = np.array(
            [[x], [y], [vx], [vy], [acceleration_prior[0]], [acceleration_prior[1]]],
            dtype=float,
        )
        self._clamp_state()
        self.P = np.diag(
            [
                self.measurement_std_px**2,
                self.measurement_std_px**2,
                30_000.0,
                30_000.0,
                300_000.0,
                300_000.0,
            ]
        )
        self.last_t = timestamp
        self.last_accepted_measurement = (x, y, timestamp)
        self.recent_measurements.clear()
        self.recent_measurements.extend(self.rejected_measurements)
        self.rejected_measurements.clear()
        self.consecutive_rejections = 0
        self.measurement_count = max(KALMAN_MIN_POINTS, self.measurement_count + 1)
        return True

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
                self.last_accepted_measurement = (x, y, timestamp)
                self.recent_measurements.append((x, y, timestamp))
                self.consecutive_rejections = 0
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
            # discard only the implausible measurement. A sustained sequence
            # is treated as a real manoeuvre rather than permanent noise.
            self.consecutive_rejections += 1
            self.rejected_measurements.append((x, y, timestamp))
            if self.consecutive_rejections >= self.reacquire_after_rejections:
                return self._reacquire_from_measurement(x, y, timestamp)
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
        self.last_accepted_measurement = (x, y, timestamp)
        self.recent_measurements.append((x, y, timestamp))
        self.rejected_measurements.clear()
        self.consecutive_rejections = 0
        return True

    def _recent_observed_speed(self) -> tuple[float, float]:
        """Return a stable local speed estimate and its observation span.

        A least-squares slope over several detector centres is much less
        sensitive to a one-pixel contour change than a two-point derivative.
        """
        if len(self.recent_measurements) < 2:
            return 0.0, 0.0

        samples = list(self.recent_measurements)
        latest_t = samples[-1][2]
        times = np.array([sample[2] - latest_t for sample in samples], dtype=float)
        span = float(times[-1] - times[0])
        if span <= 1e-6:
            return 0.0, 0.0

        design = np.column_stack((times, np.ones_like(times)))
        xs = np.array([sample[0] for sample in samples], dtype=float)
        ys = np.array([sample[1] for sample in samples], dtype=float)
        vx = float(np.linalg.lstsq(design, xs, rcond=None)[0][0])
        vy = float(np.linalg.lstsq(design, ys, rcond=None)[0][0])
        return math.hypot(vx, vy), span

    def get_current_state(self):
        return tuple(float(value) for value in self.state[:, 0])

    def predict_future(self, delta_t: float):
        if not self.initialized:
            return float(self.state[0, 0]), float(self.state[1, 0])

        delta_t = max(0.0, float(delta_t))
        future_state = self._build_F(delta_t) @ self.state
        current_position = self.state[:2, 0]
        forecast_lead = future_state[:2, 0] - current_position

        # A noisy contour can make the internally estimated acceleration very
        # large for one frame. Bound only the *forecast lead* using motion that
        # was actually observed over a short window. The confidence ramp keeps
        # the first two noisy points from producing a long prediction ray.
        observed_speed, observed_span = self._recent_observed_speed()
        history_confidence = min(1.0, observed_span / 0.12)
        if self.consecutive_rejections:
            # While the newest centre is statistically suspicious, do not
            # extend the old trajectory aggressively. If it is a real turn,
            # reacquire will establish the new direction on the next coherent
            # point; if it is a detector glitch, the state remains untouched.
            history_confidence = 0.0
        max_lead = (
            self.forecast_speed_factor
            * observed_speed
            * delta_t
            * history_confidence
            + self.forecast_noise_margin_sigma * self.measurement_std_px
        )
        lead_length = float(np.linalg.norm(forecast_lead))
        if lead_length > max_lead:
            if max_lead <= 0.0:
                forecast_lead[:] = 0.0
            else:
                forecast_lead *= max_lead / lead_length

        prediction = current_position + forecast_lead
        return float(prediction[0]), float(prediction[1])

    def is_ready(self):
        return self.measurement_count >= KALMAN_MIN_POINTS and self.initialized

    def reset(self):
        self.initialized = False
        self.measurement_count = 0
        self.state = np.zeros((6, 1), dtype=float)
        self.P = np.eye(6, dtype=float) * 1_000.0
        self.last_t = None
        self._first_measurement = None
        self.last_accepted_measurement = None
        self.recent_measurements = deque(maxlen=self.motion_history)
        self.rejected_measurements = deque(
            maxlen=max(self.motion_history, self.reacquire_after_rejections)
        )
        self.consecutive_rejections = 0
