"""Deterministic synthetic detector for trajectory-prediction experiments."""

import math
import time

import cv2
import numpy as np

from config import GRAVITY_PX


class MockDetector:
    """Render a known trajectory and return a noisy detector observation.

    The rendered position is the physical ground truth.  ``x``/``y`` in the
    returned detection intentionally contain a simulated segmentation error;
    only ``true_x``/``true_y`` may be used by the evaluator.
    """

    def __init__(
        self,
        trajectory="linear",
        width=960,
        height=540,
        fps=30,
        duration_sec=60.0,
        radius=20,
        measurement_noise_px=1.5,
        dropout_probability=0.0,
        random_seed=42,
        debug_timing=False,
    ):
        if fps <= 0:
            raise ValueError("fps must be positive")
        if measurement_noise_px < 0:
            raise ValueError("measurement_noise_px must not be negative")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("dropout_probability must be in [0, 1)")

        self.trajectory = trajectory
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.duration_sec = float(duration_sec)
        self.radius = int(radius)
        self.measurement_noise_px = float(measurement_noise_px)
        self.dropout_probability = float(dropout_probability)
        self.random_seed = random_seed
        self.debug_timing = bool(debug_timing)

        self.frame_index = 0
        self.max_frames = int(round(self.duration_sec * self.fps))
        self.start_time = time.perf_counter()
        self._rng = np.random.default_rng(self.random_seed)

        self.sim_initialized = False
        self.sim_x = None
        self.sim_y = None
        self.sim_vx = None
        self.sim_vy = None
        self.sim_bounce_count = 0
        self.resting_on_floor = False

        print(
            f"[MockDetector] trajectory={self.trajectory}, fps={self.fps}, "
            f"duration={self.duration_sec}s, frames={self.max_frames}, "
            f"noise={self.measurement_noise_px}px, dropout={self.dropout_probability:.0%}, "
            f"size={self.width}x{self.height}"
        )

    def reset(self):
        self.frame_index = 0
        self.start_time = time.perf_counter()
        self._rng = np.random.default_rng(self.random_seed)
        self.sim_initialized = False
        self.sim_x = self.sim_y = None
        self.sim_vx = self.sim_vy = None
        self.sim_bounce_count = 0
        self.resting_on_floor = False

    def get_detection(self, conf_threshold=0.6):
        if self.frame_index >= self.max_frames:
            return None, None

        started_at = time.perf_counter()
        sim_time = self.frame_index * self.dt
        x, y, visible = self._get_position(sim_time)
        timestamp = self.start_time + sim_time
        self.frame_index += 1

        frame = np.full((self.height, self.width, 3), 35, dtype=np.uint8)
        if not visible:
            return frame, None

        truth_center = (int(round(x)), int(round(y)))
        cv2.circle(frame, truth_center, self.radius, (0, 255, 255), -1)

        if self._rng.random() < self.dropout_probability:
            return frame, None

        observed_x = float(x + self._rng.normal(0.0, self.measurement_noise_px))
        observed_y = float(y + self._rng.normal(0.0, self.measurement_noise_px))

        contour = self._make_circle_contour(*truth_center, self.radius)
        x1 = max(0, int(round(x - self.radius)))
        y1 = max(0, int(round(y - self.radius)))
        x2 = min(self.width - 1, int(round(x + self.radius)))
        y2 = min(self.height - 1, int(round(y + self.radius)))

        detection = {
            "timestamp": timestamp,
            "sim_time": sim_time,
            "x": observed_x,
            "y": observed_y,
            "confidence": 0.99,
            "bbox": (x1, y1, x2, y2),
            "contour": contour,
            "true_x": float(x),
            "true_y": float(y),
        }

        if self.debug_timing:
            elapsed = time.perf_counter() - started_at
            fps_now = 1.0 / elapsed if elapsed > 1e-6 else 0.0
            print(f"[MockDetector] frame_time={elapsed:.6f}s | fps={fps_now:.2f}")

        return frame, detection

    def _make_circle_contour(self, cx, cy, radius, points=24):
        contour_points = []
        for index in range(points):
            angle = 2.0 * math.pi * index / points
            px = int(round(cx + radius * math.cos(angle)))
            py = int(round(cy + radius * math.sin(angle)))
            contour_points.append([px, py])
        return np.array(contour_points, dtype=np.int32).reshape(-1, 1, 2)

    def _inside_frame(self, x, y):
        return (
            self.radius <= x <= self.width - self.radius
            and self.radius <= y <= self.height - self.radius
        )

    def _init_bounce_state(self):
        self.sim_x = 120.0
        self.sim_y = self.height - 145.0
        self.sim_vx = 265.0
        self.sim_vy = -195.0
        self.sim_bounce_count = 0
        self.resting_on_floor = False
        self.sim_initialized = True

    def _init_wall_impact_state(self):
        self.sim_x = 120.0
        self.sim_y = 150.0
        self.sim_vx = 900.0
        self.sim_vy = 0.0
        self.sim_bounce_count = 0
        self.resting_on_floor = False
        self.sim_initialized = True

    def _linear_position(self, sim_time):
        x = 90.0 + 220.0 * sim_time
        y = self.height * 0.45
        return x, y, self._inside_frame(x, y)

    def _ballistic_position(self, sim_time):
        x = 90.0 + 210.0 * sim_time
        y = self.height - 95.0 - 455.0 * sim_time + 0.5 * GRAVITY_PX * sim_time**2
        return x, y, self._inside_frame(x, y)

    def _pendulum_position(self, sim_time):
        pivot_x = self.width * 0.5
        pivot_y = 90.0
        length = min(180.0, self.height * 0.34)
        amplitude = 0.46
        omega = 1.60
        damping = 0.02
        theta = amplitude * math.exp(-damping * sim_time) * math.sin(omega * sim_time)
        x = pivot_x + length * math.sin(theta)
        y = pivot_y + length * math.cos(theta)
        return x, y, self._inside_frame(x, y)

    def _wall_impact_position(self, sim_time):
        """Analytical free flight with one collision against the right wall."""
        x0, y0 = 120.0, 150.0
        vx0 = 900.0
        right_wall_x = self.width - self.radius
        restitution = 0.84
        impact_time = (right_wall_x - x0) / vx0

        if sim_time < impact_time:
            x = x0 + vx0 * sim_time
            vx = vx0
            impacts = 0
        else:
            x = right_wall_x - restitution * vx0 * (sim_time - impact_time)
            vx = -restitution * vx0
            impacts = 1

        y = y0 + 0.5 * GRAVITY_PX * sim_time**2
        vy = GRAVITY_PX * sim_time
        return x, y, vx, vy, impacts, self._inside_frame(x, y)

    def get_truth_at_timestamp(self, timestamp):
        """Return exact truth at an absolute mock timestamp when available.

        The closed-box bounce is stateful and therefore uses the evaluator's
        frame interpolation. The three analytical trajectories and the single
        wall impact can be evaluated exactly at any target time.
        """
        sim_time = float(timestamp) - self.start_time
        if sim_time < 0.0 or sim_time > self.duration_sec:
            return None

        if self.trajectory == "linear":
            return self._linear_position(sim_time)
        if self.trajectory == "ballistic":
            return self._ballistic_position(sim_time)
        if self.trajectory == "pendulum":
            return self._pendulum_position(sim_time)
        if self.trajectory == "wall_impact":
            x, y, _, _, _, visible = self._wall_impact_position(sim_time)
            return x, y, visible
        return None

    def get_collision_environment(self):
        """Return known physical walls for the current mock scenario."""
        if self.trajectory == "wall_impact":
            return {"right": 0.84}
        if self.trajectory == "bounce":
            return {
                "left": 0.84,
                "right": 0.84,
                "top": 0.84,
                "bottom": 0.68,
            }
        return {}

    def _simulate_bounce_step(self):
        """Advance a ball in a closed box, including a stable resting state."""
        dt = self.dt
        right_wall_x = self.width - self.radius
        left_wall_x = self.radius
        floor_y = self.height - self.radius
        ceiling_y = self.radius
        wall_restitution = 0.84
        floor_restitution = 0.68
        floor_friction = 0.975
        resting_speed = 28.0

        if self.resting_on_floor:
            self.sim_x += self.sim_vx * dt
            if self.sim_x >= right_wall_x:
                self.sim_x = right_wall_x
                self.sim_vx = -abs(self.sim_vx) * wall_restitution
            elif self.sim_x <= left_wall_x:
                self.sim_x = left_wall_x
                self.sim_vx = abs(self.sim_vx) * wall_restitution
            self.sim_vx *= floor_friction
            if abs(self.sim_vx) < 8.0:
                self.sim_vx = 0.0
            self.sim_y = floor_y
            self.sim_vy = 0.0
            return

        self.sim_vy += GRAVITY_PX * dt
        new_x = self.sim_x + self.sim_vx * dt
        new_y = self.sim_y + self.sim_vy * dt

        if new_x >= right_wall_x:
            new_x = right_wall_x
            self.sim_vx = -abs(self.sim_vx) * wall_restitution
        elif new_x <= left_wall_x:
            new_x = left_wall_x
            self.sim_vx = abs(self.sim_vx) * wall_restitution

        if new_y <= ceiling_y:
            new_y = ceiling_y
            self.sim_vy = abs(self.sim_vy) * wall_restitution

        if new_y >= floor_y:
            new_y = floor_y
            impact_speed = abs(self.sim_vy)
            self.sim_vx *= floor_friction
            self.sim_bounce_count += 1
            if impact_speed < resting_speed:
                self.sim_vy = 0.0
                self.resting_on_floor = True
            else:
                self.sim_vy = -impact_speed * floor_restitution

        self.sim_x = new_x
        self.sim_y = new_y

    def _simulate_wall_impact_step(self):
        """Free flight with one vertical wall; no artificial floor collision."""
        self.sim_vy += GRAVITY_PX * self.dt
        self.sim_x += self.sim_vx * self.dt
        self.sim_y += self.sim_vy * self.dt

        right_wall_x = self.width - self.radius
        if self.sim_x >= right_wall_x:
            self.sim_x = right_wall_x
            self.sim_vx = -abs(self.sim_vx) * 0.84
            self.sim_bounce_count += 1

    def _get_position(self, sim_time):
        if self.trajectory == "linear":
            return self._linear_position(sim_time)

        if self.trajectory == "ballistic":
            return self._ballistic_position(sim_time)

        if self.trajectory == "pendulum":
            return self._pendulum_position(sim_time)

        if self.trajectory == "bounce":
            if not self.sim_initialized:
                self._init_bounce_state()
            elif self.frame_index > 0:
                self._simulate_bounce_step()
            return self.sim_x, self.sim_y, True

        if self.trajectory == "wall_impact":
            x, y, vx, vy, impacts, visible = self._wall_impact_position(sim_time)
            self.sim_initialized = True
            self.sim_x = x
            self.sim_y = y
            self.sim_vx = vx
            self.sim_vy = vy
            self.sim_bounce_count = impacts
            return x, y, visible

        raise ValueError(f"Unknown mock trajectory: {self.trajectory}")

    def draw_detection(self, frame, detection):
        if detection is None:
            return frame

        x = int(round(detection["x"]))
        y = int(round(detection["y"]))
        confidence = detection["confidence"]
        x1, y1, x2, y2 = detection["bbox"]
        contour = detection.get("contour")

        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (255, 100, 0), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)

        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"Mock {confidence:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 100, 0),
            2,
        )
        cv2.putText(
            frame,
            f"traj={self.trajectory}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 255, 200),
            2,
        )

        if self.trajectory in {"bounce", "wall_impact"} and self.sim_initialized:
            cv2.putText(
                frame,
                f"vx={self.sim_vx:.1f} vy={self.sim_vy:.1f} impacts={self.sim_bounce_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (180, 220, 255),
                2,
            )
        return frame

    def release(self):
        cv2.destroyAllWindows()
