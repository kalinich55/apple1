"""Kalman prediction with optional reflection from known image-plane walls."""

import math

from config import (
    COLLISION_DEFAULT_RESTITUTION,
    COLLISION_MAX_EVENTS,
    COLLISION_REAL_ENABLED,
)
from .base_predictor import BasePredictor
from .kalman_accel_predictor import KalmanAccelPredictor


class CollisionAwarePredictor(BasePredictor):
    """Constant-acceleration Kalman predictor with known-wall reflections.

    It is intentionally conservative: unless an environment and explicit
    active walls are supplied, it behaves exactly like ``kalman_ca``. Camera
    image borders are not automatically assumed to be physical walls.
    """

    def __init__(self, base_predictor=None):
        self.base = base_predictor or KalmanAccelPredictor()
        self.width = None
        self.height = None
        self.radius_px = 0.0
        self.walls = {}
        self.enabled = COLLISION_REAL_ENABLED

    def set_environment(self, width, height, radius_px, *, walls=None, enabled=None):
        """Set image-plane collision geometry for subsequent forecasts.

        ``walls`` maps any of ``left``, ``right``, ``top`` and ``bottom`` to
        its restitution coefficient. Passing an empty mapping disables
        collisions even when dimensions are known.
        """
        self.width = float(width)
        self.height = float(height)
        self.radius_px = max(0.0, float(radius_px))
        self.walls = {
            name: float(restitution)
            for name, restitution in (walls or {}).items()
            if name in {"left", "right", "top", "bottom"}
        }
        if enabled is not None:
            self.enabled = bool(enabled)

    def update(self, x, y, t):
        return self.base.update(x, y, t)

    def get_current_state(self):
        return self.base.get_current_state()

    def is_ready(self):
        return self.base.is_ready()

    def reset(self):
        self.base.reset()

    @staticmethod
    def _propagate(x, y, vx, vy, ax, ay, dt):
        return (
            x + vx * dt + 0.5 * ax * dt * dt,
            y + vy * dt + 0.5 * ay * dt * dt,
            vx + ax * dt,
            vy + ay * dt,
        )

    @staticmethod
    def _root_times(position, velocity, acceleration, boundary):
        """Return positive roots of position(t) == boundary."""
        a = 0.5 * acceleration
        b = velocity
        c = position - boundary
        epsilon = 1e-9

        if abs(a) < epsilon:
            if abs(b) < epsilon:
                return []
            return [-c / b]

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return []
        root = math.sqrt(max(0.0, discriminant))
        return [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]

    def _next_collision(self, x, y, vx, vy, ax, ay, remaining):
        left = self.radius_px
        right = self.width - self.radius_px
        top = self.radius_px
        bottom = self.height - self.radius_px
        candidates = []
        epsilon = 1e-7

        def add_candidates(axis, boundary, wall_name, position, velocity, acceleration, direction):
            if wall_name not in self.walls:
                return
            for hit_time in self._root_times(position, velocity, acceleration, boundary):
                if not epsilon < hit_time <= remaining + epsilon:
                    continue
                impact_velocity = velocity + acceleration * hit_time
                if direction * impact_velocity > epsilon:
                    candidates.append((hit_time, axis, wall_name, boundary))

        add_candidates("x", left, "left", x, vx, ax, -1.0)
        add_candidates("x", right, "right", x, vx, ax, 1.0)
        add_candidates("y", top, "top", y, vy, ay, -1.0)
        add_candidates("y", bottom, "bottom", y, vy, ay, 1.0)
        return min(candidates, key=lambda item: item[0]) if candidates else None

    def _resolve_contact(self, x, y, vx, vy):
        """Reflect an already outward-moving state located on a wall."""
        epsilon = 1e-6
        left = self.radius_px
        right = self.width - self.radius_px
        top = self.radius_px
        bottom = self.height - self.radius_px

        if "left" in self.walls and x <= left + epsilon and vx < 0.0:
            x, vx = left, -vx * self.walls["left"]
        elif "right" in self.walls and x >= right - epsilon and vx > 0.0:
            x, vx = right, -vx * self.walls["right"]

        if "top" in self.walls and y <= top + epsilon and vy < 0.0:
            y, vy = top, -vy * self.walls["top"]
        elif "bottom" in self.walls and y >= bottom - epsilon and vy > 0.0:
            y, vy = bottom, -vy * self.walls["bottom"]
        return x, y, vx, vy

    def predict_future(self, delta_t):
        if not self.is_ready() or not self.enabled or not self.walls:
            return self.base.predict_future(delta_t)
        if self.width is None or self.height is None:
            return self.base.predict_future(delta_t)

        remaining = max(0.0, float(delta_t))
        x, y, vx, vy, ax, ay = self.base.get_current_state()
        x, y, vx, vy = self._resolve_contact(x, y, vx, vy)

        for _ in range(COLLISION_MAX_EVENTS):
            collision = self._next_collision(x, y, vx, vy, ax, ay, remaining)
            if collision is None:
                x, y, _, _ = self._propagate(x, y, vx, vy, ax, ay, remaining)
                return float(x), float(y)

            hit_time, axis, wall_name, boundary = collision
            hit_time = min(remaining, hit_time)
            x, y, vx, vy = self._propagate(x, y, vx, vy, ax, ay, hit_time)
            restitution = self.walls[wall_name]
            if axis == "x":
                x = boundary
                vx = -vx * restitution
            else:
                y = boundary
                vy = -vy * restitution
            remaining -= hit_time
            if remaining <= 1e-7:
                return float(x), float(y)

        # A bounded fallback avoids an infinite loop for a corner contact.
        x, y, _, _ = self._propagate(x, y, vx, vy, ax, ay, remaining)
        return float(x), float(y)
