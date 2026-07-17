import cv2
import time
import math
import numpy as np


class MockDetector:
    def __init__(
        self,
        trajectory="linear",
        width=960,
        height=540,
        fps=30,
        duration_sec=60.0,
        radius=20,
        debug_timing=False
    ):
        self.trajectory = trajectory
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.duration_sec = float(duration_sec)
        self.radius = int(radius)
        self.debug_timing = bool(debug_timing)

        self.frame_index = 0
        self.max_frames = int(round(self.duration_sec * self.fps))
        self.start_time = time.time()

        self.sim_initialized = False
        self.sim_x = None
        self.sim_y = None
        self.sim_vx = None
        self.sim_vy = None
        self.sim_bounce_count = 0

        print(
            f"[MockDetector] trajectory={self.trajectory}, "
            f"fps={self.fps}, duration={self.duration_sec}s, "
            f"frames={self.max_frames}, size={self.width}x{self.height}"
        )

    def reset(self):
        self.frame_index = 0
        self.start_time = time.time()

        self.sim_initialized = False
        self.sim_x = None
        self.sim_y = None
        self.sim_vx = None
        self.sim_vy = None
        self.sim_bounce_count = 0

    def get_detection(self, conf_threshold=0.6):
        if self.frame_index >= self.max_frames:
            return None, None

        t0 = time.time()

        sim_time = self.frame_index * self.dt
        x, y = self._get_position(sim_time)

        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (35, 35, 35)

        center = (int(round(x)), int(round(y)))
        cv2.circle(frame, center, self.radius, (0, 255, 255), -1)

        contour = self._make_circle_contour(center[0], center[1], self.radius)

        x1 = int(round(x - self.radius))
        y1 = int(round(y - self.radius))
        x2 = int(round(x + self.radius))
        y2 = int(round(y + self.radius))

        x1 = max(0, min(self.width - 1, x1))
        y1 = max(0, min(self.height - 1, y1))
        x2 = max(0, min(self.width - 1, x2))
        y2 = max(0, min(self.height - 1, y2))

        timestamp = self.start_time + sim_time

        detection = {
            "timestamp": timestamp,
            "sim_time": sim_time,
            "x": center[0],
            "y": center[1],
            "confidence": 0.99,
            "bbox": (x1, y1, x2, y2),
            "contour": contour,
            "true_x": float(x),
            "true_y": float(y),
        }

        self.frame_index += 1

        if self.debug_timing:
            elapsed = time.time() - t0
            fps_now = 1.0 / elapsed if elapsed > 1e-6 else 0.0
            print(f"[MockDetector] frame_time={elapsed:.6f}s | fps={fps_now:.2f}")

        return frame, detection

    def _make_circle_contour(self, cx, cy, r, points=24):
        contour_points = []
        for k in range(points):
            angle = 2.0 * math.pi * k / points
            px = int(round(cx + r * math.cos(angle)))
            py = int(round(cy + r * math.sin(angle)))
            contour_points.append([px, py])
        return np.array(contour_points, dtype=np.int32).reshape(-1, 1, 2)

    def _init_bounce_state(self):
        self.sim_x = 120.0
        self.sim_y = self.height - 145.0
        self.sim_vx = 265.0
        self.sim_vy = -195.0
        self.sim_bounce_count = 0
        self.sim_initialized = True

    def _simulate_bounce_step(self):
        dt = self.dt
        r = self.radius

        g = 420.0
        right_wall_x = self.width - r
        left_wall_x = r
        floor_y = self.height - r
        ceiling_y = r

        wall_restitution = 0.84
        floor_restitution = 0.68
        floor_friction = 0.975
        ceiling_restitution = 0.84

        self.sim_vy += g * dt

        new_x = self.sim_x + self.sim_vx * dt
        new_y = self.sim_y + self.sim_vy * dt

        if new_x >= right_wall_x:
            new_x = right_wall_x
            self.sim_vx = -abs(self.sim_vx) * wall_restitution

        if new_x <= left_wall_x:
            new_x = left_wall_x
            self.sim_vx = abs(self.sim_vx) * wall_restitution

        if new_y <= ceiling_y:
            new_y = ceiling_y
            self.sim_vy = abs(self.sim_vy) * ceiling_restitution

        if new_y >= floor_y:
            new_y = floor_y
            self.sim_vy = -abs(self.sim_vy) * floor_restitution
            self.sim_vx *= floor_friction
            self.sim_bounce_count += 1

            if abs(self.sim_vy) < 28:
                self.sim_vy = 0.0
            if abs(self.sim_vx) < 8:
                self.sim_vx = 0.0

        self.sim_x = new_x
        self.sim_y = new_y

    def _get_position(self, t):
        if self.trajectory == "linear":
            x0, y0 = 90.0, self.height * 0.45
            vx, vy = 220.0, 0.0
            x = x0 + vx * t
            y = y0 + vy * t

        elif self.trajectory == "ballistic":
            x0 = 90.0
            y0 = self.height - 95.0
            vx = 210.0
            vy0 = -455.0
            g = 350.0
            x = x0 + vx * t
            y = y0 + vy0 * t + 0.5 * g * t * t

        elif self.trajectory == "pendulum":
            cx = self.width * 0.5
            cy = 90.0
            L = min(180.0, self.height * 0.34)
            A = 0.46
            omega = 1.60
            damping = 0.02
            theta = A * math.exp(-damping * t) * math.sin(omega * t)
            x = cx + L * math.sin(theta)
            y = cy + L * math.cos(theta)

        elif self.trajectory == "bounce":
            if not self.sim_initialized:
                self._init_bounce_state()
            if self.frame_index == 0:
                x, y = self.sim_x, self.sim_y
            else:
                self._simulate_bounce_step()
                x, y = self.sim_x, self.sim_y
        else:
            x, y = 100.0, self.height * 0.5

        x = max(self.radius, min(self.width - self.radius, x))
        y = max(self.radius, min(self.height - self.radius, y))
        return x, y

    def draw_detection(self, frame, detection):
        if detection is None:
            return frame

        x = detection["x"]
        y = detection["y"]
        conf = detection["confidence"]
        x1, y1, x2, y2 = detection["bbox"]
        contour = detection.get("contour")

        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (255, 100, 0), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)

        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        cv2.putText(
            frame, f"Mock {conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2
        )

        cv2.putText(
            frame, f"traj={self.trajectory}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2
        )

        if self.trajectory == "bounce" and self.sim_initialized:
            cv2.putText(
                frame,
                f"vx={self.sim_vx:.1f} vy={self.sim_vy:.1f} bounces={self.sim_bounce_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 2
            )
        return frame

    def release(self):
        cv2.destroyAllWindows()