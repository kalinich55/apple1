"""Matching predictions with later object observations.

The detector and the predictor work on frame timestamps, whereas the UI loop
can be delayed by YOLO inference or rendering.  This module keeps evaluation
on the detector timeline: a prediction made at ``t`` for ``t + delta`` is
compared with an interpolated observation at that exact target time.
"""

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PredictionEvaluation:
    """One prediction compared with an interpolated ground-truth position."""

    source_time: float
    target_time: float
    x_pred: float
    y_pred: float
    x_true: float
    y_true: float
    valid: bool

    @property
    def dx(self) -> float:
        return self.x_pred - self.x_true

    @property
    def dy(self) -> float:
        return self.y_pred - self.y_true

    @property
    def error(self) -> float:
        return math.hypot(self.dx, self.dy)


@dataclass(frozen=True)
class _Observation:
    timestamp: float
    x: float
    y: float
    valid: bool


class PredictionEvaluator:
    """Queue predictions and evaluate them on a monotonic observation stream.

    ``max_interpolation_gap`` prevents a prediction from being scored against
    a long detector outage.  Such predictions are counted as skipped instead
    of contaminating the error statistics.
    """

    def __init__(self, max_interpolation_gap: float = 0.20):
        if max_interpolation_gap <= 0:
            raise ValueError("max_interpolation_gap must be positive")

        self.max_interpolation_gap = float(max_interpolation_gap)
        self._pending = deque()
        self._previous = None
        self.skipped_predictions = 0
        self.non_monotonic_observations = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def reset(self) -> None:
        # A reset marks a discontinuity in object identity/visibility.  Those
        # forecasts cannot be honestly evaluated against the next track.
        self.skipped_predictions += len(self._pending)
        self._pending.clear()
        self._previous = None

    def add_prediction(
        self,
        source_time: float,
        delta_t: float,
        x_pred: float,
        y_pred: float,
    ) -> bool:
        """Add a finite prediction and return whether it was accepted."""
        values = (source_time, delta_t, x_pred, y_pred)
        if not all(math.isfinite(float(value)) for value in values):
            return False
        if delta_t < 0:
            return False

        target_time = float(source_time) + float(delta_t)
        self._pending.append((float(source_time), target_time, float(x_pred), float(y_pred)))
        return True

    def observe(
        self,
        timestamp: float,
        x: float,
        y: float,
        *,
        valid: bool,
    ) -> list[PredictionEvaluation]:
        """Record one accepted detector observation.

        Predictions whose target time falls between the previous and current
        observations are evaluated by linear interpolation. This is much less
        biased than evaluating all overdue forecasts against the next frame.
        Analytical mock scenarios bypass this interpolation and use their
        exact physical target position.
        """
        values = (timestamp, x, y)
        if not all(math.isfinite(float(value)) for value in values):
            return []

        current = _Observation(float(timestamp), float(x), float(y), bool(valid))
        previous = self._previous

        if previous is None:
            self._previous = current
            return []

        dt = current.timestamp - previous.timestamp
        if dt <= 0:
            # Camera timestamps should be monotonic.  Do not corrupt a valid
            # interpolation interval if a backend reports a duplicate/older
            # timestamp.
            self.non_monotonic_observations += 1
            return []

        evaluations = []
        while self._pending and self._pending[0][1] <= current.timestamp:
            source_time, target_time, x_pred, y_pred = self._pending.popleft()

            if target_time < previous.timestamp or dt > self.max_interpolation_gap:
                self.skipped_predictions += 1
                continue

            ratio = (target_time - previous.timestamp) / dt
            ratio = min(1.0, max(0.0, ratio))
            x_true = previous.x + ratio * (current.x - previous.x)
            y_true = previous.y + ratio * (current.y - previous.y)

            evaluations.append(
                PredictionEvaluation(
                    source_time=source_time,
                    target_time=target_time,
                    x_pred=x_pred,
                    y_pred=y_pred,
                    x_true=x_true,
                    y_true=y_true,
                    valid=previous.valid and current.valid,
                )
            )

        self._previous = current
        return evaluations
