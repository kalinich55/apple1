from collections import deque
import math
import time
from config import HISTORY_SIZE, MAX_SPEED_PX_PER_SEC, JUMP_MARGIN_PX


class ObjectTracker:
    """
    Буфер последних N точек траектории объекта.
    Принимает данные от модуля распознавания.
    Фильтрует выбросы и пропуски.
    """

    def __init__(self, max_points: int = HISTORY_SIZE):
        self.max_points = max_points
        self.history    = deque(maxlen=max_points)

    def add_point(
        self,
        x: float,
        y: float,
        timestamp: float = None,
        confidence: float = 1.0,
    ) -> bool:
        """
        Добавить точку в буфер.

        x, y       : координаты центра объекта в пикселях
        timestamp  : время кадра в секундах
        confidence : уверенность детектора (0..1)

        Возвращает True если точка принята, False если отфильтрована.
        """
        if timestamp is None:
            timestamp = time.perf_counter()

        if not all(math.isfinite(value) for value in (x, y, timestamp)):
            print("  [tracker] Некорректная точка отфильтрована")
            return False

        # Фильтр: допустимый сдвиг должен зависеть от интервала между
        # кадрами. Иначе быстрый мяч ошибочно считается выбросом, а при
        # высокой частоте кадров разрешается слишком большой скачок.
        if len(self.history) > 0:
            last = self.history[-1]
            dt = timestamp - last[0]
            if dt <= 0:
                print(f"  [tracker] Неупорядоченный timestamp отфильтрован: dt={dt:.6f}с")
                return False

            dist = ((x - last[1])**2 + (y - last[2])**2) ** 0.5
            max_distance = JUMP_MARGIN_PX + MAX_SPEED_PX_PER_SEC * dt
            if dist > max_distance:
                print(
                    "  [tracker] Выброс отфильтрован: "
                    f"скачок {dist:.1f}px > {max_distance:.1f}px за {dt:.3f}с"
                )
                return False

        self.history.append((timestamp, x, y))
        return True

    def get_history(self) -> list:
        """Вернуть историю точек в виде списка (timestamp, x, y)."""
        return list(self.history)

    def is_ready(self, min_points: int = None) -> bool:
        """Достаточно ли точек для прогноза."""
        from config import MIN_POINTS
        n = min_points if min_points is not None else MIN_POINTS
        return len(self.history) >= n

    def clear(self):
        """Сбросить буфер."""
        self.history.clear()

    def __len__(self):
        return len(self.history)
