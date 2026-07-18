# ──────────────────────────────────────────────
# Настройки проекта
# ──────────────────────────────────────────────

# Буфер точек
HISTORY_SIZE = 6
MIN_POINTS = 3

# Минимум точек для разных предикторов
LINEAR_MIN_POINTS = 2
POLY2_MIN_POINTS = 3
POLY3_MIN_POINTS = 4
KALMAN_MIN_POINTS = 2

# Kalman CA: калибруются по дрожанию центра неподвижного объекта и по
# характерной резкости манёвров. Innovation gate = квадрат порога Махаланобиса.
KALMAN_MEASUREMENT_STD_PX = 3.0
KALMAN_JERK_NOISE = 12_000.0
KALMAN_INNOVATION_GATE = 36.0
# После нескольких подряд статистически невозможных измерений считаем, что
# траектория резко изменилась (например, удар) и переинициализируем скорость.
KALMAN_REACQUIRE_AFTER_REJECTIONS = 2
# Короткая история нужна только для защиты прогноза от единичного рывка
# центра сегментации. Она не задаёт траекторию и не знает о стенах.
KALMAN_MOTION_HISTORY = 6
KALMAN_FORECAST_SPEED_FACTOR = 1.5
KALMAN_FORECAST_NOISE_MARGIN_SIGMA = 4.0
# При reacquire большую часть скорости берём из новой серии точек, но не
# делаем мгновенный скачок состояния на 100%.
KALMAN_REACQUIRE_VELOCITY_BLEND = 0.8

# Параметры модели столкновений. В реальном режиме она выключена, пока не
# задана измеренная геометрия стены/пола; в mock геометрия известна точно.
COLLISION_REAL_ENABLED = False
COLLISION_DEFAULT_RESTITUTION = 0.84
COLLISION_MAX_EVENTS = 4
# Example for a calibrated real wall: {"right": 0.84}. Keep empty until the
# image boundary is known to coincide with an actual physical surface.
COLLISION_REAL_WALLS = {}

# Размер истории для разных предикторов
LINEAR_HISTORY_SIZE = 3
POLY2_HISTORY_SIZE = 6
POLY3_HISTORY_SIZE = 8

# Предсказание
PREDICT_DELTA = 0.08

# Максимальная пауза между достоверными измерениями, при которой допустима
# интерполяция фактической позиции для проверки прогноза.
MAX_EVALUATION_GAP = 0.12

# Физическая модель
GRAVITY_PX = 500.0
FIT_GRAVITY = False

# Детекция
MIN_CONFIDENCE = 0.6
# Порог выброса зависит от времени между кадрами.  Значения надо калибровать
# под камеру: 12000 px/s оставляет запас для быстрого теннисного мяча при
# 30–120 FPS; innovation gate Kalman дополнительно отсекает ложные точки.
MAX_SPEED_PX_PER_SEC = 12_000.0
JUMP_MARGIN_PX = 40.0

# Сглаживание координат. Kalman уже подавляет шум; окно 1 не добавляет
# задержку, критичную для быстрых движений.
SMOOTHING_WINDOW = 1

# Сколько пропущенных кадров подряд допускается
# до полного сброса трекера и предиктора
MAX_MISSED_FRAMES = 3

# Камера
# Ноутбучные вебки обычно выглядят заметно лучше в 16:9, чем в принудительном
# 640x480 (4:3), которое визуально сужает область и портит картинку.
FRAME_WIDTH = 1600
FRAME_HEIGHT = 900
FPS = 30
CAMERA_INDEX = 0

# Отображение окна реальной камеры. Меняет только размер окна на экране, а не
# данные, которые получает нейросеть.
REAL_WINDOW_SCALE = 1.0

# Окно тестового mock-режима (`main_mock.py`). Режимы 1--5 используют именно
# эту настройку; она не влияет на окно живой камеры в `main.py`.
MOCK_WINDOW_SCALE = 2.0
UI_FONT_SCALE = 0.85
UI_FONT_THICKNESS = 2

# Профиль целевого объекта. Чтобы переключить проект на теннисный мяч,
# поменяйте OBJECT_PROFILE на "tennis_ball" (или "tennis_ball_alt" для
# весов с классом "tennis-ball").
OBJECT_PROFILE = "apple"
OBJECT_PROFILES = {
    "apple": {
        "model_path": "best_apple_colab_30epochs.pt",
        "target_class_name": "apple",
        "display_name": "Apple",
    },
    "tennis_ball": {
        "model_path": "best_tennis_ball_50epochs.pt",
        "target_class_name": "ball",
        "display_name": "Tennis ball",
    },
    "tennis_ball_alt": {
        "model_path": "best_ball.pt",
        "target_class_name": "tennis-ball",
        "display_name": "Tennis ball",
    },
}


def get_object_profile():
    """Return a copy of the selected detector profile."""
    try:
        return dict(OBJECT_PROFILES[OBJECT_PROFILE])
    except KeyError as error:
        choices = ", ".join(OBJECT_PROFILES)
        raise ValueError(f"Unknown OBJECT_PROFILE={OBJECT_PROFILE!r}; choose one of: {choices}") from error

# Оценка надежности
RESIDUAL_THRESHOLD = 15.0
