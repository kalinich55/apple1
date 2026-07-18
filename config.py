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
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
CAMERA_INDEX = 0

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
