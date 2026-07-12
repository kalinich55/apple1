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

# Размер истории для разных предикторов
LINEAR_HISTORY_SIZE = 3
POLY2_HISTORY_SIZE = 6
POLY3_HISTORY_SIZE = 8

# Предсказание
PREDICT_DELTA = 0.12

# Физическая модель
GRAVITY_PX = 500.0
FIT_GRAVITY = False

# Детекция
MIN_CONFIDENCE = 0.6
MAX_JUMP_PX = 200

# Сглаживание координат
SMOOTHING_WINDOW = 2

# Сколько пропущенных кадров подряд допускается
# до полного сброса трекера и предиктора
MAX_MISSED_FRAMES = 3

# Камера
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Оценка надежности
RESIDUAL_THRESHOLD = 15.0