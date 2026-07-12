import cv2
import time
import math
import numpy as np
from collections import deque

from config import (
    HISTORY_SIZE,
    PREDICT_DELTA,
    MIN_CONFIDENCE,
    SMOOTHING_WINDOW,
    MAX_MISSED_FRAMES,
)
from tracker import ObjectTracker
from detector import AppleDetector
from predictors.factory import create_predictor


SUCCESS_THRESHOLD = 20.0   # пикселей — порог "успешного" прогноза
MIN_CHECKS_TO_SHOW = 5     # сколько проверок нужно накопить до вывода метрик
PREDICTOR_NAME = "kalman"  # "linear", "poly2", "poly3", "kalman"


def run_realtime():
    print("\n" + "=" * 60)
    print(f"  РЕАЛЬНЫЙ РЕЖИМ: {PREDICTOR_NAME}")
    print(f"  История tracker : {HISTORY_SIZE} точек")
    print(f"  Прогноз         : Δt = {PREDICT_DELTA}с")
    print("=" * 60 + "\n")

    tracker = ObjectTracker()
    detector = AppleDetector()
    predictor = create_predictor(PREDICTOR_NAME)

    smooth_buffer = deque(maxlen=SMOOTHING_WINDOW)
    pending_predictions = deque()
    last_detection_time = None
    missed_in_row = 0

    # ── Метрики детекции ──────────────────────────────────────
    total_frames = 0
    detected_frames = 0
    missed_frames = 0
    confidences = []

    # ── Метрики прогноза ──────────────────────────────────────
    dx_list = []
    dy_list = []
    error_list = []
    success_count = 0

    print("Нажмите 'q' для выхода\n")

    while True:
        frame, detection = detector.get_detection(conf_threshold=0.3)
        total_frames += 1

        if frame is None:
            print("Не удалось получить кадр.")
            break

        current_time = time.time()
        mode = "waiting"
        x_pred = y_pred = None
        current_pos = None

        if detection is not None:
            missed_in_row = 0

            raw_x = detection["x"]
            raw_y = detection["y"]
            t = detection["timestamp"]
            conf = detection["confidence"]

            detected_frames += 1
            confidences.append(conf)

            # Сглаживание координат
            smooth_buffer.append((raw_x, raw_y))
            sx = int(np.mean([p[0] for p in smooth_buffer]))
            sy = int(np.mean([p[1] for p in smooth_buffer]))

            detection["x"] = sx
            detection["y"] = sy
            current_pos = (sx, sy)

            if conf >= MIN_CONFIDENCE:
                tracker.add_point(sx, sy, timestamp=t, confidence=conf)
                predictor.update(sx, sy, t)
                last_detection_time = t

            frame = detector.draw_detection(frame, detection)

        else:
            missed_frames += 1
            missed_in_row += 1

            if missed_in_row >= MAX_MISSED_FRAMES:
                smooth_buffer.clear()
                tracker.clear()
                predictor.reset()

        # ── Прогноз ───────────────────────────────────────────
        if predictor.is_ready():
            xk, yk, vxk, vyk, axk, ayk = predictor.get_current_state()
            speed = np.sqrt(vxk**2 + vyk**2)

            if speed < 25:
                mode = "stationary"
                x_pred, y_pred = xk, yk
            else:
                mode = "moving"
                x_pred, y_pred = predictor.predict_future(PREDICT_DELTA)

            # Сохраняем прогноз для последующей проверки
            if current_pos is not None and x_pred is not None and y_pred is not None:
                pending_predictions.append({
                    "target_time": current_time + PREDICT_DELTA,
                    "x_pred": float(x_pred),
                    "y_pred": float(y_pred),
                })

            # Рисуем прогноз
            if x_pred is not None and y_pred is not None:
                h_f, w_f = frame.shape[:2]
                xd = max(0, min(int(x_pred), w_f - 1))
                yd = max(0, min(int(y_pred), h_f - 1))

                color = (0, 255, 0) if mode == "stationary" else (0, 255, 255)

                cv2.circle(frame, (xd, yd), 10, color, -1)
                cv2.putText(
                    frame,
                    f"Pred +{PREDICT_DELTA}s",
                    (xd + 12, yd),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

                if current_pos is not None:
                    cv2.line(frame, current_pos, (xd, yd), color, 2)

            # Техническая информация
            cv2.putText(
                frame,
                f"Predictor: {PREDICTOR_NAME}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )
            cv2.putText(
                frame,
                f"vx={vxk:.1f} vy={vyk:.1f}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )
            cv2.putText(
                frame,
                f"ax={axk:.1f} ay={ayk:.1f}",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )

        else:
            cv2.putText(
                frame,
                f"Predictor: {PREDICTOR_NAME}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )

        # ── Проверяем прогнозы, срок которых наступил ─────────
        if current_pos is not None:
            while pending_predictions and current_time >= pending_predictions[0]["target_time"]:
                pred = pending_predictions.popleft()
                x_true, y_true = current_pos
                dx = pred["x_pred"] - x_true
                dy = pred["y_pred"] - y_true
                error = math.sqrt(dx**2 + dy**2)

                dx_list.append(dx)
                dy_list.append(dy)
                error_list.append(error)

                if error <= SUCCESS_THRESHOLD:
                    success_count += 1

        # ── Статус сверху ─────────────────────────────────────
        det_rate = (detected_frames / total_frames * 100) if total_frames > 0 else 0.0
        cv2.putText(
            frame,
            f"Points: {len(tracker)}/{HISTORY_SIZE} | Mode: {mode} | Det: {det_rate:.1f}%",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2
        )

        # ── Метрики снизу ─────────────────────────────────────
        h, w = frame.shape[:2]
        checked = len(error_list)

        if checked >= MIN_CHECKS_TO_SHOW:
            avg_err = np.mean(error_list)
            rmse = np.sqrt(np.mean(np.square(error_list)))
            mean_dx = np.mean(dx_list)
            mean_dy = np.mean(dy_list)
            success_rate = success_count / checked * 100

            cv2.putText(
                frame,
                f"Checked: {checked}",
                (10, h - 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
            )
            cv2.putText(
                frame,
                f"AvgErr: {avg_err:.1f}px | RMSE: {rmse:.1f}px",
                (10, h - 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
            )
            cv2.putText(
                frame,
                f"dX: {mean_dx:.1f}px | dY: {mean_dy:.1f}px | OK: {success_rate:.1f}%",
                (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
            )
        else:
            cv2.putText(
                frame,
                f"Metrics: collecting... ({checked}/{MIN_CHECKS_TO_SHOW})",
                (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
            )

        cv2.imshow(f"Apple Prediction - {PREDICTOR_NAME}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.release()

    # ── Итоговые метрики в консоль ────────────────────────────
    print("\n" + "=" * 60)
    print("  ИТОГОВЫЕ МЕТРИКИ")
    print("=" * 60)

    det_rate = (detected_frames / total_frames * 100) if total_frames > 0 else 0.0
    avg_conf = np.mean(confidences) if confidences else 0.0

    print(f"Предиктор                         : {PREDICTOR_NAME}")
    print(f"Всего кадров                      : {total_frames}")
    print(f"Кадров с детекцией                : {detected_frames}")
    print(f"Кадров без детекции               : {missed_frames}")
    print(f"Detection rate                    : {det_rate:.2f}%")
    print(f"Средний confidence                : {avg_conf:.3f}")

    if error_list:
        mae = np.mean(error_list)
        rmse = np.sqrt(np.mean(np.square(error_list)))
        max_error = np.max(error_list)
        std_error = np.std(error_list)
        mean_dx = np.mean(dx_list)
        mean_dy = np.mean(dy_list)
        success_rate = success_count / len(error_list) * 100

        print(f"Проверено прогнозов               : {len(error_list)}")
        print(f"Среднее отклонение по X           : {mean_dx:.2f} px")
        print(f"Среднее отклонение по Y           : {mean_dy:.2f} px")
        print(f"MAE                               : {mae:.2f} px")
        print(f"RMSE                              : {rmse:.2f} px")
        print(f"Стандартное отклонение            : {std_error:.2f} px")
        print(f"Максимальная ошибка               : {max_error:.2f} px")
        print(f"Success rate (<={SUCCESS_THRESHOLD}px) : {success_rate:.2f}%")
    else:
        print("Недостаточно данных для оценки прогноза.")

    print("=" * 60)
    print("Программа завершена.")


if __name__ == "__main__":
    run_realtime()