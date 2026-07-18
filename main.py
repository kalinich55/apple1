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
VALIDATION_MIN_CONFIDENCE = 0.75
VALIDATION_EDGE_MARGIN = 25
VALIDATION_MIN_AREA = 120
PREDICTOR_NAME = "kalman_ca"  # "linear", "poly2", "poly3", "kalman_ca" "kalman_cv"


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
    abs_dx_list = []
    abs_dy_list = []
    error_list = []
    valid_error_list = []
    valid_dx_list = []
    valid_dy_list = []
    valid_abs_dx_list = []
    valid_abs_dy_list = []
    success_count = 0
    valid_success_count = 0
    show_metrics = False

    print("Нажмите 'q' для выхода, 'm' — показать/скрыть метрики\n")

    while True:
        frame, detection = detector.get_detection(conf_threshold=0.6)
        total_frames += 1

        if frame is None:
            print("Не удалось получить кадр.")
            break

        current_time = time.time()
        mode = "waiting"
        x_pred = y_pred = None
        current_pos = None
        current_eval_valid = False

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
            detection_conf = float(detection["confidence"]) if detection is not None else 0.0
            bbox = detection.get("bbox") if detection is not None else None
            h_f, w_f = frame.shape[:2]
            current_eval_valid = False

            if bbox is not None and detection_conf >= VALIDATION_MIN_CONFIDENCE:
                x1, y1, x2, y2 = bbox
                box_w = max(0, x2 - x1)
                box_h = max(0, y2 - y1)
                box_area = box_w * box_h
                if (
                    x1 >= VALIDATION_EDGE_MARGIN and
                    y1 >= VALIDATION_EDGE_MARGIN and
                    x2 <= w_f - VALIDATION_EDGE_MARGIN and
                    y2 <= h_f - VALIDATION_EDGE_MARGIN and
                    box_area >= VALIDATION_MIN_AREA
                ):
                    current_eval_valid = True

            while pending_predictions and current_time >= pending_predictions[0]["target_time"]:
                pred = pending_predictions.popleft()
                x_true, y_true = current_pos
                dx = pred["x_pred"] - x_true
                dy = pred["y_pred"] - y_true
                error = math.sqrt(dx**2 + dy**2)

                dx_list.append(dx)
                dy_list.append(dy)
                abs_dx_list.append(abs(dx))
                abs_dy_list.append(abs(dy))
                error_list.append(error)

                if current_eval_valid:
                    valid_error_list.append(error)
                    valid_dx_list.append(dx)
                    valid_dy_list.append(dy)
                    valid_abs_dx_list.append(abs(dx))
                    valid_abs_dy_list.append(abs(dy))

                    if error <= SUCCESS_THRESHOLD:
                        valid_success_count += 1

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
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("m"):
            show_metrics = not show_metrics

        if show_metrics:
            if checked >= MIN_CHECKS_TO_SHOW:
                avg_err = np.mean(error_list)
                rmse = np.sqrt(np.mean(np.square(error_list)))
                mean_dx = np.mean(dx_list)
                mean_dy = np.mean(dy_list)
                mean_abs_dx = np.mean(abs_dx_list)
                mean_abs_dy = np.mean(abs_dy_list)
                success_rate = success_count / checked * 100

                valid_checked = len(valid_error_list)
                valid_avg_err = np.mean(valid_error_list) if valid_error_list else 0.0
                valid_rmse = np.sqrt(np.mean(np.square(valid_error_list))) if valid_error_list else 0.0
                valid_mean_abs_dx = np.mean(valid_abs_dx_list) if valid_abs_dx_list else 0.0
                valid_mean_abs_dy = np.mean(valid_abs_dy_list) if valid_abs_dy_list else 0.0
                valid_success_rate = valid_success_count / valid_checked * 100 if valid_checked > 0 else 0.0

                cv2.putText(
                    frame,
                    f"All: {checked} | Valid: {valid_checked}",
                    (10, h - 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
                )
                cv2.putText(
                    frame,
                    f"MAE: {avg_err:.1f}px | validMAE: {valid_avg_err:.1f}px",
                    (10, h - 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
                )
                cv2.putText(
                    frame,
                    f"RMSE: {rmse:.1f}px | validRMSE: {valid_rmse:.1f}px",
                    (10, h - 71),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
                )
                cv2.putText(
                    frame,
                    f"|dX|: {mean_abs_dx:.1f}px | valid|dX|: {valid_mean_abs_dx:.1f}px",
                    (10, h - 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
                )
                cv2.putText(
                    frame,
                    f"|dY|: {mean_abs_dy:.1f}px | valid|dY|: {valid_mean_abs_dy:.1f}px",
                    (10, h - 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
                )
                cv2.putText(
                    frame,
                    f"OK: {success_rate:.1f}% | validOK: {valid_success_rate:.1f}%",
                    (10, h - 0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
                )
            else:
                cv2.putText(
                    frame,
                    f"Metrics: collecting... ({checked}/{MIN_CHECKS_TO_SHOW})",
                    (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
                )
        else:
            cv2.putText(
                frame,
                "Press 'm' for metrics",
                (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2
            )

        cv2.imshow(f"Apple Prediction - {PREDICTOR_NAME}", frame)

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
        mean_abs_dx = np.mean(abs_dx_list)
        mean_abs_dy = np.mean(abs_dy_list)
        success_rate = success_count / len(error_list) * 100

        valid_mae = np.mean(valid_error_list) if valid_error_list else 0.0
        valid_rmse = np.sqrt(np.mean(np.square(valid_error_list))) if valid_error_list else 0.0
        valid_mean_abs_dx = np.mean(valid_abs_dx_list) if valid_abs_dx_list else 0.0
        valid_mean_abs_dy = np.mean(valid_abs_dy_list) if valid_abs_dy_list else 0.0
        valid_success_rate = valid_success_count / len(valid_error_list) * 100 if valid_error_list else 0.0

        print(f"Проверено прогнозов               : {len(error_list)}")
        print(f"Среднее отклонение по X           : {mean_dx:.2f} px")
        print(f"Среднее отклонение по Y           : {mean_dy:.2f} px")
        print(f"Среднее |dX|                     : {mean_abs_dx:.2f} px")
        print(f"Среднее |dY|                     : {mean_abs_dy:.2f} px")
        print(f"MAE                               : {mae:.2f} px")
        print(f"RMSE                              : {rmse:.2f} px")
        print(f"Стандартное отклонение            : {std_error:.2f} px")
        print(f"Максимальная ошибка               : {max_error:.2f} px")
        print(f"Success rate (<={SUCCESS_THRESHOLD}px) : {success_rate:.2f}%")
        print(f"--- Валидные оценки (видимый объект) ---")
        print(f"Проверено валидных прогнозов      : {len(valid_error_list)}")
        print(f"Среднее |dX| (valid)              : {valid_mean_abs_dx:.2f} px")
        print(f"Среднее |dY| (valid)              : {valid_mean_abs_dy:.2f} px")
        print(f"MAE (valid)                       : {valid_mae:.2f} px")
        print(f"RMSE (valid)                      : {valid_rmse:.2f} px")
        print(f"Success rate (valid)              : {valid_success_rate:.2f}%")
    else:
        print("Недостаточно данных для оценки прогноза.")

    print("=" * 60)
    print("Программа завершена.")


if __name__ == "__main__":
    run_realtime()