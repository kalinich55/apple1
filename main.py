import cv2
import math
import numpy as np
from collections import deque

from config import (
    HISTORY_SIZE,
    PREDICT_DELTA,
    MIN_CONFIDENCE,
    SMOOTHING_WINDOW,
    MAX_MISSED_FRAMES,
    MAX_EVALUATION_GAP,
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FPS,
    REAL_WINDOW_SCALE,
    UI_FONT_SCALE,
    UI_FONT_THICKNESS,
    COLLISION_REAL_ENABLED,
    COLLISION_REAL_WALLS,
    get_object_profile,
)
from tracker import ObjectTracker
from detector import AppleDetector
from predictors.factory import create_predictor
from prediction_evaluator import PredictionEvaluator


SUCCESS_THRESHOLD = 20.0   # пикселей — порог "успешного" прогноза
MIN_CHECKS_TO_SHOW = 5     # сколько проверок нужно накопить до вывода метрик
VALIDATION_MIN_CONFIDENCE = 0.75
VALIDATION_EDGE_MARGIN = 25
VALIDATION_MIN_AREA = 120
PREDICTOR_NAME = "kalman_ca"  # "linear", "poly2", "poly3", "kalman_ca" "kalman_cv"
WINDOW_NAME = f"Apple Prediction - {PREDICTOR_NAME}"


def run_realtime():
    print("\n" + "=" * 60)
    print(f"  РЕАЛЬНЫЙ РЕЖИМ: {PREDICTOR_NAME}")
    print(f"  История tracker : {HISTORY_SIZE} точек")
    print(f"  Прогноз         : Δt = {PREDICT_DELTA}с")
    print("=" * 60 + "\n")

    tracker = ObjectTracker()
    object_profile = get_object_profile()
    detector = AppleDetector(
        camera_index=CAMERA_INDEX,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        fps=FPS,
        **object_profile,
    )
    predictor = create_predictor(PREDICTOR_NAME)

    smooth_buffer = deque(maxlen=SMOOTHING_WINDOW)
    evaluator = PredictionEvaluator(MAX_EVALUATION_GAP)
    missed_in_row = 0
    window_initialized = False

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

    def record_evaluations(evaluations):
        nonlocal success_count, valid_success_count

        for evaluation in evaluations:
            dx = evaluation.dx
            dy = evaluation.dy
            error = evaluation.error

            dx_list.append(dx)
            dy_list.append(dy)
            abs_dx_list.append(abs(dx))
            abs_dy_list.append(abs(dy))
            error_list.append(error)

            if evaluation.valid:
                valid_error_list.append(error)
                valid_dx_list.append(dx)
                valid_dy_list.append(dy)
                valid_abs_dx_list.append(abs(dx))
                valid_abs_dy_list.append(abs(dy))

                if error <= SUCCESS_THRESHOLD:
                    valid_success_count += 1

            if error <= SUCCESS_THRESHOLD:
                success_count += 1

    print("Нажмите 'q' для выхода, 'm' — показать/скрыть метрики\n")

    while True:
        frame, detection = detector.get_detection(conf_threshold=0.6)
        total_frames += 1

        if frame is None:
            print("Не удалось получить кадр.")
            break

        if not window_initialized:
            h0, w0 = frame.shape[:2]
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                WINDOW_NAME,
                int(round(w0 * REAL_WINDOW_SCALE)),
                int(round(h0 * REAL_WINDOW_SCALE)),
            )
            window_initialized = True

        mode = "waiting"
        x_pred = y_pred = None
        current_pos = None
        prediction_source_time = None

        if detection is not None:
            missed_in_row = 0

            raw_x = float(detection["x"])
            raw_y = float(detection["y"])
            t = float(detection["timestamp"])
            conf = float(detection["confidence"])

            detected_frames += 1
            confidences.append(conf)

            # По умолчанию рисуем исходный центр; в предиктор попадают только
            # уверенные точки, принятые фильтром выбросов.
            detection["x"] = int(round(raw_x))
            detection["y"] = int(round(raw_y))
            if conf >= MIN_CONFIDENCE:
                # Reject the raw detector point before smoothing. Otherwise a
                # large outlier can be diluted by a moving average and leak
                # into the predictor on the next frame.
                accepted = tracker.add_point(raw_x, raw_y, timestamp=t, confidence=conf)

                if accepted:
                    smooth_buffer.append((raw_x, raw_y))
                    sx = float(np.mean([p[0] for p in smooth_buffer]))
                    sy = float(np.mean([p[1] for p in smooth_buffer]))
                    detection["x"] = int(round(sx))
                    detection["y"] = int(round(sy))
                    current_pos = (detection["x"], detection["y"])

                    bbox = detection.get("bbox")
                    h_f, w_f = frame.shape[:2]
                    current_eval_valid = False
                    if bbox is not None and conf >= VALIDATION_MIN_CONFIDENCE:
                        x1, y1, x2, y2 = bbox
                        box_w = max(0, x2 - x1)
                        box_h = max(0, y2 - y1)
                        box_area = box_w * box_h
                        current_eval_valid = (
                            x1 >= VALIDATION_EDGE_MARGIN
                            and y1 >= VALIDATION_EDGE_MARGIN
                            and x2 <= w_f - VALIDATION_EDGE_MARGIN
                            and y2 <= h_f - VALIDATION_EDGE_MARGIN
                            and box_area >= VALIDATION_MIN_AREA
                        )

                    # The raw segmentation center is the best available
                    # ground truth in real mode.  Predictor input may be
                    # smoothed, but its error is measured against the actual
                    # detector observation at the target timestamp.
                    record_evaluations(
                        evaluator.observe(
                            t,
                            raw_x,
                            raw_y,
                            valid=current_eval_valid,
                        )
                    )
                    update_result = predictor.update(sx, sy, t)
                    if hasattr(predictor, "set_environment") and bbox is not None:
                        x1, y1, x2, y2 = bbox
                        radius_px = max(x2 - x1, y2 - y1) / 2.0
                        predictor.set_environment(
                            w_f,
                            h_f,
                            radius_px,
                            walls=COLLISION_REAL_WALLS,
                            enabled=COLLISION_REAL_ENABLED,
                        )
                    # Existing regression predictors use ``None`` for a
                    # successful update; Kalman explicitly returns False for
                    # an innovation-gated measurement.
                    if update_result is not False:
                        prediction_source_time = t
            frame = detector.draw_detection(frame, detection)

        else:
            missed_frames += 1
            missed_in_row += 1

            if missed_in_row >= MAX_MISSED_FRAMES:
                smooth_buffer.clear()
                tracker.clear()
                predictor.reset()
                evaluator.reset()

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

            # Queue only a prediction originating from a newly accepted frame.
            # Its target is expressed on the camera timestamp scale, not on
            # the slower UI/inference clock.
            if prediction_source_time is not None and x_pred is not None and y_pred is not None:
                evaluator.add_prediction(
                    prediction_source_time,
                    PREDICT_DELTA,
                    x_pred,
                    y_pred,
                )

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
                    cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, color, UI_FONT_THICKNESS
                )

                if current_pos is not None:
                    cv2.line(frame, current_pos, (xd, yd), color, 2)

            # Техническая информация
            cv2.putText(
                frame,
                f"Predictor: {PREDICTOR_NAME}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, (255, 255, 255), UI_FONT_THICKNESS
            )
            cv2.putText(
                frame,
                f"vx={vxk:.1f} vy={vyk:.1f}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, (255, 255, 255), UI_FONT_THICKNESS
            )
            cv2.putText(
                frame,
                f"ax={axk:.1f} ay={ayk:.1f}",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, (255, 255, 255), UI_FONT_THICKNESS
            )

        else:
            cv2.putText(
                frame,
                f"Predictor: {PREDICTOR_NAME}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, (255, 255, 255), UI_FONT_THICKNESS
            )

        # ── Статус сверху ─────────────────────────────────────
        det_rate = (detected_frames / total_frames * 100) if total_frames > 0 else 0.0
        cv2.putText(
            frame,
            f"Points: {len(tracker)}/{HISTORY_SIZE} | Mode: {mode} | Det: {det_rate:.1f}%",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, (255, 255, 255), UI_FONT_THICKNESS
        )

        # ── Метрики снизу ─────────────────────────────────────
        h, w = frame.shape[:2]
        checked = len(error_list)

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
                    cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, (0, 255, 255), UI_FONT_THICKNESS
                )
                cv2.putText(
                    frame,
                    f"Mean2D: {avg_err:.1f}px | valid: {valid_avg_err:.1f}px",
                    (10, h - 98),
                    cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, (0, 255, 255), UI_FONT_THICKNESS
                )
                cv2.putText(
                    frame,
                    f"RMSE: {rmse:.1f}px | validRMSE: {valid_rmse:.1f}px",
                    (10, h - 71),
                    cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, (0, 255, 255), UI_FONT_THICKNESS
                )
                cv2.putText(
                    frame,
                    f"|dX|: {mean_abs_dx:.1f}px | valid|dX|: {valid_mean_abs_dx:.1f}px",
                    (10, h - 44),
                    cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, (0, 255, 255), UI_FONT_THICKNESS
                )
                cv2.putText(
                    frame,
                    f"|dY|: {mean_abs_dy:.1f}px | valid|dY|: {valid_mean_abs_dy:.1f}px",
                    (10, h - 17),
                    cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, (0, 255, 255), UI_FONT_THICKNESS
                )
                cv2.putText(
                    frame,
                    f"OK: {success_rate:.1f}% | validOK: {valid_success_rate:.1f}%",
                    (10, h - 0),
                    cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, (0, 255, 255), UI_FONT_THICKNESS
                )
            else:
                cv2.putText(
                    frame,
                    f"Metrics: collecting... ({checked}/{MIN_CHECKS_TO_SHOW})",
                    (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, (0, 255, 255), UI_FONT_THICKNESS
                )
        else:
            cv2.putText(
                frame,
                "Press 'm' for metrics",
                (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, (220, 220, 220), UI_FONT_THICKNESS
            )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            show_metrics = not show_metrics

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
    print(f"Пропущено из-за разрыва кадров    : {evaluator.skipped_predictions}")
    print(f"Не проверено в конце серии        : {evaluator.pending_count}")

    if error_list:
        mae = np.mean(error_list)
        rmse = np.sqrt(np.mean(np.square(error_list)))
        max_error = np.max(error_list)
        std_error = np.std(error_list)
        median_error = np.median(error_list)
        p95_error = np.percentile(error_list, 95)
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
        print(f"Средняя 2D-ошибка                 : {mae:.2f} px")
        print(f"Медианная 2D-ошибка               : {median_error:.2f} px")
        print(f"P95 2D-ошибки                     : {p95_error:.2f} px")
        print(f"RMSE                              : {rmse:.2f} px")
        print(f"Стандартное отклонение            : {std_error:.2f} px")
        print(f"Максимальная ошибка               : {max_error:.2f} px")
        print(f"Success rate (<={SUCCESS_THRESHOLD}px) : {success_rate:.2f}%")
        print(f"--- Валидные оценки (видимый объект) ---")
        print(f"Проверено валидных прогнозов      : {len(valid_error_list)}")
        print(f"Среднее |dX| (valid)              : {valid_mean_abs_dx:.2f} px")
        print(f"Среднее |dY| (valid)              : {valid_mean_abs_dy:.2f} px")
        print(f"Средняя 2D-ошибка (valid)         : {valid_mae:.2f} px")
        print(f"RMSE (valid)                      : {valid_rmse:.2f} px")
        print(f"Success rate (valid)              : {valid_success_rate:.2f}%")
    else:
        print("Недостаточно данных для оценки прогноза.")

    print("=" * 60)
    print("Программа завершена.")


if __name__ == "__main__":
    run_realtime()
