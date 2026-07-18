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
    MAX_EVALUATION_GAP,
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FPS,
    get_object_profile,
)
from tracker import ObjectTracker
from detector import AppleDetector
from mock_detector import MockDetector
from predictors.factory import create_predictor
from prediction_evaluator import PredictionEvaluation, PredictionEvaluator


SUCCESS_THRESHOLD = 20.0
MIN_CHECKS_TO_SHOW = 5
VALIDATION_MIN_CONFIDENCE = 0.75
VALIDATION_EDGE_MARGIN = 25
VALIDATION_MIN_AREA = 120
# A constant-acceleration Kalman filter is the most stable common baseline
# for linear, ballistic and short pendulum segments. Other implementations
# remain selectable here for comparison.
PREDICTOR_NAME = "kalman_ca"   # "linear", "poly2", "poly3", "kalman_ca", "kalman_cv"

USE_MOCK = True
MOCK_TRAJECTORY = "linear"

EXPERIMENT_DURATION_SEC = 6.0
WINDOW_NAME = "Trajectory Prediction"

MOCK_WIDTH = 960
MOCK_HEIGHT = 540
MOCK_FPS = 30
MOCK_MEASUREMENT_NOISE_PX = 1.5
MOCK_DROPOUT_PROBABILITY = 0.0
MOCK_RANDOM_SEED = 42


def create_detector():
    if USE_MOCK:
        return MockDetector(
            trajectory=MOCK_TRAJECTORY,
            width=MOCK_WIDTH,
            height=MOCK_HEIGHT,
            fps=MOCK_FPS,
            duration_sec=EXPERIMENT_DURATION_SEC,
            radius=20,
            measurement_noise_px=MOCK_MEASUREMENT_NOISE_PX,
            dropout_probability=MOCK_DROPOUT_PROBABILITY,
            random_seed=MOCK_RANDOM_SEED,
            debug_timing=False
        )
    else:
        return AppleDetector(
            camera_index=CAMERA_INDEX,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            fps=FPS,
            **get_object_profile(),
        )


def print_header():
    print("\n" + "=" * 60)
    print(f"  РЕЖИМ: {'MOCK' if USE_MOCK else 'REAL'}")
    print(f"  ПРЕДИКТОР      : {PREDICTOR_NAME}")
    print(f"  История tracker: {HISTORY_SIZE} точек")
    print(f"  Прогноз        : Δt = {PREDICT_DELTA}с")
    print(f"  Длительность   : {EXPERIMENT_DURATION_SEC:.1f}с")
    if USE_MOCK:
        print(f"  Траектория mock: {MOCK_TRAJECTORY}")
        print(f"  Размер поля    : {MOCK_WIDTH}x{MOCK_HEIGHT}")
        print(f"  Наблюдение     : шум σ={MOCK_MEASUREMENT_NOISE_PX}px, dropout={MOCK_DROPOUT_PROBABILITY:.0%}")
    print("=" * 60 + "\n")


def print_metrics(
    total_frames,
    detected_frames,
    missed_frames,
    confidences,
    dx_list,
    dy_list,
    abs_dx_list,
    abs_dy_list,
    error_list,
    success_count,
    valid_error_list,
    valid_success_count,
    skipped_predictions,
    pending_predictions,
):
    print("\n" + "=" * 60)
    print("  ИТОГОВЫЕ МЕТРИКИ")
    print("=" * 60)

    det_rate = (detected_frames / total_frames * 100.0) if total_frames > 0 else 0.0
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    print(f"Предиктор                         : {PREDICTOR_NAME}")
    print(f"Режим                             : {'MOCK' if USE_MOCK else 'REAL'}")
    if USE_MOCK:
        print(f"Траектория mock                   : {MOCK_TRAJECTORY}")
    print(f"Всего кадров                      : {total_frames}")
    print(f"Кадров с детекцией                : {detected_frames}")
    print(f"Кадров без детекции               : {missed_frames}")
    print(f"Detection rate                    : {det_rate:.2f}%")
    print(f"Средний confidence                : {avg_conf:.3f}")
    print(f"Пропущено из-за разрыва кадров    : {skipped_predictions}")
    print(f"Не проверено в конце серии        : {pending_predictions}")

    if error_list:
        mae = float(np.mean(error_list))
        rmse = float(np.sqrt(np.mean(np.square(error_list))))
        max_error = float(np.max(error_list))
        std_error = float(np.std(error_list))
        median_error = float(np.median(error_list))
        p95_error = float(np.percentile(error_list, 95))
        mean_dx = float(np.mean(dx_list))
        mean_dy = float(np.mean(dy_list))
        mean_abs_dx = float(np.mean(abs_dx_list)) if abs_dx_list else 0.0
        mean_abs_dy = float(np.mean(abs_dy_list)) if abs_dy_list else 0.0
        success_rate = success_count / len(error_list) * 100.0

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

        if valid_error_list:
            valid_mae = float(np.mean(valid_error_list))
            valid_rmse = float(np.sqrt(np.mean(np.square(valid_error_list))))
            valid_success_rate = valid_success_count / len(valid_error_list) * 100.0
            print(f"Valid MAE                         : {valid_mae:.2f} px")
            print(f"Valid RMSE                        : {valid_rmse:.2f} px")
            print(f"Valid success rate                : {valid_success_rate:.2f}%")
    else:
        print("Недостаточно данных для оценки прогноза.")

    print("=" * 60)
    print("Эксперимент завершен. Окно остается открытым.")


def run_single_experiment():
    tracker = ObjectTracker()
    detector = create_detector()
    predictor = create_predictor(PREDICTOR_NAME)

    smooth_buffer = deque(maxlen=SMOOTHING_WINDOW)
    evaluator = PredictionEvaluator(MAX_EVALUATION_GAP)
    missed_in_row = 0

    total_frames = 0
    detected_frames = 0
    missed_frames = 0
    confidences = []

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

    experiment_start = time.perf_counter()
    last_frame = np.zeros((MOCK_HEIGHT, MOCK_WIDTH, 3), dtype=np.uint8)
    last_frame[:] = (35, 35, 35)

    while True:
        elapsed_experiment = time.perf_counter() - experiment_start
        if elapsed_experiment >= EXPERIMENT_DURATION_SEC:
            print(f"[Main] Эксперимент завершен по таймеру: {elapsed_experiment:.2f} сек")
            break

        frame, detection = detector.get_detection(conf_threshold=0.6)

        if frame is None:
            print("[Main] Источник кадров завершился.")
            break

        total_frames += 1
        last_frame = frame.copy()

        mode = "waiting"
        x_pred = y_pred = None
        current_pos = None
        prediction_source_time = None

        if detection is not None:
            missed_in_row = 0

            conf = float(detection["confidence"])
            detected_frames += 1
            confidences.append(conf)

            det_time = float(detection["timestamp"])
            raw_x = float(detection["x"])
            raw_y = float(detection["y"])
            metric_x = float(detection.get("true_x", raw_x)) if USE_MOCK else raw_x
            metric_y = float(detection.get("true_y", raw_y)) if USE_MOCK else raw_y

            detection["x"] = int(round(raw_x))
            detection["y"] = int(round(raw_y))
            if conf >= MIN_CONFIDENCE:
                accepted = tracker.add_point(raw_x, raw_y, timestamp=det_time, confidence=conf)

                if accepted:
                    smooth_buffer.append((raw_x, raw_y))
                    sx = float(np.mean([point[0] for point in smooth_buffer]))
                    sy = float(np.mean([point[1] for point in smooth_buffer]))
                    detection["x"] = int(round(sx))
                    detection["y"] = int(round(sy))
                    current_pos = (sx, sy)

                    if USE_MOCK:
                        # Mock has an oracle truth signal, including near an
                        # edge or impact. Do not hide difficult frames.
                        current_eval_valid = True
                    else:
                        bbox = detection.get("bbox")
                        h_f, w_f = frame.shape[:2]
                        current_eval_valid = False
                        if bbox is not None and conf >= VALIDATION_MIN_CONFIDENCE:
                            x1, y1, x2, y2 = bbox
                            current_eval_valid = (
                                x1 >= VALIDATION_EDGE_MARGIN
                                and y1 >= VALIDATION_EDGE_MARGIN
                                and x2 <= w_f - VALIDATION_EDGE_MARGIN
                                and y2 <= h_f - VALIDATION_EDGE_MARGIN
                                and (x2 - x1) * (y2 - y1) >= VALIDATION_MIN_AREA
                            )

                    record_evaluations(
                        evaluator.observe(
                            det_time,
                            metric_x,
                            metric_y,
                            valid=current_eval_valid,
                        )
                    )
                    update_result = predictor.update(sx, sy, det_time)
                    if update_result is not False:
                        prediction_source_time = det_time

            frame = detector.draw_detection(frame, detection)

        else:
            missed_frames += 1
            missed_in_row += 1

            if missed_in_row >= MAX_MISSED_FRAMES:
                predictor.reset()
                tracker.clear()
                smooth_buffer.clear()
                evaluator.reset()
                missed_in_row = 0

        if predictor.is_ready():
            xk, yk, vxk, vyk, axk, ayk = predictor.get_current_state()
            speed = math.hypot(vxk, vyk)
            if speed < 25.0:
                mode = "stationary"
                x_pred, y_pred = xk, yk
            else:
                mode = "moving"
                x_pred, y_pred = predictor.predict_future(PREDICT_DELTA)

            if prediction_source_time is not None:
                target_time = prediction_source_time + PREDICT_DELTA
                exact_truth = (
                    detector.get_truth_at_timestamp(target_time)
                    if USE_MOCK and hasattr(detector, "get_truth_at_timestamp")
                    else None
                )
                mock_finished_before_target = (
                    USE_MOCK
                    and hasattr(detector, "start_time")
                    and hasattr(detector, "duration_sec")
                    and target_time > detector.start_time + detector.duration_sec
                )
                if mock_finished_before_target:
                    evaluator.skipped_predictions += 1
                elif exact_truth is None:
                    evaluator.add_prediction(prediction_source_time, PREDICT_DELTA, x_pred, y_pred)
                else:
                    true_x, true_y, target_visible = exact_truth
                    if target_visible:
                        record_evaluations(
                            [
                                PredictionEvaluation(
                                    source_time=prediction_source_time,
                                    target_time=target_time,
                                    x_pred=float(x_pred),
                                    y_pred=float(y_pred),
                                    x_true=float(true_x),
                                    y_true=float(true_y),
                                    valid=True,
                                )
                            ]
                        )
                    else:
                        # The object has left the observable image plane;
                        # this is neither a successful nor a failed 2D match.
                        evaluator.skipped_predictions += 1

            h_f, w_f = frame.shape[:2]
            xd = max(0, min(int(round(x_pred)), w_f - 1))
            yd = max(0, min(int(round(y_pred)), h_f - 1))
            color = (0, 255, 0) if mode == "stationary" else (0, 255, 255)
            cv2.circle(frame, (xd, yd), 10, color, -1)
            cv2.putText(
                frame,
                f"Pred +{PREDICT_DELTA}s",
                (xd + 12, yd),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            if current_pos is not None:
                cv2.line(
                    frame,
                    (int(round(current_pos[0])), int(round(current_pos[1]))),
                    (xd, yd),
                    color,
                    2,
                )
            cv2.putText(
                frame,
                f"vx={vxk:.1f} vy={vyk:.1f}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"ax={axk:.1f} ay={ayk:.1f}",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Collecting predictor history...",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                2,
            )

        cv2.putText(
            frame,
            f"Predictor: {PREDICTOR_NAME}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        det_rate = (detected_frames / total_frames * 100.0) if total_frames > 0 else 0.0
        mode_text = "MOCK" if USE_MOCK else "REAL"

        cv2.putText(
            frame,
            f"Mode: {mode_text} | Points: {len(tracker)}/{HISTORY_SIZE} | Motion: waiting/{mode} | Det: {det_rate:.1f}%",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )

        if USE_MOCK:
            cv2.putText(
                frame,
                f"Trajectory: {MOCK_TRAJECTORY}",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 2
            )
            cv2.putText(
                frame,
                "1-linear 2-ballistic 3-pendulum 4-box bounce 5-wall impact | m/r/q",
                (10, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 255, 180), 2
            )

        h, w = frame.shape[:2]
        checked = len(error_list)

        if show_metrics:
            if checked >= MIN_CHECKS_TO_SHOW:
                avg_err = float(np.mean(error_list))
                rmse = float(np.sqrt(np.mean(np.square(error_list))))
                mean_dx = float(np.mean(dx_list))
                mean_dy = float(np.mean(dy_list))
                mean_abs_dx = float(np.mean(abs_dx_list))
                mean_abs_dy = float(np.mean(abs_dy_list))
                success_rate = success_count / checked * 100.0

                valid_checked = len(valid_error_list)
                valid_avg_err = float(np.mean(valid_error_list)) if valid_error_list else 0.0
                valid_rmse = float(np.sqrt(np.mean(np.square(valid_error_list)))) if valid_error_list else 0.0
                valid_mean_abs_dx = float(np.mean(valid_abs_dx_list)) if valid_abs_dx_list else 0.0
                valid_mean_abs_dy = float(np.mean(valid_abs_dy_list)) if valid_abs_dy_list else 0.0
                valid_success_rate = valid_success_count / valid_checked * 100.0 if valid_checked > 0 else 0.0

                cv2.putText(
                    frame,
                    f"All: {checked} | Valid: {valid_checked}",
                    (10, h - 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
                )
                cv2.putText(
                    frame,
                    f"Mean2D: {avg_err:.1f}px | valid: {valid_avg_err:.1f}px",
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

        cv2.putText(
            frame,
            f"Time: {elapsed_experiment:.2f}/{EXPERIMENT_DURATION_SEC:.2f}s",
            (10, h - 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 180), 2
        )

        cv2.imshow(WINDOW_NAME, frame)
        last_frame = frame.copy()
        key_delay_ms = max(1, int(round(1000.0 / MOCK_FPS))) if USE_MOCK else 1
        key = cv2.waitKey(key_delay_ms) & 0xFF

        if key == ord("q"):
            detector.release()
            return "quit", last_frame
        if key == ord("m"):
            show_metrics = not show_metrics

        if USE_MOCK and key == ord("1"):
            detector.release()
            return "linear", last_frame
        if USE_MOCK and key == ord("2"):
            detector.release()
            return "ballistic", last_frame
        if USE_MOCK and key == ord("3"):
            detector.release()
            return "pendulum", last_frame
        if USE_MOCK and key == ord("4"):
            detector.release()
            return "bounce", last_frame
        if USE_MOCK and key == ord("5"):
            detector.release()
            return "wall_impact", last_frame
        if key == ord("r"):
            detector.release()
            return "restart", last_frame

    detector.release()

    print_metrics(
        total_frames, detected_frames, missed_frames, confidences,
        dx_list, dy_list, abs_dx_list, abs_dy_list, error_list, success_count,
        valid_error_list, valid_success_count,
        evaluator.skipped_predictions, evaluator.pending_count,
    )

    return "finished", last_frame


def wait_for_next_command(last_frame):
    frame = last_frame.copy()
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (40, 40), (w - 40, h - 40), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    lines = [
        "Experiment finished",
        f"Current trajectory: {MOCK_TRAJECTORY}",
        "",
        "Press:",
        "1 - linear",
        "2 - ballistic",
        "3 - pendulum",
        "4 - bounce",
        "5 - wall impact",
        "r - restart current",
        "q - quit",
    ]

    y = 100
    for line in lines:
        cv2.putText(
            frame,
            line,
            (80, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        y += 40

    while True:
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(50) & 0xFF

        if key == ord("q"):
            return "quit"
        if key == ord("r"):
            return "restart"
        if key == ord("1"):
            return "linear"
        if key == ord("2"):
            return "ballistic"
        if key == ord("3"):
            return "pendulum"
        if key == ord("4"):
            return "bounce"
        if key == ord("5"):
            return "wall_impact"


def main():
    global MOCK_TRAJECTORY

    print("Управление:")
    print("1=linear  2=ballistic  3=pendulum  4=box bounce  5=wall impact")
    print("r=restart current  q=quit\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, MOCK_WIDTH, MOCK_HEIGHT)

    last_frame = np.zeros((MOCK_HEIGHT, MOCK_WIDTH, 3), dtype=np.uint8)
    last_frame[:] = (35, 35, 35)

    while True:
        print_header()
        command, last_frame = run_single_experiment()

        if command == "quit":
            break

        if command in ("linear", "ballistic", "pendulum", "bounce", "wall_impact"):
            MOCK_TRAJECTORY = command
            continue

        if command == "restart":
            continue

        if command == "finished":
            next_command = wait_for_next_command(last_frame)

            if next_command == "quit":
                break
            if next_command == "restart":
                continue
            if next_command in ("linear", "ballistic", "pendulum", "bounce", "wall_impact"):
                MOCK_TRAJECTORY = next_command
                continue

    cv2.destroyAllWindows()
    print("Программа завершена.")


if __name__ == "__main__":
    main()
