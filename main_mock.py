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
from mock_detector import MockDetector
from predictors.factory import create_predictor


SUCCESS_THRESHOLD = 20.0
MIN_CHECKS_TO_SHOW = 5
PREDICTOR_NAME = "kalman_cv"   # "linear", "poly2", "poly3", "kalman_ca", "kalman_cv"

USE_MOCK = True
MOCK_TRAJECTORY = "linear"

EXPERIMENT_DURATION_SEC = 6.0
WINDOW_NAME = "Trajectory Prediction"

MOCK_WIDTH = 960
MOCK_HEIGHT = 540


def create_detector():
    if USE_MOCK:
        return MockDetector(
            trajectory=MOCK_TRAJECTORY,
            width=MOCK_WIDTH,
            height=MOCK_HEIGHT,
            fps=30,
            duration_sec=EXPERIMENT_DURATION_SEC,
            radius=20,
            debug_timing=False
        )
    else:
        return AppleDetector()


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
    print("=" * 60 + "\n")


def print_metrics(total_frames, detected_frames, missed_frames, confidences,
                  dx_list, dy_list, error_list, success_count):
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

    if error_list:
        mae = float(np.mean(error_list))
        rmse = float(np.sqrt(np.mean(np.square(error_list))))
        max_error = float(np.max(error_list))
        std_error = float(np.std(error_list))
        mean_dx = float(np.mean(dx_list))
        mean_dy = float(np.mean(dy_list))
        success_rate = success_count / len(error_list) * 100.0

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
    print("Эксперимент завершен. Окно остается открытым.")


def run_single_experiment():
    tracker = ObjectTracker()
    detector = create_detector()
    predictor = create_predictor(PREDICTOR_NAME)

    smooth_buffer = deque(maxlen=SMOOTHING_WINDOW)
    pending_predictions = deque()
    missed_in_row = 0

    total_frames = 0
    detected_frames = 0
    missed_frames = 0
    confidences = []

    dx_list = []
    dy_list = []
    error_list = []
    success_count = 0

    experiment_start = time.time()
    last_frame = np.zeros((MOCK_HEIGHT, MOCK_WIDTH, 3), dtype=np.uint8)
    last_frame[:] = (35, 35, 35)

    while True:
        elapsed_experiment = time.time() - experiment_start
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
        x_pred = None
        y_pred = None
        current_pos = None
        current_eval_time = None

        if detection is not None:
            missed_in_row = 0

            conf = float(detection["confidence"])
            detected_frames += 1
            confidences.append(conf)

            det_time = float(detection["timestamp"])
            current_eval_time = det_time

            if USE_MOCK:
                raw_x = float(detection.get("true_x", detection["x"]))
                raw_y = float(detection.get("true_y", detection["y"]))
                sx = raw_x
                sy = raw_y
            else:
                raw_x = float(detection["x"])
                raw_y = float(detection["y"])
                smooth_buffer.append((raw_x, raw_y))
                sx = float(np.mean([p[0] for p in smooth_buffer]))
                sy = float(np.mean([p[1] for p in smooth_buffer]))

            detection["x"] = int(round(sx))
            detection["y"] = int(round(sy))
            current_pos = (float(sx), float(sy))

            tracker.add_point(sx, sy, timestamp=det_time, confidence=conf)

            if conf >= MIN_CONFIDENCE:
                predictor.update(sx, sy, det_time)

            frame = detector.draw_detection(frame, detection)

            try:
                xk, yk, vxk, vyk, axk, ayk = predictor.get_current_state()
                speed = math.sqrt(vxk ** 2 + vyk ** 2)

                if speed < 25:
                    mode = "stationary"
                    x_pred, y_pred = xk, yk
                else:
                    mode = "moving"
                    x_pred, y_pred = predictor.predict_future(PREDICT_DELTA)

                if current_pos is not None and x_pred is not None and y_pred is not None:
                    pending_predictions.append({
                        "target_time": det_time + PREDICT_DELTA,
                        "x_pred": float(x_pred),
                        "y_pred": float(y_pred),
                    })

                if x_pred is not None and y_pred is not None:
                    h_f, w_f = frame.shape[:2]
                    xd = max(0, min(int(round(x_pred)), w_f - 1))
                    yd = max(0, min(int(round(y_pred)), h_f - 1))

                    color = (0, 255, 0) if mode == "stationary" else (0, 255, 255)

                    cv2.circle(frame, (xd, yd), 10, color, -1)
                    cv2.putText(
                        frame,
                        f"Pred +{PREDICT_DELTA}s",
                        (xd + 12, yd),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )

                    cv2.line(
                        frame,
                        (int(round(current_pos[0])), int(round(current_pos[1]))),
                        (xd, yd),
                        color,
                        2
                    )

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

            except Exception:
                cv2.putText(
                    frame,
                    f"Predictor: {PREDICTOR_NAME}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
                )
                cv2.putText(
                    frame,
                    "Predict not ready",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2
                )

        else:
            missed_frames += 1
            missed_in_row += 1

            if missed_in_row >= MAX_MISSED_FRAMES:
                predictor.reset()
                tracker.clear()
                smooth_buffer.clear()
                pending_predictions.clear()
                missed_in_row = 0

        if current_pos is not None and current_eval_time is not None:
            while pending_predictions and current_eval_time >= pending_predictions[0]["target_time"]:
                pred = pending_predictions.popleft()
                x_true, y_true = current_pos

                dx = pred["x_pred"] - x_true
                dy = pred["y_pred"] - y_true
                error = math.sqrt(dx ** 2 + dy ** 2)

                dx_list.append(dx)
                dy_list.append(dy)
                error_list.append(error)

                if error <= SUCCESS_THRESHOLD:
                    success_count += 1

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
                "1-linear 2-ballistic 3-pendulum 4-bounce | r-restart | q-quit",
                (10, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 255, 180), 2
            )

        h, w = frame.shape[:2]
        checked = len(error_list)

        if checked >= MIN_CHECKS_TO_SHOW:
            avg_err = float(np.mean(error_list))
            rmse = float(np.sqrt(np.mean(np.square(error_list))))
            mean_dx = float(np.mean(dx_list))
            mean_dy = float(np.mean(dy_list))
            success_rate = success_count / checked * 100.0

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

        cv2.putText(
            frame,
            f"Time: {elapsed_experiment:.2f}/{EXPERIMENT_DURATION_SEC:.2f}s",
            (10, h - 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 180), 2
        )

        cv2.imshow(WINDOW_NAME, frame)
        last_frame = frame.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            detector.release()
            return "quit", last_frame

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
        if key == ord("r"):
            detector.release()
            return "restart", last_frame

    detector.release()

    print_metrics(
        total_frames, detected_frames, missed_frames, confidences,
        dx_list, dy_list, error_list, success_count
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


def main():
    global MOCK_TRAJECTORY

    print("Управление:")
    print("1=linear  2=ballistic  3=pendulum  4=bounce")
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

        if command in ("linear", "ballistic", "pendulum", "bounce"):
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
            if next_command in ("linear", "ballistic", "pendulum", "bounce"):
                MOCK_TRAJECTORY = next_command
                continue

    cv2.destroyAllWindows()
    print("Программа завершена.")


if __name__ == "__main__":
    main()