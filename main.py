import cv2
import numpy as np
from collections import deque

from config import HISTORY_SIZE, PREDICT_DELTA, MIN_CONFIDENCE
from tracker import ObjectTracker
from detector import AppleDetector
from kalman_accel_predictor import KalmanAccelPredictor


def run_realtime():
    print("\n" + "="*60)
    print("  РЕАЛЬНЫЙ РЕЖИМ: Kalman + constant acceleration")
    print(f"  История   : {HISTORY_SIZE} точек")
    print(f"  Прогноз   : Δt = {PREDICT_DELTA}с")
    print("="*60 + "\n")

    tracker   = ObjectTracker()
    detector  = AppleDetector()
    predictor = KalmanAccelPredictor()

    smooth_buffer = deque(maxlen=3)
    last_detection_time = None

    print("Нажмите 'q' для выхода\n")

    while True:
        frame, detection = detector.get_detection(conf_threshold=0.3)

        if frame is None:
            print("Не удалось получить кадр.")
            break

        mode = "waiting"
        reliable = False
        x_pred = y_pred = None
        current_pos = None

        if detection is not None:
            raw_x = detection["x"]
            raw_y = detection["y"]
            t     = detection["timestamp"]
            conf  = detection["confidence"]

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
            # Если объект пропал — можно чуть подождать, но пока просто сброс
            smooth_buffer.clear()
            tracker.clear()
            predictor.reset()

        # Прогноз делаем, когда накопилось достаточно точек
        if tracker.is_ready():
            xk, yk, vxk, vyk, axk, ayk = predictor.get_current_state()

            speed = np.sqrt(vxk**2 + vyk**2)

            # Классификация режима по оцененному состоянию
            if speed < 25:
                mode = "stationary"
                x_pred, y_pred = xk, yk
                reliable = True
            else:
                mode = "moving"
                x_pred, y_pred = predictor.predict_future(PREDICT_DELTA)
                reliable = True

            # Рисуем текущую сглаженную оценку Калмана
            cv2.circle(frame, (int(xk), int(yk)), 6, (255, 0, 255), -1)
            cv2.putText(
                frame,
                "Kalman",
                (int(xk) + 10, int(yk)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                2
            )

            # Рисуем прогноз
            if x_pred is not None and y_pred is not None:
                h, w = frame.shape[:2]
                xd = max(0, min(int(x_pred), w - 1))
                yd = max(0, min(int(y_pred), h - 1))

                color = (0, 255, 0) if mode == "stationary" else (0, 255, 255)

                cv2.circle(frame, (xd, yd), 10, color, -1)
                cv2.putText(
                    frame,
                    f"Pred +{PREDICT_DELTA}s",
                    (xd + 12, yd),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                if current_pos is not None:
                    cv2.line(frame, current_pos, (xd, yd), color, 2)

            # Техническая информация
            info1 = f"vx={vxk:.1f} vy={vyk:.1f}"
            info2 = f"ax={axk:.1f} ay={ayk:.1f}"

            cv2.putText(frame, info1, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, info2, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Статус
        status = f"Points: {len(tracker)}/{HISTORY_SIZE} | Mode: {mode}"
        cv2.putText(frame, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Apple Prediction - Kalman Acceleration", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.release()
    print("Программа завершена.")


if __name__ == "__main__":
    run_realtime()