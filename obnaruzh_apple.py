import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys

# =========================
# НАСТРОЙКИ
# =========================
MODEL_PATH = r"best.pt"
TARGET_CLASS = "apple"
CONFIDENCE_THRESHOLD = 0.80

print("Загружаем модель...")

if not os.path.exists(MODEL_PATH):
    print(f"ОШИБКА: файл модели не найден: {MODEL_PATH}")
    print("Положи best.pt рядом с obnaruzh_apple.py")
    sys.exit()

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    sys.exit()

print("Модель загружена!")
print("Классы модели:", model.names)

if TARGET_CLASS not in model.names.values():
    print(f"ОШИБКА: класс '{TARGET_CLASS}' не найден в модели!")
    print("Доступные классы:", model.names)
    sys.exit()

print("Открываем камеру...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ОШИБКА: Камера не найдена!")
    sys.exit()

print("Камера работает!")
print("Нажмите 'q' для выхода")
print("-" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка чтения кадра")
        break

    results = model(frame, verbose=False)

    apple_found = False

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        if r.masks is None:
            continue

        boxes = r.boxes
        masks = r.masks.data

        for i, box in enumerate(boxes):
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = model.names[class_id]

            if class_name != TARGET_CLASS or confidence < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            w = x2 - x1
            h = y2 - y1
            ratio = w / h if h != 0 else 0

            # Отсекаем слишком вытянутые объекты
            if ratio < 0.6 or ratio > 1.4:
                continue

            mask = masks[i].cpu().numpy()
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask = (mask > 0.5).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            cnt = max(contours, key=cv2.contourArea)

            area = cv2.contourArea(cnt)
            frame_area = frame.shape[0] * frame.shape[1]

            # Отсекаем слишком маленькие и слишком большие объекты
            if area < 500:
                continue

            if area > frame_area * 0.25:
                continue

            apple_found = True

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

            overlay = frame.copy()
            overlay[mask == 1] = (0, 255, 0)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            cv2.drawContours(frame, [cnt], -1, (255, 0, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"Apple {confidence:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"Center: ({cx}, {cy})",
                (x1, min(frame.shape[0] - 10, y2 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            print(f"Яблоко найдено | Центр: ({cx}, {cy}) | Уверенность: {confidence:.2f}")

    if not apple_found:
        cv2.putText(
            frame,
            "Apple not found",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Apple Segmentation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Программа завершена")