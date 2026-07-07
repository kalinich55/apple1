import cv2
import time
import numpy as np
from ultralytics import YOLO


class AppleDetector:
    """
    Детекция яблока с камеры через YOLO segmentation.
    Без tracking — только стабильная покадровая детекция.
    """

    def __init__(self, model_path="yolov8n-seg.pt", camera_index=0):
        print("Загружаем модель YOLO...")
        self.model = YOLO(model_path)
        print("Модель загружена!")

        print("Открываем камеру...")
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("ОШИБКА: Камера не найдена!")

        print("Камера работает!")

    def get_detection(self, conf_threshold=0.3):
        """
        Считывает кадр и пытается найти яблоко.

        conf_threshold: минимальный порог confidence.
                        Снижен до 0.3 для лучшего обнаружения.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        # Обычная детекция без tracking
        results = self.model(frame, verbose=False)

        best_detection = None

        for r in results:
            if r.boxes is None:
                continue

            for i, box in enumerate(r.boxes):
                class_id   = int(box.cls[0])
                confidence = float(box.conf[0])

                if self.model.names[class_id] != "apple":
                    continue

                if confidence < conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Пробуем взять центр по маске
                cx, cy, contour = self._get_mask_center(r, i, frame)

                # Если маска не вышла — берем центр bbox
                if cx is None:
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                if best_detection is None or confidence > best_detection["confidence"]:
                    best_detection = {
                        "timestamp"  : time.time(),
                        "x"          : cx,
                        "y"          : cy,
                        "confidence" : confidence,
                        "bbox"       : (x1, y1, x2, y2),
                        "contour"    : contour,
                    }

        return frame, best_detection

    def _get_mask_center(self, result, i, frame):
        """
        Получить центр объекта по маске сегментации.
        Возвращает (cx, cy, contour) или (None, None, None).
        """
        try:
            if result.masks is None:
                return None, None, None

            if i >= len(result.masks.xy):
                return None, None, None

            polygon = result.masks.xy[i]
            if len(polygon) < 3:
                return None, None, None

            contour = polygon.astype(np.int32).reshape(-1, 1, 2)

            M = cv2.moments(contour)
            if M["m00"] == 0:
                return None, None, None

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            return cx, cy, contour

        except Exception:
            return None, None, None

    def draw_detection(self, frame, detection):
        """
        Рисует контур яблока, его центр и confidence.
        """
        if detection is None:
            return frame

        x    = detection["x"]
        y    = detection["y"]
        conf = detection["confidence"]
        x1, y1, x2, y2 = detection["bbox"]
        contour = detection.get("contour")

        # Если есть контур маски — рисуем его
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (255, 100, 0), 2)
        else:
            # Иначе просто рамка
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)

        # Центр яблока
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Подпись
        cv2.putText(
            frame,
            f"Apple {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 100, 0),
            2
        )

        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()