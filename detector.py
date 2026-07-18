import cv2
import time
import numpy as np
from ultralytics import YOLO


def estimate_center(mask_center, bbox, confidence):
    """
    Возвращает более устойчивую точку центра объекта.

    Если центроид маски сильно расходится с центром bbox, значит
    маска, вероятно, частично скрыта или смещена; в этом случае
    используем bbox center. Иначе — смешиваем обе точки.
    """
    if mask_center is None:
        x1, y1, x2, y2 = bbox
        return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))

    x1, y1, x2, y2 = bbox
    bbox_cx = (x1 + x2) / 2.0
    bbox_cy = (y1 + y2) / 2.0
    mask_cx, mask_cy = mask_center

    shift = np.hypot(mask_cx - bbox_cx, mask_cy - bbox_cy)
    box_diag = np.hypot(x2 - x1, y2 - y1)

    if box_diag <= 1.0:
        return (float(mask_cx), float(mask_cy))

    if shift > max(18.0, 0.35 * box_diag):
        return (float(bbox_cx), float(bbox_cy))

    if confidence >= 0.9:
        mask_weight = 0.7
    elif confidence >= 0.8:
        mask_weight = 0.6
    else:
        mask_weight = 0.5

    bbox_weight = 1.0 - mask_weight
    return (
        float(mask_cx * mask_weight + bbox_cx * bbox_weight),
        float(mask_cy * mask_weight + bbox_cy * bbox_weight),
    )


class AppleDetector:
    """
    Детекция объекта с камеры через YOLO segmentation.
    Без tracking — только стабильная покадровая детекция.
    """

    def __init__(
        self,
        model_path="best_apple_colab_30epochs.pt",
        target_class_name="apple",
        display_name="Apple",
        camera_index=0,
        debug_timing=True,
        debug_detection=True
    ):
        print("Загружаем модель YOLO...")
        self.model = YOLO(model_path)
        print("Модель загружена!")

        self.target_class_name = target_class_name
        self.display_name = display_name
        self.debug_timing = debug_timing
        self.debug_detection = debug_detection

        print("Классы модели:", self.model.names)
        print(f"Ищем класс: {self.target_class_name}")

        print("Открываем камеру...")
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("ОШИБКА: Камера не найдена!")

        print("Камера работает!")

    def get_detection(self, conf_threshold=0.6):
        """
        Считывает кадр и пытается найти целевой объект.
        """
        t_start = time.time()

        ret, frame = self.cap.read()
        if not ret:
            return None, None

        t_frame = time.time()

        results = self.model(frame, verbose=False)

        best_detection = None

        for r_idx, r in enumerate(results):
            if self.debug_detection:
                print(f"\n[DEBUG] Result #{r_idx}")
                print("  boxes exist:", r.boxes is not None)
                print("  masks exist:", r.masks is not None)

                if r.boxes is not None:
                    print("  num boxes:", len(r.boxes))
                if r.masks is not None and hasattr(r.masks, "xy"):
                    print("  num masks:", len(r.masks.xy))

            if r.boxes is None:
                continue

            for i, box in enumerate(r.boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]

                if self.debug_detection:
                    print(f"    box #{i}: class={class_name}, conf={confidence:.3f}")

                if class_name != self.target_class_name:
                    continue

                if confidence < conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                mask_center = self._get_mask_center(r, i)
                cx, cy, contour = mask_center

                if cx is None or cy is None:
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                else:
                    cx, cy = estimate_center((cx, cy), (x1, y1, x2, y2), confidence)

                if self.debug_detection:
                    if contour is not None:
                        print(f"      -> contour OK, center=({cx}, {cy})")
                    else:
                        print(f"      -> contour NONE, fallback bbox center=({cx}, {cy})")

                if best_detection is None or confidence > best_detection["confidence"]:
                    best_detection = {
                        "timestamp": t_frame,
                        "x": cx,
                        "y": cy,
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2),
                        "contour": contour,
                        "class_name": class_name,
                    }

        t_end = time.time()

        if self.debug_timing:
            dt = t_end - t_start
            fps = 1.0 / dt if dt > 1e-6 else 0.0
            print(f"[Detector] frame_time={dt:.4f}s | fps={fps:.2f}")

        if self.debug_detection and best_detection is None:
            print("[DEBUG] Подходящая детекция не найдена.")

        return frame, best_detection

    def _get_mask_center(self, result, i):
        """
        Получить центр объекта по маске сегментации.
        Возвращает (cx, cy, contour) или (None, None, None).
        """
        try:
            if result.masks is None:
                return None, None, None

            if not hasattr(result.masks, "xy"):
                return None, None, None

            if i >= len(result.masks.xy):
                return None, None, None

            polygon = result.masks.xy[i]
            if polygon is None or len(polygon) < 3:
                return None, None, None

            contour = polygon.astype(np.int32).reshape(-1, 1, 2)

            M = cv2.moments(contour)
            if M["m00"] == 0:
                return None, None, None

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            return cx, cy, contour

        except Exception as e:
            if self.debug_detection:
                print(f"[DEBUG] Ошибка в _get_mask_center: {e}")
            return None, None, None

    def draw_detection(self, frame, detection):
        """
        Рисует контур объекта, его центр и confidence.
        """
        if detection is None:
            return frame

        x = detection["x"]
        y = detection["y"]
        conf = detection["confidence"]
        x1, y1, x2, y2 = detection["bbox"]
        contour = detection.get("contour")

        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (255, 100, 0), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)

        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"{self.display_name} {conf:.2f}",
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