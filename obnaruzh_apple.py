import cv2
from ultralytics import YOLO
import numpy as np

print("Загружаем модель YOLO с сегментацией...")
model = YOLO('yolov8n-seg.pt')  # ← ГЛАВНОЕ ИЗМЕНЕНИЕ
print("Модель загружена!")

print("Открываем камеру...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ОШИБКА: Камера не найдена!")
    exit()

print("Камера работает!")
print("Нажмите 'q' для выхода")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Детекция + сегментация
    results = model(frame)
    
    for r in results:
        # Проверяем, есть ли маски (сегментация)
        if r.masks is not None:
            for i, box in enumerate(r.boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if model.names[class_id] == 'apple' and confidence > 0.5:
                    # Получаем координаты рамки
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    # ========================================
                    # ГЛАВНОЕ: РИСУЕМ МАСКУ (точный контур)
                    # ========================================
                    mask = r.masks[i].data[0].cpu().numpy()
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask = (mask > 0.5).astype(np.uint8)
                    
                    # Находим контуры маски
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Рисуем контур яблока (голубой)
                    cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)
                    
                    # Заливка маски (полупрозрачная)
                    overlay = frame.copy()
                    overlay[mask == 1] = [0, 255, 0]  # Зелёная заливка
                    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
                    
                    # Рисуем центр (красная точка)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # Текст
                    cv2.putText(frame, f"Apple {confidence:.2f}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    print(f"Яблоко: ({cx}, {cy})")
    
    cv2.imshow('Apple Detector (Segmentation)', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Программа завершена")