import cv2
from ultralytics import YOLO

print("Загружаем модель YOLO...")
model = YOLO('yolov8n.pt')
print("Модель загружена!")

print("Открываем камеру...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ОШИБКА: Камера не найдена!")
    print("Проверь, что камера подключена и не используется другой программой")
    exit()

print("Камера работает!")
print("Нажмите 'q' для выхода")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Детекция объектов
    results = model(frame)
    
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # apple = класс 47 в COCO
            if model.names[class_id] == 'apple' and confidence > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Рисуем рамку
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Рисуем центр
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # Текст
                cv2.putText(frame, f"Apple {confidence:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Выводим координаты в консоль
                print(f"Яблоко: ({cx}, {cy})")
    
    cv2.imshow('Apple Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Программа завершена")