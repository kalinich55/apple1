from ultralytics import YOLO

print("Загружаем модель...")
model = YOLO('best.pt')  # ← используем существующую модель

print("Дообучаем на яблоках...")
model.train(
    data='C:/Users/kalin/Desktop/apple11/apple1/dataset/data.yaml',
    epochs=20,           # можно увеличить до 50, если нужно
    imgsz=640,
    batch=16,
    device='cpu',
    lr0=0.001,           # меньшая скорость обучения (для дообучения)
    verbose=True
)

print("✅ Модель сохранена в runs/segment/train/weights/best.pt")