from ultralytics import YOLO

model = YOLO("best.pt")
print("Классы модели:")
print(model.names)