from ultralytics import YOLO

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train on your custom dataset
model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640
)
