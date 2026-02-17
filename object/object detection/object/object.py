import cv2
from ultralytics import YOLO
import os

# -------------------------------
# SILENCE YOLO LOGS (IMPORTANT)
# -------------------------------
os.environ["YOLO_VERBOSE"] = "False"

# Load YOLOv8 nano model (fastest)
model = YOLO(r"C:\runs\detect\train2\weights\best.pt")

model.verbose = False   # 🔇 no terminal output

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference (no printing)
    results = model(frame, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            cls = int(box.cls[0])
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

    cv2.imshow("YOLO Live Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
