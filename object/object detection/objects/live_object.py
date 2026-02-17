import cv2
import numpy as np
import joblib
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

MODEL = ResNet50(weights="imagenet", include_top=False, pooling="avg")

features, labels = joblib.load("object_features.pkl")
clf = joblib.load("object_model.pkl")

def extract_feature(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return MODEL.predict(img, verbose=0)[0]

cap = cv2.VideoCapture(0)
print("Live Object Recognition started. Press Q to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    roi = frame[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]

    feat = extract_feature(roi)

    sims = cosine_similarity([feat], features)[0]
    best = sims.argmax()

    if sims[best] > 0.54:
        label = labels[best]
    else:
        label = "Unknown"

    cv2.rectangle(frame, (int(w*0.25), int(h*0.25)),
                  (int(w*0.75), int(h*0.75)), (0,255,0), 2)
    cv2.putText(frame, label, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    cv2.imshow("Object Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
