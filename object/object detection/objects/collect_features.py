import os
import cv2
import numpy as np
import joblib
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

MODEL = ResNet50(weights="imagenet", include_top=False, pooling="avg")

KNOWN_DIR = "known_objects"

features = []
labels = []

def extract_feature(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return MODEL.predict(img, verbose=0)[0]

print("Collecting object features...")

for obj in os.listdir(KNOWN_DIR):
    obj_path = os.path.join(KNOWN_DIR, obj)
    if not os.path.isdir(obj_path):
        continue

    for img_name in os.listdir(obj_path):
        img_path = os.path.join(obj_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        feat = extract_feature(img)
        features.append(feat)
        labels.append(obj)

joblib.dump((features, labels), "object_features.pkl")
print("Object features saved.")
