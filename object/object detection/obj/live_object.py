import cv2
import os

ORB = cv2.ORB_create(nfeatures=1000)
BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

KNOWN_DIR = "known_objects"

known_descriptors = []
known_labels = []

print("Loading known objects...")

for label in os.listdir(KNOWN_DIR):
    path = os.path.join(KNOWN_DIR, label)
    if not os.path.isdir(path):
        continue

    for img_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_name), 0)
        if img is None:
            continue

        kp, des = ORB.detectAndCompute(img, None)
        if des is not None:
            known_descriptors.append(des)
            known_labels.append(label)

print("Objects loaded.")

cap = cv2.VideoCapture(0)
print("Live camera started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = ORB.detectAndCompute(gray, None)

    label = "Unknown"
    best_matches = 0

    if des2 is not None:
        for des1, lbl in zip(known_descriptors, known_labels):
            matches = BF.match(des1, des2)
            good = [m for m in matches if m.distance < 50]

            if len(good) > best_matches and len(good) > 20:
                best_matches = len(good)
                label = lbl

    cv2.putText(frame, label, (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("ORB Object Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
