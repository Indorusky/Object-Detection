import joblib
from sklearn.neighbors import KNeighborsClassifier

features, labels = joblib.load("object_features.pkl")

model = KNeighborsClassifier(n_neighbors=3)
model.fit(features, labels)

joblib.dump(model, "object_model.pkl")
print("Object recognition model trained.")
