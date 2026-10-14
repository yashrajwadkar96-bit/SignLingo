import numpy as np
import pickle
import json
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

LANDMARKS_DIR = "data/landmarks"
MODEL_PATH = "models/best_model.keras"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

# 1) Load label encoder
with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)
classes = le.classes_
print("Classes:", classes)

# 2) Load model
model = keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# 3) Load metadata
meta_path = os.path.join(LANDMARKS_DIR, "metadata.json")
with open(meta_path, "r") as f:
    metadata = json.load(f)["samples"]

# 4) Load all samples into X, y
X_list = []
y_list = []

video_cache = {}

for label, video_base, num_sequences in metadata:
    npy_path = os.path.join(LANDMARKS_DIR, f"{video_base}.npy")
    
    if video_base not in video_cache:
        video_cache[video_base] = np.load(npy_path, allow_pickle=True)
    
    sequences = video_cache[video_base]
    # num_sequences is the number of sliding-window samples
    for i in range(num_sequences):
        X_list.append(sequences[i])
        y_list.append(label)

X = np.array(X_list)
y = np.array(y_list)
print("Loaded dataset:", X.shape, y.shape)

# 5) Encode labels
y_enc = le.transform(y)

# 6) Predictions
pred_prob = model.predict(X)
pred = pred_prob.argmax(axis=1)

# 7) Metrics
accuracy = accuracy_score(y_enc, pred) * 100
print(f"\nFINAL ACCURACY: {accuracy:.2f}%\n")

print("Classification Report:")
print(classification_report(y_enc, pred, target_names=classes))

print("Confusion Matrix:")
print(confusion_matrix(y_enc, pred))
