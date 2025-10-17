import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils.hand_cropper_opencv import OpenCVHandCropper
from utils.label_utils import load_label_map

# Load trained model
model_path = "models/signclassifier.h5"
model = load_model(model_path)

# Load labels dynamically from saved JSON and fix string keys
label_map = load_label_map("models/labels.json")
label_map = {int(k): v for k, v in label_map.items()}  # convert string keys to int
labels = [label_map[i] for i in sorted(label_map.keys())]  # ensure correct order

# Hand cropper
cropper = OpenCVHandCropper()
cap = cv2.VideoCapture(0)

print("🎥 Starting real-time sign detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hand_crop = cropper.crop_hand(frame)

    if hand_crop is not None:
        img = hand_crop.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        preds = model.predict(img, verbose=0)
        pred_label = labels[np.argmax(preds)]
        confidence = np.max(preds) * 100
        cv2.putText(frame, f"{pred_label} ({confidence:.1f}%)", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    else:
        cv2.putText(frame, "No hand detected", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
