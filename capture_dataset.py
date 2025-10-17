import argparse
import cv2
from pathlib import Path
import numpy as np

# Change this list to include all gestures you want to capture
GESTURES = ["hello", "thanks", "yes", "no", "sorry", "stop", "friends"]

class OpenCVHandCropper:
    """Simple skin-color based hand cropper"""
    def __init__(self):
        pass

    def crop_hand(self, frame, img_size=160):
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        # Morphology
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        # Largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (img_size, img_size))
        return crop

def main(out_dir='dataset', img_size=160):
    out_dir = Path(out_dir)
    for label in GESTURES:
        (out_dir / label).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    current_label_idx = 0
    counts = {label: len(list((out_dir / label).glob('*.jpg'))) for label in GESTURES}
    print("Starting capture. Press keys 1-{} to change gesture. C: capture, Q: quit.".format(len(GESTURES)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        cropper = OpenCVHandCropper()
        hand_crop = cropper.crop_hand(frame, img_size=img_size)

        label = GESTURES[current_label_idx]
        cv2.putText(display, f"Gesture: {label} (1-{len(GESTURES)} to change)", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(display, "C: capture | Q: quit", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        if hand_crop is not None:
            # show crop in top-right corner
            h, w = display.shape[:2]
            crop_h = int(h * 0.4)
            crop_vis = cv2.resize(hand_crop, (crop_h, crop_h))
            display[10:10+crop_h, w-10-crop_h:w-10] = crop_vis

        cv2.imshow("Dataset Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord(str(i)) for i in range(1, len(GESTURES)+1)]:
            current_label_idx = int(chr(key)) - 1
        elif key == ord('c'):
            if hand_crop is None:
                print("No hand detected. Move hand into view.")
                continue
            label_dir = out_dir / label
            next_idx = len(list(label_dir.glob("*.jpg"))) + 1
            filename = label_dir / f"{label}_{next_idx:04d}.jpg"
            cv2.imwrite(str(filename), hand_crop)
            print("Saved", filename)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='dataset')
    parser.add_argument('--img_size', type=int, default=160)
    args = parser.parse_args()
    main(out_dir=args.out_dir, img_size=args.img_size)
