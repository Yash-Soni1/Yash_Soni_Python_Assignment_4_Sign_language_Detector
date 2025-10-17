import cv2
import numpy as np

class OpenCVHandCropper:
    def __init__(self):
        # HSV range for skin detection (adjustable depending on lighting/skin tone)
        self.lower = np.array([0, 30, 60], dtype=np.uint8)
        self.upper = np.array([20, 150, 255], dtype=np.uint8)

    def crop_hand(self, frame, img_size=160):
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        # Largest contour = hand
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 5000:  # skip small noise
            return None

        x, y, w, h = cv2.boundingRect(largest_contour)
        crop = frame[y:y+h, x:x+w]
        crop = cv2.resize(crop, (img_size, img_size))
        return crop
