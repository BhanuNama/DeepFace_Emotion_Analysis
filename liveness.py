import cv2
import numpy as np

def is_real_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness > 50  # Replace with real EAR/blink if needed
