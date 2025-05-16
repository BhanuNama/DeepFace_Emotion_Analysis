import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from deepface import DeepFace
from liveness import is_real_face
import numpy as np

st.set_page_config(page_title="Emotion Detector (WebRTC)", layout="wide")
st.title("🧠 Real-time Emotion Detection with Anti-Spoofing")

class EmotionLivenessDetector(VideoTransformerBase):
    def __init__(self):
        self.emotion = "Analyzing..."
        self.score = 0.0
        self.live = True

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            self.live = is_real_face(img)

            if self.live:
                res = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                self.emotion = res[0]['dominant_emotion']
                self.score = res[0]['emotion'][self.emotion]
                label = f"{self.emotion} ({self.score:.1f}%) | Real Face ✅"
                color = (0, 255, 0)
            else:
                label = "Fake Face ❌"
                color = (0, 0, 255)

            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2, cv2.LINE_AA)

        except Exception as e:
            print("Error:", e)
            self.emotion = "Error"
            cv2.putText(img, "Error detecting", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        return img

webrtc_streamer(key="emotion-stream", video_processor_factory=EmotionLivenessDetector)
