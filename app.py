import streamlit as st
import av
import cv2
import numpy as np
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from liveness import is_real_face
import asyncio
import sys

# Fix asyncio issue on Windows (optional for Linux/Cloud)
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

st.set_page_config(page_title="Real-time Emotion Detection", layout="wide")
st.title("🧠 Real-time Emotion Detection with Anti-Spoofing")

# RTC config for Streamlit Cloud
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Define the video processor using recv()
class EmotionLivenessDetector(VideoProcessorBase):
    def __init__(self):
        self.emotion = "Analyzing..."
        self.score = 0.0
        self.live = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        try:
            # Liveness Detection
            self.live = is_real_face(img)

            if self.live:
                # Emotion Detection
                res = DeepFace.analyze(
                    img,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                self.emotion = res[0]['dominant_emotion']
                self.score = res[0]['emotion'][self.emotion]
                label = f"{self.emotion} ({self.score:.1f}%) | Real Face ✅"
                color = (0, 255, 0)
            else:
                label = "Fake Face ❌"
                color = (0, 0, 255)

            # Draw result
            cv2.putText(img, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        except Exception as e:
            print("Error during processing:", e)
            cv2.putText(img, "Error detecting face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Stream video and apply the processor
webrtc_streamer(
    key="emotion-stream",
    video_processor_factory=EmotionLivenessDetector,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)

st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)
