import streamlit as st
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer
from posture_engine import analyze_posture
import mediapipe as mp

st.title("PostureSense AI")
st.write("Real-time posture monitoring")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class PostureProcessor:

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        posture_text, color, angle, results = analyze_posture(img)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="posture",
    video_processor_factory=PostureProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
