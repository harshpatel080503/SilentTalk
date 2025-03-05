import cv2
import numpy as np
import mediapipe as mp
import csv
import pandas as pd
import joblib
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ----------------- FUNCTIONS AND MODEL LOADING (same as before) -----------------
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks_G(image, results):
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=1)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=1)
    )

def draw_styled_landmarks_np_nf_B(image, results):
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=1)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=1)
    )

# Load your model
model_L = joblib.load("MP_model_head.pkl")

# ----------------- VIDEO TRANSFORMER CLASS -----------------
class HandSignTransformer(VideoTransformerBase):
    def __init__(self):
        # Initialize Mediapipe holistic and sign lists once
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, 
                                             min_tracking_confidence=0.5)
        self.predictions = []
        self.sentence = []
        self.sentence_out = []
        self.last_sign_list = []
        self.one_sign_list = []
        # Read CSV files to load sign lists
        with open("multi_sign.csv", newline="") as multisign_file:
            reader = csv.reader(multisign_file)
            for row in reader:
                self.last_sign_list.append(row[-1])
        with open("single_sign.csv", newline="") as singlesign_file:
            reader = csv.reader(singlesign_file)
            for row in reader:
                self.one_sign_list.append(row[0])
    
    def transform(self, frame):
        # frame is a dict with "data" as the image frame in BGR format
        image = frame.copy()
        image, results = mediapipe_detection(image, self.holistic)
        draw_styled_landmarks_np_nf_B(image, results)
        
        # Extract landmarks
        lh_row = list(np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
                      if results.left_hand_landmarks else np.zeros(21*3))
        rh_row = list(np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
                      if results.right_hand_landmarks else np.zeros(21*3))
        head = list(np.zeros(3))
        if results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                if id == 0 and lm.visibility > 0.8:
                    head = [lm.x, lm.y, lm.z]
        row = lh_row + rh_row + head
        X = pd.DataFrame([row])
        sign_class = model_L.predict(X)[0]
        sign_prob = model_L.predict_proba(X)[0]
        threshold = 0.85
        pr = 3
        
        if sign_prob[np.argmax(sign_prob)] > threshold:
            self.predictions.append(sign_class)
            if len(self.predictions) >= pr and self.predictions[-pr:] == [sign_class] * pr:
                if not self.sentence or sign_class != self.sentence[-1]:
                    self.sentence.append(sign_class)
                    draw_styled_landmarks_G(image, results)
                    if sign_class in self.last_sign_list:
                        self.sentence_out.append(sign_class)
                    if sign_class in self.one_sign_list:
                        self.sentence_out.append(sign_class)
        
        if len(self.sentence_out) > 6:
            self.sentence_out = self.sentence_out[-6:]
        if len(self.sentence) > 6:
            self.sentence = self.sentence[-6:]
        
        cv2.rectangle(image, (0, 0), (640, 40), (0, 0, 0), -1)
        cv2.putText(image, ' '.join(self.sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(image, (0, 80), (640, 40), (255, 0, 0), -1)
        cv2.putText(image, ' '.join(self.sentence_out), (3, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return image

# ----------------- STREAMLIT APP MAIN FUNCTION -----------------
st.title("Live Hand Sign Language Detection")
st.write("Allow access to your webcam to start detection.")

webrtc_streamer(key="hand-sign", video_transformer_factory=HandSignTransformer)
