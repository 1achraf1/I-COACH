import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load the trained model
def load_classifier_model(model_path):
    return load_model(model_path)

# Extract keypoints from a MediaPipe pose result
def extract_keypoints(results):
    landmarks = results.pose_landmarks.landmark
    keypoints = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
    return keypoints

# Process a frame to get pose landmarks
def detect_pose_in_frame(frame, pose_estimator):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose_estimator.process(image)
    return results

# Predict the exercise class from keypoints
def predict_pose_class(model, keypoints):
    input_data = np.expand_dims(keypoints, axis=0)
    prediction = model.predict(input_data)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_index, confidence

# Draw pose landmarks on frame
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )
    return image