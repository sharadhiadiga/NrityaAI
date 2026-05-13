import numpy as np
import mediapipe as mp
import cv2
import pandas as pd
import joblib
import os

mp_pose = mp.solutions.pose

# ---------------- LOAD MODEL ----------------

MODEL_PATH = "pose_model.pkl"

model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
else:
    print(f"Model not found at {MODEL_PATH}")


# ---------------- DISTANCE ----------------

def distance(a, b):
    return np.linalg.norm(
        np.array([a["x"], a["y"]]) -
        np.array([b["x"], b["y"]])
    )


# ---------------- ANGLE ----------------

def calculate_angle(a, b, c):

    a = np.array([a["x"], a["y"]])
    b = np.array([b["x"], b["y"]])
    c = np.array([c["x"], c["y"]])

    ba = a - b
    bc = c - b

    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 0

    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.arccos(cosine)

    return np.degrees(angle)


# ---------------- LANDMARK EXTRACTION ----------------

def extract_landmarks(image):

    with mp_pose.Pose(static_image_mode=True) as pose:

        results = pose.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )

        if not results.pose_landmarks:
            return None

        landmarks = []

        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                "x": lm.x,
                "y": lm.y
            })

        return landmarks


# ---------------- FEATURE EXTRACTION ----------------

def get_features(landmarks):

    features = {}

    # ---------------- CORE ANGLES ----------------

    features["left_knee"] = calculate_angle(
        landmarks[23], landmarks[25], landmarks[27]
    )

    features["right_knee"] = calculate_angle(
        landmarks[24], landmarks[26], landmarks[28]
    )

    features["left_elbow"] = calculate_angle(
        landmarks[11], landmarks[13], landmarks[15]
    )

    features["right_elbow"] = calculate_angle(
        landmarks[12], landmarks[14], landmarks[16]
    )

    features["left_hip"] = calculate_angle(
        landmarks[11], landmarks[23], landmarks[25]
    )

    features["right_hip"] = calculate_angle(
        landmarks[12], landmarks[24], landmarks[26]
    )

    features["left_shoulder"] = calculate_angle(
        landmarks[13], landmarks[11], landmarks[23]
    )

    features["right_shoulder"] = calculate_angle(
        landmarks[14], landmarks[12], landmarks[24]
    )

    # ---------------- BODY DISTANCES ----------------

    shoulder_width = distance(
        landmarks[11],
        landmarks[12]
    )

    hip_width = distance(
        landmarks[23],
        landmarks[24]
    )

    knee_width = distance(
        landmarks[25],
        landmarks[26]
    )

    ankle_width = distance(
        landmarks[27],
        landmarks[28]
    )

    wrist_width = distance(
        landmarks[15],
        landmarks[16]
    )

    if shoulder_width == 0:
        shoulder_width = 1

    features["hip_width_norm"] = hip_width / shoulder_width
    features["knee_width_norm"] = knee_width / shoulder_width
    features["ankle_width_norm"] = ankle_width / shoulder_width
    features["wrist_width_norm"] = wrist_width / shoulder_width

    # ---------------- SYMMETRY ----------------

    features["knee_symmetry"] = abs(
        features["left_knee"] -
        features["right_knee"]
    )

    features["elbow_symmetry"] = abs(
        features["left_elbow"] -
        features["right_elbow"]
    )

    # ---------------- TORSO FEATURES ----------------

    features["shoulder_slope"] = abs(
        landmarks[11]["y"] -
        landmarks[12]["y"]
    )

    features["hip_slope"] = abs(
        landmarks[23]["y"] -
        landmarks[24]["y"]
    )

    # ---------------- NORMALIZED LANDMARKS ----------------

    center_x = (
        landmarks[23]["x"] +
        landmarks[24]["x"]
    ) / 2

    center_y = (
        landmarks[23]["y"] +
        landmarks[24]["y"]
    ) / 2

    for i, lm in enumerate(landmarks):

        features[f"x_{i}"] = (
            lm["x"] - center_x
        ) / shoulder_width

        features[f"y_{i}"] = (
            lm["y"] - center_y
        ) / shoulder_width

    return features


# ---------------- ML CLASSIFICATION ----------------

def classify_pose(features):

    if model is None:
        print("Model not loaded, returning none")
        return {
            "pose": "none",
            "confidence": 0,
            "matched": False
        }

    X = pd.DataFrame([features])

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    confidence = float(np.max(probabilities))

    pose = prediction if confidence >= 0.60 else "none"

    matched = confidence >= 0.60

    print(f"Prediction: {prediction}, Confidence: {confidence:.2f}, Final pose: {pose}")

    return {
        "pose": pose,
        "confidence": round(confidence, 2),
        "matched": matched
    }


# ---------------- FEEDBACK ----------------

def generate_feedback(features):

    feedback = []

    if (
        features["left_knee"] > 100 or
        features["right_knee"] > 100
    ):
        feedback.append("Bend your knees more")

    if (
        features["shoulder_slope"] > 0.05
    ):
        feedback.append("Keep shoulders balanced")

    if (
        features["hip_slope"] > 0.05
    ):
        feedback.append("Keep hips aligned")

    if (
        features["knee_symmetry"] > 15
    ):
        feedback.append("Balance both legs evenly")

    return feedback


# ---------------- MAIN ANALYSIS ----------------

def analyze_posture(landmarks):

    features = get_features(landmarks)

    classification = classify_pose(features)

    # Improved posture scoring with weights
    # Prioritize: knee geometry (30%), symmetry (25%), hip alignment (20%), spine alignment (15%), other (10%)
    knee_score = max(0, 100 - (abs(features["left_knee"] - 90) + abs(features["right_knee"] - 90)) / 2)
    symmetry_score = max(0, 100 - (features["knee_symmetry"] + features["elbow_symmetry"]) / 2)
    hip_score = max(0, 100 - features["hip_slope"] * 1000)  # Normalize hip slope
    spine_score = max(0, 100 - features["shoulder_slope"] * 1000)  # Normalize shoulder slope
    other_score = max(0, 100 - (features["left_elbow"] + features["right_elbow"]) / 2)  # Elbow angles

    posture_score = (
        knee_score * 0.3 +
        symmetry_score * 0.25 +
        hip_score * 0.2 +
        spine_score * 0.15 +
        other_score * 0.1
    )

    feedback = generate_feedback(features)

    result = {
        "pose": classification["pose"],
        "confidence": classification["confidence"],
        "matched": classification["matched"],
        "posture_score": round(posture_score, 2),
        "feedback": feedback
    }

    print(f"Analysis result: {result}")

    return result