"""
Feature Extraction Service
"""

import numpy as np


def extract_features(landmarks):
    """
    Convert 21 landmarks into normalized feature vector
    """

    if not landmarks or len(landmarks) < 21:
        return None

    try:
        # Convert to numpy array
        coords = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks])

        # Wrist (reference point)
        wrist = coords[0]

        # Normalize relative to wrist
        normalized = coords - wrist

        # Flatten (21 * 3 = 63 features)
        features = normalized.flatten().tolist()

        return features

    except Exception as e:
        print("Feature extraction error:", e)
        return None