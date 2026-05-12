from fastapi import APIRouter, UploadFile, File
import numpy as np
import cv2
import logging

from app.services.mediapipe_service import extract_hand_landmarks
from app.services.feature_extractor import extract_features
from app.services import mudra_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mudra", tags=["mudra"])

# Load model on startup
mudra_model.load_model()


@router.post("/detect-hand")
async def detect_hand(file: UploadFile = File(...)):
    """Detect hand landmarks from an uploaded image."""
    
    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    hands_landmarks = extract_hand_landmarks(image)

    return {
        "hands_detected": len(hands_landmarks),
        "landmarks": hands_landmarks
    }


@router.post("/predict-mudra")
async def predict_mudra(file: UploadFile = File(...)):
    """Predict mudra class from an uploaded hand image."""
    
    try:
        contents = await file.read()

        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Step 1: Extract landmarks
        hands_landmarks = extract_hand_landmarks(image)

        if not hands_landmarks:
            return {
                "error": "No hand detected",
                "mudra": None,
                "confidence": None
            }

        # Step 2: Use first detected hand
        landmarks = hands_landmarks[0]

        # Step 3: Extract features
        features = extract_features(landmarks)

        if features is None:
            return {
                "error": "Could not extract features",
                "mudra": None,
                "confidence": None
            }

        # Step 4: Predict
        mudra_label, confidence = mudra_model.predict_mudra(features)

        if mudra_label is None:
            return {
                "error": "Model prediction failed",
                "mudra": None,
                "confidence": None
            }

        # Step 5: Clean label
        mudra_label = mudra_label.split("(")[0]

        # Step 6: Confidence threshold
        if confidence < 0.6:
            return {
                "mudra": "Unknown",
                "confidence": confidence
            }

        # Step 7: Final response
        return {
            "mudra": mudra_label,
            "confidence": confidence
        }

    except Exception as e:
        logger.error(f"Error in predict_mudra: {e}")
        return {
            "error": str(e),
            "mudra": None,
            "confidence": None
        }