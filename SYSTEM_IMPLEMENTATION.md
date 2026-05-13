# COMPLETE SYSTEM IMPLEMENTATION GUIDE
# Bharatanatyam Mudra Detection - FastAPI + MediaPipe + ML

## ============================================================
## 1. FEATURE EXTRACTOR (backend/app/services/feature_extractor.py)
## ============================================================
"""
Feature extraction from hand landmarks - CONSISTENT for training and inference
"""
import numpy as np


def extract_features(landmarks):
    """
    Convert 21 hand landmarks into a feature vector of size 63 (21 * 3).
    Normalizes landmarks relative to wrist (landmark 0).
    
    CRITICAL: This is the EXACT function used in BOTH training and inference.
    
    Args:
        landmarks: List of dicts with 'x', 'y', 'z' keys
        
    Returns:
        List of 63 floats, or None if invalid
    """
    if not landmarks or len(landmarks) < 21:
        return None
    
    try:
        # Convert to numpy array: 21 landmarks × 3 coordinates
        coords = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        
        # Normalize relative to wrist (landmark 0)
        wrist = coords[0]
        normalized = coords - wrist
        
        # Flatten to feature vector: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
        features = normalized.flatten().tolist()
        
        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None


## ============================================================
## 2. MEDIAPIPE SERVICE (backend/app/services/mediapipe_service.py)
## ============================================================
"""
MediaPipe hand detection service
Uses consistent configuration for real-time webcam processing
"""
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands with specific configuration
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def extract_hand_landmarks(image):
    """
    Extract hand landmarks from an image.
    
    Args:
        image: OpenCV image in BGR format
        
    Returns:
        List of hand landmarks. Each hand has 21 landmarks with x, y, z.
        Returns empty list if no hands detected.
    """
    try:
        # Convert BGR to RGB (MediaPipe requirement)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process the image
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True
        
        hands_landmarks = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = []
                for lm in hand_landmarks.landmark:
                    hand_data.append({
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z)
                    })
                hands_landmarks.append(hand_data)
        
        return hands_landmarks
    
    except Exception as e:
        print(f"Error extracting landmarks: {e}")
        return []


## ============================================================
## 3. API ROUTES (backend/app/routes/mudra.py)
## ============================================================
"""
FastAPI endpoints for mudra detection and prediction
"""
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


@router.post("/predict-frame")
async def predict_frame(file: UploadFile = File(...)):
    """
    Predict mudra from a webcam frame.
    
    - Reads image frame
    - Detects hand landmarks using MediaPipe (SAME as training)
    - Extracts features using feature_extractor (SAME as training)
    - Predicts mudra and confidence
    
    Returns:
        {
            "mudra": "Anjali",
            "confidence": 0.87
        }
    """
    try:
        # Read image
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"mudra": "No Hand", "confidence": 0}
        
        # Extract landmarks using MediaPipe (SAME as training)
        hands_landmarks = extract_hand_landmarks(image)
        
        if not hands_landmarks or len(hands_landmarks) == 0:
            return {"mudra": "No Hand", "confidence": 0}
        
        # Use first hand
        landmarks = hands_landmarks[0]
        
        # Extract features (EXACT SAME function as training)
        features = extract_features(landmarks)
        
        if features is None:
            return {"mudra": "Error", "confidence": 0}
        
        # Predict mudra
        mudra_label, confidence = mudra_model.predict_mudra(features)
        
        if mudra_label is None:
            return {"mudra": "Error", "confidence": 0}
        
        return {
            "mudra": mudra_label,
            "confidence": float(confidence)
        }
    
    except Exception as e:
        logger.error(f"Error in predict_frame: {e}")
        return {"mudra": "Error", "confidence": 0}


## ============================================================
## 4. FRONTEND (frontend/webcam.html)
## ============================================================
# See frontend/webcam.html
# - Real-time webcam with mirror view
# - Guide box for hand positioning
# - Majority voting over last 5 frames
# - Prediction locking when stable
# - Response time tracking
# - Frame buffer visualization

## ============================================================
## DEPLOYMENT STEPS
## ============================================================

# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Train model
cd backend
python -m app.services.train_model

# 3. Start API server
uvicorn app.main:app --reload --port 8000

# 4. Open frontend
# Visit: frontend/webcam.html in browser

## ============================================================
## VERIFICATION CHECKLIST
## ============================================================

✓ Feature extraction consistent between training/inference
✓ MediaPipe configuration same throughout
✓ Model saves to correct path
✓ Label map JSON created
✓ API endpoint accepts image files
✓ Frontend sends frames correctly
✓ Majority voting implemented
✓ Prediction locking implemented
✓ Response time tracking working
✓ Frame buffer visualization ready
