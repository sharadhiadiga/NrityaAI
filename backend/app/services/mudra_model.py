import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)

# Use relative paths
MODEL_PATH = "app/models/mudra_model.pkl"
LABEL_MAP_PATH = "app/models/label_map.json"
DATA_PATH = "datasets/mudra_data.csv"

# Global model and label map
_model = None
_label_map = None


def load_model():
    """Load trained model and label map."""
    global _model, _label_map
    
    try:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
        
        if os.path.exists(LABEL_MAP_PATH):
            with open(LABEL_MAP_PATH, 'r') as f:
                _label_map = json.load(f)
            logger.info(f"Label map loaded from {LABEL_MAP_PATH}")
        
        return _model is not None and _label_map is not None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def predict_mudra(features):
    """Predict mudra label from features.
    
    Args:
        features: List or array of 63 features
        
    Returns:
        Tuple of (mudra_label, confidence) or (None, None) if error
    """
    global _model, _label_map
    
    try:
        if _model is None:
            load_model()
        
        if _model is None or _label_map is None:
            logger.error("Model or label map not loaded")
            return None, None
        
        # Predict
        prediction_encoded = _model.predict([features])[0]
        confidence = float(max(_model.predict_proba([features])[0]))
        
        # Decode label using label_map
        if isinstance(_label_map, dict):
            # label_map format: {"0": "Anjali", "1": "Pataka", ...}
            prediction_label = _label_map.get(str(prediction_encoded), "Unknown")
        else:
            prediction_label = prediction_encoded
        
        return prediction_label, confidence
        
    except Exception as e:
        logger.error(f"Error in predict_mudra: {e}")
        return None, None