import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = "E:/NrityaAI/backend/app/models/mudra_model.pkl"
LABEL_ENCODER_PATH = "app/models/label_encoder.pkl"
DATA_PATH = "datasets/mudra_data.csv"

# Global model and label encoder
_model = None
_label_encoder = None


def load_model():
    """Load trained model and label encoder."""
    global _model, _label_encoder
    
    try:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
        
        if os.path.exists(LABEL_ENCODER_PATH):
            _label_encoder = joblib.load(LABEL_ENCODER_PATH)
        
        return _model is not None and _label_encoder is not None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def train_model():
    """Train model from CSV dataset."""
    print("Loading dataset...")

    if not os.path.exists(DATA_PATH):
        print("Dataset not found!")
        return False

    data = pd.read_csv(DATA_PATH)

    if data.empty:
        print("Dataset is empty!")
        return False

    X = data.drop("label", axis=1)
    y = data["label"]

    print("Training model...")

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    os.makedirs("app/models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved at {MODEL_PATH}")
    return True


def predict_mudra(features):
    import os

    print("Checking model path:", MODEL_PATH)
    print("Exists?", os.path.exists(MODEL_PATH))

    if not os.path.exists(MODEL_PATH):
        print("Model file not found!")
        return None, None

    model = joblib.load(MODEL_PATH)

    prediction = model.predict([features])[0]
    confidence = max(model.predict_proba([features])[0])

    return prediction, float(confidence)