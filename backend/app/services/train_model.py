"""
Complete Training Pipeline for Bharatanatyam Mudra Classification
- Load dataset from local folder structure
- Extract features using MediaPipe (same as inference)
- Train RandomForestClassifier with 80/20 split
- Save model and label map
"""

import os
import json
import cv2
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import mediapipe as mp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "app/models/mudra_model.pkl"
LABEL_MAP_PATH = "app/models/label_map.json"

# MediaPipe Hands - SAME config as inference
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def extract_landmarks_from_image(image):
    """Extract 21 hand landmarks from image using MediaPipe."""
    try:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True
        
        # Extract landmarks
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.append({
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z)
                })
            return landmarks
        
        return None
    except Exception as e:
        logger.debug(f"Error extracting landmarks: {e}")
        return None


def extract_features(landmarks):
    """Extract normalized features - EXACT SAME function as feature_extractor.py."""
    if not landmarks or len(landmarks) < 21:
        return None
    
    try:
        coords = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        wrist = coords[0]
        normalized = coords - wrist
        features = normalized.flatten().tolist()
        return features
    except Exception as e:
        logger.debug(f"Error extracting features: {e}")
        return None


def process_dataset(dataset_path):
    """Process images and build feature vectors."""
    if not dataset_path or not os.path.exists(dataset_path):
        logger.error(f"Dataset path not found: {dataset_path}")
        return [], []
    
    logger.info(f"Processing dataset from: {dataset_path}")
    
    features_list = []
    labels_list = []
    total = 0
    processed = 0
    skipped = 0
    
    # Find mudra class directories
    class_dirs = [(d, os.path.join(dataset_path, d)) 
                  for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
    
    logger.info(f"Found {len(class_dirs)} mudra classes")
    
    for class_name, class_path in sorted(class_dirs):
        logger.info(f"  Processing '{class_name}'...")
        
        # Find all images (recursive)
        image_files = []
        for root, dirs, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(root, file))
        
        # Process each image
        for image_path in image_files:
            total += 1
            
            try:
                image = cv2.imread(image_path)
                if image is None:
                    skipped += 1
                    continue
                
                landmarks = extract_landmarks_from_image(image)
                if landmarks is None:
                    skipped += 1
                    continue
                
                features = extract_features(landmarks)
                if features is None:
                    skipped += 1
                    continue
                
                features_list.append(features)
                labels_list.append(class_name)
                processed += 1
                
            except Exception as e:
                logger.debug(f"Error: {e}")
                skipped += 1
                continue
    
    logger.info("\n" + "="*60)
    logger.info("DATASET PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total images:       {total}")
    logger.info(f"Processed:          {processed}")
    logger.info(f"Skipped:            {skipped}")
    if total > 0:
        logger.info(f"Success rate:       {processed/total*100:.1f}%")
    logger.info("="*60 + "\n")
    
    return features_list, labels_list


def train_and_save_model(features_list, labels_list):
    """Train RandomForest with train/test split (80/20) and save model."""
    if not features_list or not labels_list:
        logger.error("No data to train!")
        return False
    
    logger.info("="*60)
    logger.info("MODEL TRAINING")
    logger.info("="*60)
    
    try:
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Feature size: {X.shape[1]} (21 landmarks × 3 coords)")
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        logger.info(f"Classes: {len(label_encoder.classes_)}")
        logger.info(f"  {', '.join(label_encoder.classes_[:5])}{'...' if len(label_encoder.classes_) > 5 else ''}")
        
        # Train/test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        logger.info(f"\nTrain set: {len(X_train)} samples (80%)")
        logger.info(f"Test set:  {len(X_test)} samples (20%)")
        
        # Train RandomForestClassifier
        logger.info("\nTraining RandomForestClassifier...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        logger.info(f"\n✓ Training accuracy:  {train_acc:.4f} ({train_acc*100:.2f}%)")
        logger.info(f"✓ Test accuracy:      {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Classification report
        y_pred = model.predict(X_test)
        logger.info("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        # Save model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        logger.info(f"\n✓ Model saved to: {MODEL_PATH}")
        
        # Save label encoder as JSON
        label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
        with open(LABEL_MAP_PATH, 'w') as f:
            json.dump(label_map, f, indent=2)
        logger.info(f"✓ Label map saved to: {LABEL_MAP_PATH}")
        
        logger.info("="*60 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return False


def main():
    """Execute complete training pipeline."""
    logger.info("="*60)
    logger.info("BHARATANATYAM MUDRA CLASSIFICATION - TRAINING")
    logger.info("="*60 + "\n")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Load local dataset
    logger.info("[Step 1/3] Loading dataset...")
    dataset_path = "../datasets/Bharatanatyam-Mudra-Dataset-master"
    logger.info(f"Looking for dataset at: {os.path.abspath(dataset_path)}")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at: {dataset_path}")
        return False
    
    # Process dataset
    logger.info("\n[Step 2/3] Processing dataset...")
    features_list, labels_list = process_dataset(dataset_path)
    if not features_list:
        logger.error("No data processed")
        return False
    
    # Train model
    logger.info("\n[Step 3/3] Training model...")
    success = train_and_save_model(features_list, labels_list)
    
    if success:
        logger.info("="*60)
        logger.info("✓ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("Ready for inference with FastAPI")
        logger.info("="*60 + "\n")
        return True
    
    return False


if __name__ == "__main__":
    main()
