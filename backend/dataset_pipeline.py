"""
Automated Dataset Pipeline for Mudra Classification
- Downloads dataset from Kaggle
- Processes images and extracts hand landmarks
- Generates feature vectors
- Trains RandomForest model
"""

import os
import csv
import logging
import cv2
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Import MediaPipe
import mediapipe as mp

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
KAGGLE_DATASET = "onkarpathrikar/bharatanatyam-mudra-dataset-master-bm"
OUTPUT_CSV = "datasets/mudra_data.csv"
MODEL_PATH = "app/models/mudra_model.pkl"
LABEL_ENCODER_PATH = "app/models/label_encoder.pkl"

# MediaPipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)


def download_dataset():
    """Download dataset from Kaggle using kagglehub."""
    try:
        import kagglehub
        logger.info(f"Downloading dataset: {KAGGLE_DATASET}")
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        logger.info(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return None


def extract_hand_landmarks(image):
    """Extract 21 hand landmarks from image using MediaPipe."""
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z
                })
            return landmarks
        return None
    except Exception as e:
        logger.debug(f"Error extracting landmarks: {e}")
        return None


def extract_features(landmarks):
    """Convert 21 landmarks into normalized feature vector (63 features)."""
    if not landmarks or len(landmarks) < 21:
        return None
    
    try:
        coords = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks])
        wrist = coords[0]  # Reference point
        normalized = coords - wrist
        features = normalized.flatten().tolist()
        return features
    except Exception as e:
        logger.debug(f"Error in feature extraction: {e}")
        return None


def process_dataset(dataset_path):
    """Process dataset and generate CSV with features and labels."""
    if not dataset_path or not os.path.exists(dataset_path):
        logger.error(f"Dataset path not found: {dataset_path}")
        return 0, 0, 0
    
    logger.info(f"Processing dataset from: {dataset_path}")
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    total_images = 0
    processed_images = 0
    skipped_images = 0
    
    # Find class folders
    class_folders = [
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]
    
    logger.info(f"Found {len(class_folders)} mudra classes")
    
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = None
        
        for class_name in sorted(class_folders):
            class_path = os.path.join(dataset_path, class_name)
            image_files = [
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            logger.info(f"Processing class '{class_name}': {len(image_files)} images")
            
            for image_file in image_files:
                total_images += 1
                image_path = os.path.join(class_path, image_file)
                
                try:
                    # Read image
                    image = cv2.imread(image_path)
                    if image is None:
                        logger.debug(f"Could not read: {image_file}")
                        skipped_images += 1
                        continue
                    
                    # Extract landmarks
                    landmarks = extract_hand_landmarks(image)
                    if landmarks is None:
                        logger.debug(f"No hand detected: {image_file}")
                        skipped_images += 1
                        continue
                    
                    # Extract features
                    features = extract_features(landmarks)
                    if features is None:
                        logger.debug(f"Feature extraction failed: {image_file}")
                        skipped_images += 1
                        continue
                    
                    # Initialize CSV writer on first valid sample
                    if writer is None:
                        fieldnames = [f'f{i+1}' for i in range(63)] + ['label']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                    
                    # Write row
                    row = {f'f{i+1}': features[i] for i in range(63)}
                    row['label'] = class_name
                    writer.writerow(row)
                    
                    processed_images += 1
                    
                except Exception as e:
                    logger.debug(f"Error processing {image_file}: {e}")
                    skipped_images += 1
                    continue
    
    return total_images, processed_images, skipped_images


def train_model_from_csv():
    """Train RandomForest model from CSV dataset."""
    logger.info("Loading dataset from CSV...")
    
    if not os.path.exists(OUTPUT_CSV):
        logger.error(f"CSV file not found: {OUTPUT_CSV}")
        return False
    
    try:
        import pandas as pd
        
        # Load data
        data = pd.read_csv(OUTPUT_CSV)
        
        if data.empty:
            logger.error("Dataset is empty!")
            return False
        
        logger.info(f"Loaded {len(data)} samples")
        
        # Prepare features and labels
        X = data.drop("label", axis=1).values
        y = data["label"].values
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        logger.info(f"Training RandomForestClassifier with {len(np.unique(y_encoded))} classes...")
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y_encoded)
        
        # Save model and encoder
        os.makedirs("app/models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(label_encoder, LABEL_ENCODER_PATH)
        
        logger.info(f"Model saved to: {MODEL_PATH}")
        logger.info(f"Label encoder saved to: {LABEL_ENCODER_PATH}")
        
        # Print model info
        logger.info(f"Model accuracy on training data: {model.score(X, y_encoded):.3f}")
        logger.info(f"Classes: {label_encoder.classes_}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return False


def main():
    """Main pipeline execution."""
    logger.info("=" * 60)
    logger.info("MUDRA CLASSIFICATION DATASET PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Download dataset
    logger.info("\n[Step 1] Downloading dataset...")
    dataset_path = download_dataset()
    
    if not dataset_path:
        logger.error("Failed to download dataset")
        return False
    
    # Step 2: Process dataset
    logger.info("\n[Step 2] Processing dataset...")
    total, processed, skipped = process_dataset(dataset_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("DATASET PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total images found:     {total}")
    logger.info(f"Successfully processed: {processed}")
    logger.info(f"Skipped/Failed:         {skipped}")
    logger.info(f"Success rate:           {processed/total*100:.1f}% (if total > 0)")
    logger.info(f"CSV saved to:           {OUTPUT_CSV}")
    logger.info("=" * 60)
    
    if processed == 0:
        logger.error("No images were successfully processed!")
        return False
    
    # Step 3: Train model
    logger.info("\n[Step 3] Training model...")
    success = train_model_from_csv()
    
    if not success:
        logger.error("Failed to train model")
        return False
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Model ready for predictions at: {MODEL_PATH}")
    logger.info("=" * 60 + "\n")
    
    return True


if __name__ == "__main__":
    main()
