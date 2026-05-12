"""
Complete End-to-End Mudra Classification Pipeline
- Download dataset from Kaggle
- Process images and extract hand landmarks
- Build feature vectors
- Train RandomForestClassifier
- Save model for inference
"""

import os
import cv2
import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import mediapipe as mp

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
KAGGLE_DATASET = "onkarpathrikar/bharatanatyam-mudra-dataset-master-bm"
CSV_OUTPUT = "datasets/mudra_data.csv"
MODEL_OUTPUT = "app/models/mudra_model.pkl"

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)


def download_dataset():
    path = "E:/NrityaAI/datasets/bharatanatyam-mudra-dataset-master"
    print("Using local dataset:", path)
    return path


def extract_landmarks_from_image(image):
    """Extract 21 hand landmarks from image using MediaPipe.
    
    Args:
        image: OpenCV image (BGR)
        
    Returns:
        List of 21 landmarks with {x, y, z} or None if no hand detected
    """
    try:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = hands.process(image_rgb)
        
        # Check if hand detected
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
    """Extract normalized feature vector from landmarks.
    
    Args:
        landmarks: List of 21 landmarks {x, y, z}
        
    Returns:
        List of 63 normalized features (21 * 3) or None if error
    """
    try:
        if not landmarks or len(landmarks) < 21:
            return None
        
        # Convert to numpy array
        coords = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks])
        
        # Normalize relative to wrist (landmark 0)
        wrist = coords[0]
        normalized = coords - wrist
        
        # Flatten to 63-length vector
        features = normalized.flatten().tolist()
        
        return features
    except Exception as e:
        logger.debug(f"Error extracting features: {e}")
        return None


def process_dataset(dataset_path):
    """Process dataset and build feature vectors with labels.
    
    Args:
        dataset_path: Path to dataset root directory
        
    Returns:
        Tuple of (features_list, labels_list)
    """
    if not dataset_path or not os.path.exists(dataset_path):
        logger.error(f"Dataset path not found: {dataset_path}")
        return [], []
    
    logger.info(f"Processing dataset from: {dataset_path}")
    
    features_list = []
    labels_list = []
    total_images = 0
    processed_images = 0
    skipped_images = 0
    
    # Find all subdirectories (mudra classes)
    class_dirs = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            class_dirs.append((item, item_path))
    
    logger.info(f"Found {len(class_dirs)} mudra classes")
    
    # Process each class
    for class_name, class_path in sorted(class_dirs):
        logger.info(f"Processing class: '{class_name}'")
        
        # Find all images in class directory (recursively)
        image_files = []
        for root, dirs, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(root, file))
        
        logger.info(f"  Found {len(image_files)} images")
        
        # Process each image
        for image_path in image_files:
            total_images += 1
            
            try:
                # Read image
                image = cv2.imread(image_path)
                if image is None:
                    logger.debug(f"  Could not read: {image_path}")
                    skipped_images += 1
                    continue
                
                # Extract landmarks
                landmarks = extract_landmarks_from_image(image)
                if landmarks is None:
                    logger.debug(f"  No hand detected: {os.path.basename(image_path)}")
                    skipped_images += 1
                    continue
                
                # Extract features
                features = extract_features(landmarks)
                if features is None:
                    logger.debug(f"  Feature extraction failed: {os.path.basename(image_path)}")
                    skipped_images += 1
                    continue
                
                # Add to lists
                features_list.append(features)
                labels_list.append(class_name)
                processed_images += 1
                
            except Exception as e:
                logger.debug(f"  Error processing {image_path}: {e}")
                skipped_images += 1
                continue
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("DATASET PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total images found:       {total_images}")
    logger.info(f"Successfully processed:   {processed_images}")
    logger.info(f"Skipped/Failed:           {skipped_images}")
    if total_images > 0:
        logger.info(f"Success rate:             {processed_images/total_images*100:.1f}%")
    logger.info("="*60 + "\n")
    
    return features_list, labels_list


def build_csv(features_list, labels_list):
    """Build and save CSV file with features and labels.
    
    Args:
        features_list: List of 63-dimensional feature vectors
        labels_list: List of labels (mudra class names)
        
    Returns:
        DataFrame created
    """
    if not features_list or not labels_list:
        logger.error("No data to save!")
        return None
    
    logger.info(f"Building CSV with {len(features_list)} samples...")
    
    try:
        # Create column names
        feature_cols = [f'f{i+1}' for i in range(63)]
        
        # Create dataframe
        data = {col: [features[i] for features in features_list] for i, col in enumerate(feature_cols)}
        data['label'] = labels_list
        
        df = pd.DataFrame(data)
        
        # Save CSV
        os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
        df.to_csv(CSV_OUTPUT, index=False)
        
        logger.info(f"CSV saved to: {CSV_OUTPUT}")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Classes: {df['label'].nunique()}")
        
        return df
    except Exception as e:
        logger.error(f"Error building CSV: {e}")
        return None


def train_model(df):
    """Train RandomForestClassifier and save model.
    
    Args:
        df: Pandas DataFrame with features and labels
        
    Returns:
        Trained model or None if error
    """
    if df is None or df.empty:
        logger.error("DataFrame is empty!")
        return None
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING MODEL")
    logger.info("="*60)
    
    try:
        # Prepare features and labels
        X = df.drop('label', axis=1).values
        y = df['label'].values
        
        logger.info(f"Training RandomForestClassifier...")
        logger.info(f"  Features shape: {X.shape}")
        logger.info(f"  Labels: {np.unique(y)}")
        logger.info(f"  Classes: {len(np.unique(y))}")
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        
        # Save model
        os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
        joblib.dump(model, MODEL_OUTPUT)
        
        # Print results
        train_accuracy = model.score(X, y)
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Model saved to: {MODEL_OUTPUT}")
        logger.info(f"Training accuracy: {train_accuracy:.3f}")
        logger.info(f"Feature importance (top 5):")
        
        # Get feature importance
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-5:][::-1]
        for idx in top_indices:
            logger.info(f"  f{idx+1}: {importances[idx]:.4f}")
        
        logger.info("="*60 + "\n")
        
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None


def predict_from_image(image):
    """Predict mudra class from an image.
    
    Args:
        image: OpenCV image (BGR)
        
    Returns:
        Dict with {label, confidence} or error message
    """
    try:
        # Check if model exists
        if not os.path.exists(MODEL_OUTPUT):
            return {"error": "Model not found. Run training first."}
        
        # Extract landmarks
        landmarks = extract_landmarks_from_image(image)
        if landmarks is None:
            return {"error": "No hand detected"}
        
        # Extract features
        features = extract_features(landmarks)
        if features is None:
            return {"error": "Could not extract features"}
        
        # Load model
        model = joblib.load(MODEL_OUTPUT)
        
        # Predict
        prediction = model.predict([features])[0]
        confidence = float(max(model.predict_proba([features])[0]))
        
        return {
            "label": prediction,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return {"error": str(e)}


def main():
    """Execute complete pipeline."""
    logger.info("="*60)
    logger.info("MUDRA CLASSIFICATION PIPELINE")
    logger.info("="*60 + "\n")
    
    # Step 1: Download dataset
    logger.info("[Step 1] Downloading dataset...")
    dataset_path = download_dataset()
    
    if not dataset_path:
        logger.error("Failed to download dataset")
        return False
    
    # Step 2: Process dataset
    logger.info("\n[Step 2] Processing dataset...")
    features_list, labels_list = process_dataset(dataset_path)
    
    if not features_list:
        logger.error("No data processed")
        return False
    
    # Step 3: Build CSV
    logger.info("\n[Step 3] Building CSV...")
    df = build_csv(features_list, labels_list)
    
    if df is None:
        logger.error("Failed to build CSV")
        return False
    
    # Step 4: Train model
    logger.info("\n[Step 4] Training model...")
    model = train_model(df)
    
    if model is None:
        logger.error("Failed to train model")
        return False
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info(f"CSV: {CSV_OUTPUT}")
    logger.info(f"Model: {MODEL_OUTPUT}")
    logger.info("Ready for predictions!")
    logger.info("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    main()
