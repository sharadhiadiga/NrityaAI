"""
Bharatanatyam Mudra Classification Model Training Pipeline
Complete end-to-end training with MediaPipe feature extraction
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import json
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


class MudraDataPreprocessor:
    """Handle data loading, preprocessing, and feature extraction"""
    
    def __init__(self, dataset_path: str, image_size: tuple = (224, 224)):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.scaler = StandardScaler()
        
    def load_dataset(self):
        """Load dataset from kagglehub path"""
        try:
            import kagglehub
            path = kagglehub.dataset_download("onkarpathrikar/bharatanatyam-mudra-dataset-master-bm")
            logger.info(f"Dataset downloaded to: {path}")
            return path
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            # If kagglehub fails, use provided path
            return self.dataset_path
    
    def extract_hand_landmarks(self, image_path: str):
        """
        Extract 21 hand landmarks using MediaPipe
        Returns: numpy array of shape (42,) for single hand or (84,) for two hands
        or None if no hands detected
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not read image: {image_path}")
                return None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                landmarks = []
                # Extract up to 2 hands (max 2 hands in Bharatanatyam mudras)
                for hand_landmarks in results.multi_hand_landmarks[:2]:
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Pad to ensure consistent feature vector size (2 hands * 21 landmarks * 3 coordinates)
                while len(landmarks) < 126:  # 2 * 21 * 3
                    landmarks.append(0.0)
                
                return np.array(landmarks[:126], dtype=np.float32)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error extracting landmarks from {image_path}: {e}")
            return None
    
    def prepare_data(self, dataset_path: str):
        """
        Prepare training data from dataset structure
        Expects: dataset_path/mudra_name/image.jpg
        """
        X = []
        y = []
        label_map = {}
        label_idx = 0
        
        # Traverse dataset directory
        dataset_root = Path(dataset_path)
        if not dataset_root.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return None, None, None
        
        for mudra_folder in sorted(dataset_root.iterdir()):
            if not mudra_folder.is_dir():
                continue
            
            mudra_name = mudra_folder.name
            if mudra_name not in label_map:
                label_map[mudra_name] = label_idx
                label_idx += 1
            
            logger.info(f"Processing mudra: {mudra_name}")
            
            # Process all images in mudra folder
            image_files = list(mudra_folder.glob("*.jpg")) + list(mudra_folder.glob("*.png"))
            
            for img_path in tqdm(image_files, desc=f"Extracting {mudra_name}"):
                landmarks = self.extract_hand_landmarks(str(img_path))
                if landmarks is not None:
                    X.append(landmarks)
                    y.append(label_map[mudra_name])
        
        if not X:
            logger.error("No valid landmarks extracted from dataset")
            return None, None, None
        
        logger.info(f"Extracted {len(X)} samples from {len(label_map)} mudra classes")
        
        return np.array(X), np.array(y), label_map
    
    def normalize_features(self, X_train, X_test):
        """Normalize feature vectors"""
        X_train_norm = self.scaler.fit_transform(X_train)
        X_test_norm = self.scaler.transform(X_test)
        return X_train_norm, X_test_norm, self.scaler


class MudraClassificationModel:
    """Build and train mudra classification model"""
    
    def __init__(self, input_dim: int = 126, num_classes: int = None):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self):
        """Build deep neural network for mudra classification"""
        self.model = Sequential([
            layers.Input(shape=(self.input_dim,)),
            
            # First dense block
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second dense block
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third dense block
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Fourth dense block
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return self.model
    
    def compile_model(self):
        """Compile model with optimizer and loss"""
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info("Model compiled successfully")
    
    def train(self, X_train, y_train, X_val, y_val, epochs: int = 100, batch_size: int = 32):
        """Train the model"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test Loss: {loss:.4f}")
        return loss, accuracy
    
    def save_model(self, model_path: str):
        """Save trained model"""
        self.model.save(model_path)
        logger.info(f"Model saved to: {model_path}")


def main():
    """Main training pipeline"""
    
    # Configuration
    DATASET_PATH = None  # Will use kagglehub if not set
    MODEL_OUTPUT_PATH = "ml-models/mudra/mudra_model.h5"
    SCALER_PATH = "ml-models/mudra/scaler.pkl"
    LABEL_MAP_PATH = "ml-models/mudra/label_map.json"
    EPOCHS = 100
    BATCH_SIZE = 32
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    # Create output directory
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("MUDRA CLASSIFICATION MODEL TRAINING")
    logger.info("=" * 60)
    
    # Step 1: Load and preprocess data
    logger.info("\n[Step 1] Loading and preprocessing data...")
    preprocessor = MudraDataPreprocessor(DATASET_PATH)
    
    # Try to load dataset from kagglehub
    dataset_path = preprocessor.load_dataset()
    
    # Prepare data
    X, y, label_map = preprocessor.prepare_data(dataset_path)
    
    if X is None:
        logger.error("Failed to prepare dataset. Exiting.")
        return
    
    logger.info(f"Total samples: {len(X)}")
    logger.info(f"Feature dimension: {X.shape[1]}")
    logger.info(f"Number of mudra classes: {len(label_map)}")
    logger.info(f"Label mapping: {label_map}")
    
    # Step 2: Split data
    logger.info("\n[Step 2] Splitting data into train/val/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=42, stratify=y_train
    )
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Step 3: Normalize features
    logger.info("\n[Step 3] Normalizing features...")
    X_train_norm, X_test_norm, scaler = preprocessor.normalize_features(X_train, X_test)
    X_val_norm = scaler.transform(X_val)
    
    # Step 4: Build model
    logger.info("\n[Step 4] Building neural network model...")
    model_trainer = MudraClassificationModel(
        input_dim=X_train_norm.shape[1],
        num_classes=len(label_map)
    )
    model_trainer.build_model()
    model_trainer.compile_model()
    
    logger.info(model_trainer.model.summary())
    
    # Step 5: Train model
    logger.info("\n[Step 5] Training model...")
    history = model_trainer.train(
        X_train_norm, y_train,
        X_val_norm, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Step 6: Evaluate model
    logger.info("\n[Step 6] Evaluating model on test set...")
    loss, accuracy = model_trainer.evaluate(X_test_norm, y_test)
    
    # Step 7: Save model and artifacts
    logger.info("\n[Step 7] Saving model and artifacts...")
    model_trainer.save_model(MODEL_OUTPUT_PATH)
    
    # Save scaler
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to: {SCALER_PATH}")
    
    # Save label map
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump(label_map, f, indent=2)
    logger.info(f"Label map saved to: {LABEL_MAP_PATH}")
    
    # Save reverse label map for inference
    reverse_label_map = {v: k for k, v in label_map.items()}
    reverse_map_path = LABEL_MAP_PATH.replace(".json", "_reverse.json")
    with open(reverse_map_path, 'w') as f:
        json.dump(reverse_label_map, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Model saved: {MODEL_OUTPUT_PATH}")
    logger.info(f"Scaler saved: {SCALER_PATH}")
    logger.info(f"Label map saved: {LABEL_MAP_PATH}")
    logger.info(f"Final Test Accuracy: {accuracy:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
