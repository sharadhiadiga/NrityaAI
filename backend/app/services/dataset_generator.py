"""
Dataset Generator - Read images and generate mudra dataset CSV
"""

import os
import csv
import logging
from pathlib import Path
import cv2
import numpy as np

from app.services.mediapipe_service import extract_hand_landmarks
from app.services.feature_extractor import extract_features

logger = logging.getLogger(__name__)

DATASET_RAW_PATH = "datasets/raw"
OUTPUT_CSV_PATH = "datasets/mudra_data.csv"


def generate_dataset():
    """Read images from datasets/raw and generate mudra_data.csv.
    
    Expected structure:
    datasets/raw/
        mudra_class_1/
            image1.jpg
            image2.jpg
        mudra_class_2/
            image1.jpg
    
    Returns:
        Number of samples processed, or 0 if error
    """
    try:
        if not os.path.exists(DATASET_RAW_PATH):
            logger.warning(f"Dataset path not found: {DATASET_RAW_PATH}")
            return 0
        
        # Prepare output
        os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
        
        samples_count = 0
        
        # Open CSV for writing
        with open(OUTPUT_CSV_PATH, 'w', newline='') as csvfile:
            writer = None
            
            # Iterate through mudra classes
            for class_name in sorted(os.listdir(DATASET_RAW_PATH)):
                class_path = os.path.join(DATASET_RAW_PATH, class_name)
                
                if not os.path.isdir(class_path):
                    continue
                
                logger.info(f"Processing class: {class_name}")
                
                # Iterate through images in class
                for image_file in os.listdir(class_path):
                    if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    
                    image_path = os.path.join(class_path, image_file)
                    
                    try:
                        # Read image
                        image = cv2.imread(image_path)
                        if image is None:
                            logger.warning(f"Could not read image: {image_path}")
                            continue
                        
                        # Extract landmarks
                        hands_landmarks = extract_hand_landmarks(image)
                        
                        # Skip if no hand detected
                        if not hands_landmarks or len(hands_landmarks) == 0:
                            logger.debug(f"No hand detected in: {image_file}")
                            continue
                        
                        # Use first hand only
                        landmarks = hands_landmarks[0]
                        
                        # Extract features
                        features = extract_features(landmarks)
                        
                        if features is None:
                            logger.debug(f"Could not extract features from: {image_file}")
                            continue
                        
                        # Write to CSV
                        if writer is None:
                            # Create header: f1, f2, ..., f63, label
                            fieldnames = [f'f{i+1}' for i in range(63)] + ['label']
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()
                        
                        # Create row
                        row = {f'f{i+1}': features[i] for i in range(63)}
                        row['label'] = class_name
                        writer.writerow(row)
                        
                        samples_count += 1
                        logger.debug(f"Processed: {image_file}")
                        
                    except Exception as e:
                        logger.error(f"Error processing {image_path}: {e}")
                        continue
        
        logger.info(f"Dataset generation complete. Samples: {samples_count}")
        logger.info(f"CSV saved to: {OUTPUT_CSV_PATH}")
        return samples_count
        
    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        return 0


def load_dataset_from_csv():
    """Load features and labels from mudra_data.csv.
    
    Returns:
        Tuple of (X, y) where X is features array and y is labels array
    """
    try:
        if not os.path.exists(OUTPUT_CSV_PATH):
            logger.error(f"CSV file not found: {OUTPUT_CSV_PATH}")
            return None, None
        
        features_list = []
        labels_list = []
        
        with open(OUTPUT_CSV_PATH, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                features = [float(row[f'f{i+1}']) for i in range(63)]
                label = row['label']
                
                features_list.append(features)
                labels_list.append(label)
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        logger.info(f"Loaded {len(X)} samples from CSV")
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None, None
