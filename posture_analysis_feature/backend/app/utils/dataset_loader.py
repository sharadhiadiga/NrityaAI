import csv
import os
import cv2

from app.services.posture_analysis import (
    extract_landmarks,
    get_features
)

def load_dataset(dataset_path):
    """Load dataset from directory structure"""
    dataset = {}
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}")
        return dataset
    
    for pose_folder in os.listdir(dataset_path):
        pose_path = os.path.join(dataset_path, pose_folder)
        
        if not os.path.isdir(pose_path):
            continue
        
        images = []
        for img_file in os.listdir(pose_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(pose_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
        
        if images:
            dataset[pose_folder] = images
            print(f"Loaded {len(images)} images for pose: {pose_folder}")
    
    return dataset

def create_csv(dataset):

    with open("training_data.csv", "w", newline="") as f:

        writer = None

        for pose_name, images in dataset.items():

            print("Processing:", pose_name)

            for img in images:

                landmarks = extract_landmarks(img)

                if landmarks is None:
                    continue

                features = get_features(landmarks)

                features["label"] = pose_name

                if writer is None:

                    writer = csv.DictWriter(
                        f,
                        fieldnames=features.keys()
                    )

                    writer.writeheader()

                writer.writerow(features)