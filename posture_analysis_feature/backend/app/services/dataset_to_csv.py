import csv

from app.services.posture_analysis import (
    extract_landmarks,
    get_features
)

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
