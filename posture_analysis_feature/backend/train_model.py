import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Loading CSV...")

df = pd.read_csv("training_data.csv")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    random_state=42
)

print("Training model...")

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\\nAccuracy:")
print(accuracy_score(y_test, preds))

print("\\nClassification Report:")
print(classification_report(y_test, preds))

print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

joblib.dump(model, "pose_model.pkl")

print("\\nModel saved as pose_model.pkl")