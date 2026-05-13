from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2
import os

from app.utils.dataset_loader import load_dataset

from app.services.dataset_to_csv import create_csv

from app.services.posture_analysis import (
    extract_landmarks,
    analyze_posture
)

app = FastAPI()

# ---------------- CORS ----------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- HOME ----------------

@app.get("/")
def home():
    return {"message": "NrityaAI running"}

# ---------------- CSV GENERATION ----------------

@app.get("/generate-csv")
def generate_csv():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    dataset_path = os.path.join(
        BASE_DIR,
        "..",
        "..",
        "dataset"
    )

    dataset = load_dataset(dataset_path)

    create_csv(dataset)

    return {
        "message": "CSV generated successfully"
    }

# ---------------- POSTURE ANALYSIS ----------------

@app.post("/analyze-posture")
async def analyze(file: UploadFile = File(...)):

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)

    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image"}

    landmarks = extract_landmarks(image)

    if landmarks is None:
        return {"error": "No pose detected"}

    result = analyze_posture(landmarks)

    return result