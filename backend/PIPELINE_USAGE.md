# Mudra Classification Dataset Pipeline

Complete automated pipeline for downloading, processing, and training a mudra classification model using MediaPipe and RandomForest.

## Overview

The `dataset_pipeline.py` script automates the entire workflow:
1. **Download** - Kaggle dataset using kagglehub
2. **Process** - Extract hand landmarks using MediaPipe
3. **Feature Extraction** - Normalize and flatten landmarks into 63-dimensional vectors
4. **Train** - RandomForest model on processed data
5. **Save** - Model and label encoder for inference

## Prerequisites

All dependencies are in `requirements.txt`:
```bash
cd backend
pip install -r requirements.txt
```

Kaggle API setup (one-time):
```bash
kaggle auth login
# This creates ~/.kaggle/kaggle.json with your credentials
```

## Usage

Run the complete pipeline:
```bash
cd backend
python dataset_pipeline.py
```

## Pipeline Steps

### Step 1: Download Dataset
- Downloads from Kaggle: `onkarpathrikar/bharatanatyam-mudra-dataset-master-bm`
- Automatically handled by `download_dataset()`

### Step 2: Process Images
- **Input**: Raw images organized by mudra class in `datasets/raw/`
- **Process**: 
  - Read each image with OpenCV (BGR format)
  - Extract 21 hand landmarks using MediaPipe
  - Skip images with no detected hand
  - Normalize landmarks relative to wrist position
  - Flatten to 63-dimensional feature vector
- **Output**: `datasets/mudra_data.csv`

### Step 3: Training
- Load feature vectors and labels from CSV
- Train RandomForestClassifier (100 trees, max_depth=20)
- Save model to: `app/models/mudra_model.pkl`
- Save label encoder to: `app/models/label_encoder.pkl`

## Output Files

After successful execution:
```
datasets/
└── mudra_data.csv          # Feature matrix and labels

app/models/
├── mudra_model.pkl         # Trained RandomForest model
└── label_encoder.pkl       # Label encoder for decoding predictions
```

## CSV Format

`datasets/mudra_data.csv`:
```
f1,f2,...,f63,label
0.123,0.456,...,0.789,mudra_class_name
...
```

- `f1` to `f63`: Normalized landmark coordinates (21 landmarks × 3 coordinates)
- `label`: Mudra class name (from folder name)

## API Usage

After training, use the REST API:

**Predict mudra from image:**
```bash
curl -X POST "http://localhost:8000/mudra/predict-mudra" \
  -F "file=@hand_image.jpg"
```

**Response:**
```json
{
  "mudra": "mudra_class_name",
  "confidence": 0.95
}
```

**Detect hand landmarks (without prediction):**
```bash
curl -X POST "http://localhost:8000/mudra/detect-hand" \
  -F "file=@hand_image.jpg"
```

## Troubleshooting

### "Dataset path not found"
- Check kagglehub is properly authenticated
- Run `kaggle auth login` to setup credentials

### "No hand detected" / Low processed count
- Ensure images have visible hands
- Check image quality and resolution
- MediaPipe works best with clear, frontal hand poses

### Import errors
- Verify all packages installed: `pip install -r requirements.txt`
- Ensure you're in the virtual environment

### Model not found during prediction
- Run the pipeline first to train the model
- Check that `app/models/mudra_model.pkl` exists

## Performance

Typical execution:
- Download: 1-5 minutes (depends on dataset size)
- Processing: 5-30 minutes (depends on image count)
- Training: 1-5 minutes (for 100-1000 images)

Success metrics:
- Images with detected hands: 70-90% (typical)
- Model training accuracy: 85-95% (depends on dataset)

## Code Structure

```python
# dataset_pipeline.py
download_dataset()           # Download from Kaggle
extract_hand_landmarks()     # MediaPipe landmark extraction
extract_features()           # Normalize & flatten landmarks
process_dataset()            # Main processing loop
train_model_from_csv()       # Train RandomForest
main()                       # Execute full pipeline
```

## Configuration

Edit `dataset_pipeline.py` to customize:

```python
KAGGLE_DATASET = "onkarpathrikar/bharatanatyam-mudra-dataset-master-bm"
OUTPUT_CSV = "datasets/mudra_data.csv"
MODEL_PATH = "app/models/mudra_model.pkl"

# RandomForest settings
RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=20,          # Tree depth limit
    random_state=42,       # Reproducibility
    n_jobs=-1              # Use all CPU cores
)
```

## Logging

The pipeline prints detailed logs:
```
[Dataset Processing Summary]
Total images found:     1234
Successfully processed: 1100
Skipped/Failed:         134
Success rate:           89.1%
```

## Next Steps

1. Run: `python dataset_pipeline.py`
2. Start server: `uvicorn app.main:app --reload`
3. Test predictions: `curl -X POST http://localhost:8000/mudra/predict-mudra -F "file=@image.jpg"`
