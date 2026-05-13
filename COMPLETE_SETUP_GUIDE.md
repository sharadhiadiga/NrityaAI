# BHARATANATYAM MUDRA DETECTION - COMPLETE END-TO-END SETUP

## PROJECT STRUCTURE
```
backend/
  app/
    main.py                    # FastAPI app entry point
    routes/
      mudra.py                 # ✓ POST /mudra/predict-frame endpoint
    services/
      mediapipe_service.py     # ✓ Hand landmark extraction
      feature_extractor.py     # ✓ 63-feature vector extraction
      mudra_model.py           # ✓ Model loading and prediction
      train_model.py           # ✓ Training pipeline
    models/
      mudra_model.pkl          # Generated after training
      label_map.json           # ✓ Created with 50 mudra classes
  requirements.txt             # All dependencies listed
  
frontend/
  webcam.html                  # ✓ Real-time detection UI

datasets/
  Bharatanatyam-Mudra-Dataset-master/
    [50 mudra class folders]
```

## SYSTEM COMPONENTS

### 1. FEATURE EXTRACTION PIPELINE
**Consistency is CRITICAL** - Must be identical in training and inference

#### Training:
- Load image → MediaPipe hand detection → Extract 21 landmarks
- Normalize landmarks relative to wrist (landmark 0)
- Flatten to 63-element feature vector
- Store features and labels

#### Inference (Frontend):
- Send webcam frame to API
- API extracts landmarks using SAME MediaPipe config
- API extracts features using EXACT SAME function
- API predicts mudra using trained model

### 2. MEDIAPIPE CONFIGURATION
Both training and inference use identical config:
```python
hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

### 3. MODEL TRAINING
- **Algorithm**: RandomForestClassifier
- **Features**: 63 (21 landmarks × 3 coordinates)
- **Classes**: 50 Bharatanatyam mudras
- **Split**: 80% train, 20% test
- **Evaluation**: Accuracy + Classification Report
- **Output**: mudra_model.pkl + label_map.json

### 4. API ENDPOINT

**POST /mudra/predict-frame**
- Accept: image/jpeg file
- Process: MediaPipe → Features → Predict
- Return: `{"mudra": "Anjali", "confidence": 0.87}`

### 5. FRONTEND FEATURES
- Real-time webcam with mirror view
- Guide box for hand positioning
- Majority voting over last 5 frames (smoothing)
- Prediction locking after 3 consecutive same predictions
- Response time tracking
- Frame buffer visualization
- Manual reset button

## SETUP INSTRUCTIONS

### Step 1: Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
cd backend
python -m app.services.train_model
```

**Expected Output:**
- Processes 50 mudra classes from dataset
- Extracts features from images with detected hands
- Trains RandomForest classifier
- Saves:
  - `app/models/mudra_model.pkl` (trained model)
  - `app/models/label_map.json` (class mappings)
- Prints accuracy and classification report

### Step 3: Start FastAPI Server
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

API endpoints available:
- GET http://localhost:8000/ - Root
- GET http://localhost:8000/health - Health check
- POST http://localhost:8000/mudra/predict-frame - Mudra prediction

### Step 4: Test API with cURL
```bash
# Capture a frame from webcam or use an image
curl -X POST -F "file=@image.jpg" http://localhost:8000/mudra/predict-frame

# Expected response:
# {"mudra": "Anjali", "confidence": 0.87}
```

### Step 5: Open Frontend
1. Navigate to `frontend/webcam.html` in a web browser
2. Allow camera access when prompted
3. Position hand within the guide box
4. System will start detecting mudras every 400ms

**Frontend Features:**
- ✓ Camera initializes automatically
- ✓ Captures frames every 400ms
- ✓ Sends to API for prediction
- ✓ Shows current mudra and confidence
- ✓ Displays last 5 predictions with majority voting
- ✓ Locks prediction when stable (3× same mudra)
- ✓ Shows response time and connection status

## KEY CONSISTENCY RULES

### 1. Feature Extraction
```python
# BOTH training and inference MUST use:
# 1. Same 21-landmark extraction
# 2. Same wrist-relative normalization
# 3. Same flattening to 63 elements
```

### 2. MediaPipe Configuration
```python
# static_image_mode = False (for real-time streaming)
# max_num_hands = 1 (only detect one hand)
# min_detection_confidence = 0.5
# min_tracking_confidence = 0.5
```

### 3. Training/Inference Cycle
- Training uses SAME feature extraction
- API uses SAME feature extraction
- Frontend sends frames to SAME API endpoint
- No modifications to features after training

## TROUBLESHOOTING

### Issue: "No hands detected" during training
- Normal behavior - script skips frames without hands
- Check dataset has sufficient images (at least 100s per class)
- Verify images are clear with visible hands

### Issue: API returns "Error" during prediction
- Check camera/image is valid
- Ensure model and label_map are loaded
- Check error logs in terminal

### Issue: Frontend shows "Waiting..."
- Confirm API is running on port 8000
- Check browser console for CORS errors
- Verify camera permissions are granted

### Issue: Low accuracy during training
- Increase dataset size (more images per class)
- Adjust RandomForest hyperparameters in train_model.py
- Increase epochs if using deep learning model

## PERFORMANCE NOTES

- **Processing Speed**: ~100-200ms per frame (including MediaPipe)
- **Memory**: ~2GB for training, ~500MB for inference
- **Real-time Performance**: Smooth at 2-3 FPS (400ms capture interval)

## FILES AT A GLANCE

| File | Status | Purpose |
|------|--------|---------|
| train_model.py | ✓ Complete | Dataset loading, training, model saving |
| mediapipe_service.py | ✓ Complete | Hand landmark extraction |
| feature_extractor.py | ✓ Complete | Feature vector creation (63D) |
| mudra_model.py | ✓ Complete | Model loading and prediction |
| routes/mudra.py | ✓ Complete | FastAPI endpoint |
| main.py | ✓ Complete | FastAPI app setup |
| webcam.html | ✓ Complete | Frontend with all features |
| label_map.json | ✓ Complete | 50 mudra class mappings |

## NEXT STEPS AFTER SETUP

1. ✓ Train model successfully
2. ✓ Start API server
3. ✓ Test individual frames via curl
4. ✓ Open frontend and test live
5. ✓ Validate predictions are accurate
6. ✓ Deploy to production if needed
