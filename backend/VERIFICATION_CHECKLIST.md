# ✅ Mudra Detection System - Verification Checklist

Use this checklist to verify your installation and setup before running the system.

---

## 📋 Pre-Flight Checklist

### 1. Dependencies Installed
```bash
cd backend
pip list | grep -E "fastapi|mediapipe|opencv|scikit-learn|joblib|pandas|kagglehub"
```
**Expected:** All packages should be listed

### 2. Kaggle Authentication
```bash
cat ~/.kaggle/kaggle.json
```
**Expected:** JSON file with `username` and `key`

If not setup:
```bash
kaggle auth login
```

### 3. Project Structure
```bash
# From backend directory
ls -la app/services/
```
**Expected files:**
- ✅ `mediapipe_service.py`
- ✅ `feature_extractor.py`
- ✅ `mudra_model.py`
- ✅ `train_model.py`

```bash
ls -la app/routes/
```
**Expected:**
- ✅ `mudra.py` (with predict-frame endpoint)

```bash
ls -la ../frontend/
```
**Expected:**
- ✅ `webcam.html`

### 4. API Health Check
```bash
# Terminal 1: Start API
uvicorn app.main:app --reload

# Terminal 2: Test endpoint
curl -X GET http://localhost:8000/docs
```
**Expected:** Swagger UI opens (shows all endpoints)

---

## 🎯 Training Phase Checklist

### Before Training
- [ ] Kaggle account setup & authenticated
- [ ] `requirements.txt` installed
- [ ] API can start (`uvicorn app.main:app`)

### Start Training
```bash
cd backend
python -m app.services.train_model
```

### Monitor Training
```
✅ [Step 1/3] Downloading dataset...
✅ [Step 2/3] Processing dataset...
✅ [Step 3/3] Training model...
```

### After Training - Verify Files
```bash
ls -la app/models/
```
**Expected:**
- ✅ `mudra_model.pkl` (50-100MB)
- ✅ `label_map.json` (1-2KB)

Check label map:
```bash
cat app/models/label_map.json
```
**Expected:** JSON like:
```json
{
  "0": "Anjali",
  "1": "Pataka",
  "2": "Mudra3",
  ...
}
```

### Training Accuracy Check
Look for output like:
```
Training accuracy:  0.956
Test accuracy:      0.923
```
**Expected:** Both > 0.80

---

## 🌐 Backend API Verification

### Start API (if not running)
```bash
cd backend
uvicorn app.main:app --reload
```

### Test /mudra/detect-hand (Landmark Detection)
```bash
curl -X POST http://localhost:8000/mudra/detect-hand \
  -F "file=@your_hand_image.jpg"
```
**Expected Response:**
```json
{
  "hands_detected": 1,
  "landmarks": [
    [{"x": 0.5, "y": 0.3, "z": -0.1}, ...],
    ...
  ]
}
```

### Test /mudra/predict-frame (Prediction)
```bash
curl -X POST http://localhost:8000/mudra/predict-frame \
  -F "file=@your_hand_image.jpg"
```
**Expected Response:**
```json
{
  "mudra": "Anjali",
  "confidence": 0.87
}
```

### Test /mudra/predict-mudra (Image Upload)
```bash
curl -X POST http://localhost:8000/mudra/predict-mudra \
  -F "file=@your_hand_image.jpg"
```
**Expected Response:**
```json
{
  "mudra": "Anjali",
  "confidence": 0.87
}
```

---

## 🎨 Frontend Verification

### Setup
1. Open `frontend/webcam.html` in browser
2. Allow camera access when prompted

### Checklist
- [ ] Camera stream displays (flipped/mirrored)
- [ ] Green guide box visible
- [ ] "Detecting..." shows in prediction area
- [ ] Frames Processed counter increments
- [ ] API Response time shown (e.g., "350ms")

### Test with Hand Gesture
1. Show mudra to camera
2. Hold steady
3. Wait 2-3 seconds

**Expected Behavior:**
- Mudra name appears: "Anjali" (or other mudra)
- Confidence shows: e.g., "87.3%"
- Buffer shows last 5 predictions
- After 3 same predictions → "🔒 LOCKED" badge appears

### Issues?
- **No camera detected**: Check browser permissions
- **"No Hand" shows**: Need better lighting, closer hand
- **High latency (>1000ms)**: API might be slow
- **Predictions wrong**: Model needs more training data

---

## 🔧 Consistency Verification

### Critical: Feature Extraction Must Match

**Training (train_model.py):**
```python
def extract_features(landmarks):
    coords = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
    wrist = coords[0]
    normalized = coords - wrist
    features = normalized.flatten().tolist()
    return features
```

**Inference (feature_extractor.py):**
```python
def extract_features(landmarks):
    coords = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
    wrist = coords[0]
    normalized = coords - wrist
    features = normalized.flatten().tolist()
    return features
```

**Verification:**
```bash
cd backend
python -c "
from app.services.feature_extractor import extract_features
import numpy as np

# Test data
test_landmarks = [{'x': i*0.01, 'y': i*0.01, 'z': i*0.001} for i in range(21)]
features = extract_features(test_landmarks)
print(f'Features shape: {len(features)}')
print(f'Expected: 63')
assert len(features) == 63, 'Feature vector wrong size!'
print('✅ Feature extraction OK')
"
```

---

## 📊 Model Diagnostics

### Check Model File
```bash
cd backend
python -c "
import joblib
import os

model_path = 'app/models/mudra_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f'✅ Model loaded successfully')
    print(f'  - Model type: {type(model).__name__}')
    print(f'  - N estimators: {model.n_estimators}')
    print(f'  - Classes: {model.n_classes_}')
else:
    print('❌ Model file not found')
"
```

### Check Label Map
```bash
cd backend
python -c "
import json

label_path = 'app/models/label_map.json'
with open(label_path) as f:
    labels = json.load(f)
    print(f'✅ {len(labels)} mudra classes:')
    for code, name in sorted(labels.items()):
        print(f'  {code}: {name}')
"
```

---

## 🚀 Full System Test

### Step 1: Start Backend
```bash
# Terminal 1
cd backend
uvicorn app.main:app --reload
# Wait for: Uvicorn running on http://127.0.0.1:8000
```

### Step 2: Verify API is Ready
```bash
# Terminal 2
curl -s http://localhost:8000/docs | grep -q "Swagger" && echo "✅ API Ready"
```

### Step 3: Open Frontend
```bash
# Terminal 3
cd frontend
python -m http.server 5500
# Then visit: http://localhost:5500/webcam.html
```

### Step 4: Test Full Pipeline
1. Browser shows camera feed ✅
2. Green guide box visible ✅
3. Show hand gesture ✅
4. Wait for prediction ✅
5. Mudra name appears ✅
6. Confidence shown ✅
7. After 3 same predictions → LOCKED ✅

---

## ⚠️ Common Issues & Fixes

### Issue: "Model not found"
```
Solution:
1. Run training: python -m app.services.train_model
2. Check file exists: ls -la app/models/mudra_model.pkl
3. Verify paths in mudra_model.py are relative (not absolute)
```

### Issue: "No hand detected"
```
Solution:
1. Improve lighting
2. Get hand closer to camera
3. Ensure full hand is visible
4. Try with MediaPipe test: python -c "
   import cv2, mediapipe as mp
   hands = mp.solutions.hands.Hands()
   print('MediaPipe OK')
"
```

### Issue: CORS error in browser console
```
Solution (Pick ONE):
1. Serve frontend on same port:
   cd frontend
   python -m http.server 8000
   
2. Use Live Server in VS Code
3. Add CORS to FastAPI:
   from fastapi.middleware.cors import CORSMiddleware
   app.add_middleware(CORSMiddleware, allow_origins=["*"])
```

### Issue: Predictions are random/incorrect
```
Solution:
1. Check training accuracy > 0.85
2. Verify feature extraction is identical
3. Ensure MediaPipe config same everywhere (static_image_mode=False)
4. Retrain with more samples
```

### Issue: Slow predictions (>2 seconds)
```
Solution:
1. Check API is running on same machine
2. Try reducing frame resolution in webcam.html
3. Increase FRAME_INTERVAL (e.g., 500ms instead of 400ms)
4. Consider GPU acceleration if available
```

---

## ✅ Final Verification

Run this complete test:

```bash
#!/bin/bash

echo "🔍 System Verification"
echo "====================="

# 1. Check Python
python -c "print('✅ Python OK')" || echo "❌ Python missing"

# 2. Check packages
pip show fastapi mediapipe opencv-python scikit-learn > /dev/null && echo "✅ Dependencies OK" || echo "❌ Install: pip install -r requirements.txt"

# 3. Check files
[ -f app/services/train_model.py ] && echo "✅ train_model.py exists" || echo "❌ Missing"
[ -f app/services/feature_extractor.py ] && echo "✅ feature_extractor.py exists" || echo "❌ Missing"
[ -f app/services/mediapipe_service.py ] && echo "✅ mediapipe_service.py exists" || echo "❌ Missing"
[ -f app/services/mudra_model.py ] && echo "✅ mudra_model.py exists" || echo "❌ Missing"
[ -f app/routes/mudra.py ] && echo "✅ mudra.py exists" || echo "❌ Missing"
[ -f ../frontend/webcam.html ] && echo "✅ webcam.html exists" || echo "❌ Missing"

# 4. Check trained model
[ -f app/models/mudra_model.pkl ] && echo "✅ Model trained" || echo "⚠️  Not trained yet (run: python -m app.services.train_model)"

echo ""
echo "Ready to run:"
echo "1. uvicorn app.main:app --reload"
echo "2. python -m http.server 5500  (in frontend/)"
echo "3. Open http://localhost:5500/webcam.html"
```

Save as `verify.sh` and run:
```bash
bash verify.sh
```

---

## 📞 Quick Help

**Training stuck?**
- Check internet: `ping kaggle.com`
- Check disk space: `df -h`

**API won't start?**
- Check port 8000 is free: `lsof -i :8000`
- Try different port: `uvicorn app.main:app --port 8001`

**Webcam not working?**
- Try different browser (Chrome, Firefox)
- Check permissions: Settings → Privacy → Camera

**Predictions wrong?**
- Increase training data
- Check hand pose: Frontal, clear lighting
- Verify MediaPipe is detecting: Use `/mudra/detect-hand` endpoint

---

**Status:** ✅ Ready for Production
**Last Verified:** May 12, 2026
