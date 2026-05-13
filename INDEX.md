# 🙏 Bharatanatyam Mudra Detection - Complete System Index

## 📑 Documentation & Guides

Start here based on your needs:

### 🚀 **Quick Start** (5 minutes)
→ Read: `MUDRA_SYSTEM_SUMMARY.md` 
- What's been built
- Step-by-step execution
- Key features

### 📖 **Detailed Setup Guide**
→ Read: `backend/MUDRA_SETUP_GUIDE.md`
- Complete installation
- Configuration details
- Performance metrics
- Troubleshooting

### ✅ **Verification Checklist**
→ Read: `backend/VERIFICATION_CHECKLIST.md`
- Pre-flight checks
- Testing procedures
- Diagnostic commands
- Common issues & fixes

### 📊 **Original Implementation Notes**
→ Read: `MUDRA_DETECTION_IMPLEMENTATION.md`
- Architecture details
- Dataset information
- Previous implementations

---

## 🎯 Core Components

### Training System
**File:** `backend/app/services/train_model.py`
**Purpose:** Complete training pipeline
**Features:**
- Downloads Kaggle dataset automatically
- Extracts 21 hand landmarks using MediaPipe
- Generates 63-dimensional feature vectors
- Trains RandomForestClassifier (100 trees)
- 80/20 train/test split
- Saves model + label mapping
- Prints accuracy & classification report

**Run:**
```bash
cd backend
python -m app.services.train_model
```

**Output Files:**
- `app/models/mudra_model.pkl` (trained model)
- `app/models/label_map.json` (label encoding)
- `datasets/mudra_data.csv` (training features)

---

### Backend API
**File:** `backend/app/routes/mudra.py`
**Purpose:** FastAPI endpoints for mudra prediction
**Endpoints:**
- `POST /mudra/predict-frame` - Real-time prediction
- `POST /mudra/detect-hand` - Landmark detection
- `POST /mudra/predict-mudra` - Image prediction

**Run:**
```bash
cd backend
uvicorn app.main:app --reload
```

**API Docs:** `http://localhost:8000/docs`

---

### Support Services
**Files:**
- `backend/app/services/mediapipe_service.py` - Hand detection
- `backend/app/services/feature_extractor.py` - Feature generation
- `backend/app/services/mudra_model.py` - Model loading & inference

**Critical:** These use the EXACT same logic for training and inference!

---

### Frontend UI
**File:** `frontend/webcam.html`
**Purpose:** Real-time mudra detection from webcam
**Features:**
- Live video feed with mirror view
- Green guide box for hand positioning
- Majority voting (last 5 frames)
- Prediction locking (same mudra 3x)
- Real-time statistics
- Beautiful, responsive design

**Open:** `frontend/webcam.html` in browser
(Serve from `python -m http.server 5500` if needed)

---

## 🔄 System Architecture

```
┌─────────────────────────────────────────────┐
│  TRAINING PIPELINE                          │
│  (train_model.py)                           │
│  ✅ Kaggle Dataset Download                │
│  ✅ Image Loading                          │
│  ✅ Landmark Extraction (MediaPipe)        │
│  ✅ Feature Generation (63D)               │
│  ✅ RandomForest Training                  │
│  ✅ Model Saving                           │
└───────────┬─────────────────────────────────┘
            │ Produces
            ▼
    mudra_model.pkl
    label_map.json
            │
            ▼
┌─────────────────────────────────────────────┐
│  BACKEND API                                │
│  (mudra.py + support services)              │
│  ✅ Load Model                             │
│  ✅ Receive Image                          │
│  ✅ Extract Landmarks (SAME code)          │
│  ✅ Generate Features (SAME code)          │
│  ✅ Predict Mudra                          │
│  ✅ Return Confidence                      │
└───────────┬─────────────────────────────────┘
            │ Serves
            ▼
┌─────────────────────────────────────────────┐
│  FRONTEND UI                                │
│  (webcam.html)                              │
│  ✅ Webcam Capture                         │
│  ✅ Frame Sending                          │
│  ✅ Majority Voting                        │
│  ✅ Prediction Locking                     │
│  ✅ Real-time Display                      │
└─────────────────────────────────────────────┘
```

---

## 📋 Quick Reference

### Installation
```bash
cd backend
pip install -r requirements.txt
```

### Authentication
```bash
kaggle auth login
```

### Training
```bash
python -m app.services.train_model
```

### Backend
```bash
uvicorn app.main:app --reload
```

### Frontend
```bash
# Option 1: Python server
python -m http.server 5500
# Then: http://localhost:5500/webcam.html

# Option 2: VS Code Live Server
# Right-click webcam.html → Open with Live Server
```

---

## 🎓 Key Concepts

### Feature Extraction
- **Input:** 21 hand landmarks (x, y, z)
- **Processing:** Normalize to wrist, flatten
- **Output:** 63-dimensional vector
- **Critical:** SAME function used for training & inference!

### Model Training
- **Algorithm:** RandomForestClassifier (100 trees)
- **Data Split:** 80% training, 20% testing
- **Features:** 63 normalized coordinates
- **Output:** mudra_model.pkl + label_map.json

### Real-Time Prediction
- **Input:** Webcam frame every 400ms
- **Processing:** Majority voting (last 5 frames)
- **Locking:** Same mudra 3x = locked prediction
- **Output:** Mudra name + confidence

### MediaPipe Configuration
- `static_image_mode=False` - For video/webcam
- `max_num_hands=1` - Single hand (faster)
- `min_detection_confidence=0.5`
- `min_tracking_confidence=0.5`

---

## 📊 Expected Performance

### Training
- **Time:** 5-10 minutes (depends on dataset size)
- **Accuracy:** 85-95% (test accuracy on Kaggle dataset)
- **Model Size:** 50-100MB

### Inference
- **Per-frame latency:** 300-500ms (on CPU)
- **API latency:** 100-200ms
- **Frames processed:** ~2-3 per second

### System
- **Total API calls:** Unlimited (async)
- **Concurrent frames:** Limited by CPU/bandwidth
- **Accuracy:** Improves with majority voting over frames

---

## 🔍 File Locations

```
├── MUDRA_SYSTEM_SUMMARY.md           ← Start here
├── MUDRA_DETECTION_IMPLEMENTATION.md
├── backend/
│   ├── MUDRA_SETUP_GUIDE.md          ← Detailed guide
│   ├── VERIFICATION_CHECKLIST.md     ← Testing guide
│   ├── app/
│   │   ├── main.py
│   │   ├── services/
│   │   │   ├── mediapipe_service.py
│   │   │   ├── feature_extractor.py
│   │   │   ├── mudra_model.py
│   │   │   └── train_model.py        ← RUN THIS
│   │   ├── routes/
│   │   │   └── mudra.py              ← API endpoints
│   │   ├── models/
│   │   │   ├── mudra_model.pkl       (generated)
│   │   │   └── label_map.json        (generated)
│   │   └── __init__.py
│   ├── requirements.txt
│   └── PIPELINE_USAGE.md
├── frontend/
│   └── webcam.html                   ← Open in browser
├── datasets/
│   └── mudra_data.csv                (generated)
└── ml_models/
    └── mudra/
        ├── (previous implementations)
```

---

## ⚡ Common Commands

### Check System Status
```bash
# Test Python & packages
python -c "import fastapi, mediapipe, sklearn; print('✅ All OK')"

# Check Kaggle auth
cat ~/.kaggle/kaggle.json | head -1

# Test API
curl -X GET http://localhost:8000/docs
```

### Debug & Test
```bash
# Test landmark extraction
python -c "
from app.services.mediapipe_service import extract_hand_landmarks
import cv2
img = cv2.imread('test.jpg')
lms = extract_hand_landmarks(img)
print(f'Found {len(lms)} hands')
"

# Test feature extraction
python -c "
from app.services.feature_extractor import extract_features
features = extract_features([{'x': 0, 'y': 0, 'z': 0} for _ in range(21)])
print(f'Features shape: {len(features)}')
"

# Test model loading
python -c "
from app.services.mudra_model import load_model
if load_model():
    print('✅ Model loaded')
else:
    print('❌ Model not found')
"
```

### API Testing
```bash
# Detect landmarks
curl -X POST http://localhost:8000/mudra/detect-hand \
  -F "file=@test_hand.jpg"

# Predict mudra
curl -X POST http://localhost:8000/mudra/predict-frame \
  -F "file=@test_hand.jpg"
```

---

## 🚨 Troubleshooting

### Training Issues
- "Dataset not found" → Run `kaggle auth login`
- "No hands detected" → Improve image quality
- "Out of memory" → Dataset too large

### Backend Issues
- "Port 8000 in use" → Use `--port 8001`
- "Model not found" → Run training first
- "CORS error" → Serve frontend from Python

### Frontend Issues
- "No camera" → Check browser permissions
- "No predictions" → Check backend is running
- "Slow responses" → Check API latency

---

## 📞 Support & Resources

### Documentation
- `MUDRA_SETUP_GUIDE.md` - Complete setup
- `VERIFICATION_CHECKLIST.md` - Testing
- `MUDRA_SYSTEM_SUMMARY.md` - Overview

### Quick Checks
```bash
# Is everything installed?
bash VERIFICATION_CHECKLIST.md

# Is training data downloaded?
ls app/models/

# Is API running?
curl -s http://localhost:8000/docs

# Is frontend working?
python -m http.server 5500
```

---

## ✅ Pre-Launch Checklist

- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Kaggle authenticated: `kaggle auth login`
- [ ] Model trained: `python -m app.services.train_model`
- [ ] API starts: `uvicorn app.main:app --reload`
- [ ] Frontend loads: Open `webcam.html`
- [ ] Predictions work: Show hand gesture to camera
- [ ] Lock works: Hold mudra 3+ seconds

---

## 🎯 Next Steps

1. **Read:** `MUDRA_SYSTEM_SUMMARY.md` (5 min overview)
2. **Setup:** Follow `backend/MUDRA_SETUP_GUIDE.md` (10 min)
3. **Train:** Run `python -m app.services.train_model` (10 min)
4. **Verify:** Use `backend/VERIFICATION_CHECKLIST.md` (5 min)
5. **Run:** Start API and open webcam UI (2 min)
6. **Enjoy:** Detect mudras in real-time! 🙏

---

## 📝 Notes

- **System is fully dynamic** - works with ANY number of mudra classes
- **Feature extraction is critical** - must be identical in training & inference
- **Majority voting improves accuracy** - last 5 frames averaged
- **Lock mechanism stabilizes UI** - reduces flickering
- **No deep learning used** - RandomForest for simplicity & speed

---

**Version:** 1.0  
**Status:** ✅ Production Ready  
**Last Updated:** May 12, 2026

---

## 🙏 Thank You!

Complete Bharatanatyam Mudra Detection System  
Real-time · Accurate · Production-Ready
