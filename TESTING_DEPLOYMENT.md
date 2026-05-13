# TESTING & DEPLOYMENT GUIDE - BHARATANATYAM MUDRA DETECTION

## PHASE 1: VERIFY TRAINING COMPLETION

### Check if model files were created
```bash
cd backend
ls -la app/models/

# Expected output:
# mudra_model.pkl        (trained RandomForest model)
# label_map.json         (50 mudra class mappings)
```

### Verify label_map.json structure
```bash
type app/models/label_map.json

# Expected output (sample):
# {
#   "0": "Alapadmam(1)",
#   "1": "Anjali(1)",
#   "2": "Aralam(1)",
#   ...
#   "49": "Varaha(1)"
# }
```

---

## PHASE 2: START API SERVER

### Start FastAPI with uvicorn
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### Expected Console Output
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Verify API is Running
```bash
# Open browser or curl
curl http://localhost:8000/

# Expected response:
# {"message":"Bharatanatyam Mudra Detection API"}
```

### Check Health Endpoint
```bash
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy"}
```

---

## PHASE 3: TEST PREDICTION ENDPOINT

### Method 1: Using cURL with an image file
```bash
# Make sure you have a test image
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/mudra/predict-frame

# Expected response format:
# {"mudra":"Anjali","confidence":0.87}
```

### Method 2: Using Python requests
```python
import requests

url = "http://localhost:8000/mudra/predict-frame"
with open("test_image.jpg", "rb") as f:
    response = requests.post(url, files={"file": f})
    print(response.json())

# Output: {"mudra": "Anjali", "confidence": 0.87}
```

### Method 3: Using JavaScript/Fetch (from frontend)
```javascript
const formData = new FormData();
formData.append("file", imageBlob);

fetch("http://localhost:8000/mudra/predict-frame", {
    method: "POST",
    body: formData
})
.then(res => res.json())
.then(data => console.log(data));

// Output: {"mudra": "Anjali", "confidence": 0.87}
```

### Test Response Scenarios

#### Scenario 1: Valid hand detected
```json
{
    "mudra": "Anjali",
    "confidence": 0.87
}
```

#### Scenario 2: No hand detected
```json
{
    "mudra": "No Hand",
    "confidence": 0
}
```

#### Scenario 3: Invalid image
```json
{
    "mudra": "Error",
    "confidence": 0
}
```

---

## PHASE 4: OPEN FRONTEND

### Access the Web Interface
```
File → Open: frontend/webcam.html
or
File path: file:///e:/NrityaAI/frontend/webcam.html
```

### Expected Frontend Display
1. Camera initialization message
2. Webcam stream with mirror view
3. Green guide box in center
4. "Current Prediction" section (shows -- initially)
5. "Frames Processed", "Response Time", "Status" metrics
6. "Reset" button
7. Frame buffer showing last 5 predictions

### Test Workflow

#### Step 1: Camera Access
- Browser should prompt for camera permission
- Allow access
- Status should change to "✅ Camera ready"

#### Step 2: Hand Positioning
- Position hand within the green guide box
- Ensure hand is visible and clear
- System captures frame every 400ms

#### Step 3: Live Detection
- After a few frames, you should see mudra name appear
- Confidence percentage will update
- Frame buffer will show last 5 predictions

#### Step 4: Prediction Locking
- Keep hand in same mudra position
- After 3 consecutive same predictions, "🔒 LOCKED" badge appears
- This indicates stable prediction

#### Step 5: Reset
- Click "🔄 Reset" button to clear predictions
- Frame buffer clears
- Lock is released

---

## PERFORMANCE METRICS TO TRACK

### Expected Response Times
| Stage | Time |
|-------|------|
| Image upload | 5-10ms |
| MediaPipe detection | 50-100ms |
| Feature extraction | 1-5ms |
| Model prediction | 10-20ms |
| JSON response | <1ms |
| **Total** | **70-140ms** |

### Frame Processing
- Capture interval: 400ms (2.5 FPS)
- Smooth visual experience
- No lag or stuttering

### Prediction Quality
- Expected accuracy: 80-95%
- Confidence usually 0.7-0.99 for correct mudras
- Quick stabilization (3-5 frames)

---

## TROUBLESHOOTING

### Issue: "Failed to load model"
```
Symptom: API returns {"mudra": "Error", "confidence": 0}
Cause: mudra_model.pkl or label_map.json not found
Fix:
- Check that training completed
- Verify files in backend/app/models/
- Check file permissions
```

### Issue: "Camera not working"
```
Symptom: Frontend shows "❌ Camera error"
Cause: Camera access denied or no camera available
Fix:
- Check browser camera permissions
- Verify camera hardware works
- Try different browser
- Check browser console for errors
```

### Issue: "Cannot connect to API"
```
Symptom: Frontend shows "❌ Error" after clicking Start
Cause: API not running or CORS issue
Fix:
- Check API is running on port 8000
- Check browser console for CORS errors
- Verify uvicorn startup message shows port 8000
```

### Issue: "Slow predictions"
```
Symptom: Response time > 500ms per frame
Cause: Heavy CPU load or MediaPipe initialization
Fix:
- Close other applications
- Reduce image resolution
- Check CPU usage with Task Manager
- First few frames will be slower (MediaPipe init)
```

### Issue: "Always predicting 'No Hand'"
```
Symptom: Frame buffer shows all "No Hand"
Cause: Hand not detected by MediaPipe
Fix:
- Ensure good lighting
- Position hand clearly in frame
- Bring hand closer to camera
- Check hand orientation
- Verify MediaPipe confidence thresholds
```

---

## VALIDATION CHECKLIST

### Pre-Deployment ✓
- [ ] Training completed successfully
- [ ] mudra_model.pkl exists
- [ ] label_map.json contains 50 classes
- [ ] API starts without errors
- [ ] Health endpoint responds

### API Testing ✓
- [ ] POST /mudra/predict-frame works
- [ ] Accepts image files
- [ ] Returns correct JSON format
- [ ] Handles no-hand case
- [ ] Handles error case

### Frontend Testing ✓
- [ ] Camera access works
- [ ] Frames captured every 400ms
- [ ] API calls successful
- [ ] Predictions display correctly
- [ ] Majority voting works
- [ ] Prediction locking works
- [ ] Reset button works
- [ ] Responsive design works

### Performance ✓
- [ ] Response time < 200ms typical
- [ ] No lag in video stream
- [ ] Smooth prediction updates
- [ ] Frame buffer updates correctly
- [ ] No memory leaks (test for 5+ minutes)

---

## PRODUCTION DEPLOYMENT

### Environment Setup
```bash
# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create logs directory
mkdir -p logs

# Create configuration
cat > config.env << EOF
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
MODEL_PATH=app/models/mudra_model.pkl
LABEL_MAP_PATH=app/models/label_map.json
EOF
```

### Run with Gunicorn (Production)
```bash
cd backend
gunicorn -w 4 -b 0.0.0.0:8000 -k uvicorn.workers.UvicornWorker app.main:app
```

### Docker Deployment (Optional)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Build & Run
```bash
docker build -t mudra-detection .
docker run -p 8000:8000 mudra-detection
```

---

## MONITORING & LOGGING

### Check Application Logs
```bash
# View recent logs (last 50 lines)
tail -50 logs/app.log

# View logs in real-time
tail -f logs/app.log

# Search for errors
grep ERROR logs/app.log
```

### Monitor API Performance
```bash
# Check response times
watch -n 1 'curl -s -o /dev/null -w "%{time_total}\n" http://localhost:8000/health'
```

### System Resource Monitoring
```bash
# Check memory usage
ps aux | grep python

# Check port usage
netstat -an | grep 8000

# Monitor CPU
top | grep python
```

---

## NEXT STEPS AFTER DEPLOYMENT

1. ✓ Train model successfully
2. ✓ Start API server
3. ✓ Test endpoints with curl
4. ✓ Open frontend in browser
5. ✓ Test real-time predictions
6. ✓ Validate accuracy
7. Consider: Fine-tuning model parameters
8. Consider: Adding more training data
9. Consider: Deploying to cloud service
10. Consider: Adding authentication/rate limiting

---

## QUICK START COMMAND SEQUENCE

### Windows (PowerShell)
```powershell
# Terminal 1: Start API
cd e:\NrityaAI\backend
python -m app.services.train_model  # If needed
uvicorn app.main:app --reload --port 8000

# Terminal 2: Open frontend
start "file:///e:/NrityaAI/frontend/webcam.html"
```

### Linux/Mac (Bash)
```bash
# Terminal 1: Start API
cd nrityaai/backend
python -m app.services.train_model  # If needed
uvicorn app.main:app --reload --port 8000

# Terminal 2: Open frontend
open webcam.html
# or
python -m http.server 8001 --directory .
# Then visit: http://localhost:8001/frontend/webcam.html
```
