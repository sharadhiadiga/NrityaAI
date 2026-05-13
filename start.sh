#!/bin/bash
# QUICK START SCRIPT - Bharatanatyam Mudra Detection

echo "=========================================="
echo "Bharatanatyam Mudra Detection System"
echo "Quick Start Script"
echo "=========================================="
echo ""

# Check if training model exists
if [ ! -f "backend/app/models/mudra_model.pkl" ]; then
    echo "❌ Model not found. Running training..."
    cd backend
    python -m app.services.train_model
    cd ..
    echo "✓ Training complete"
else
    echo "✓ Model found: backend/app/models/mudra_model.pkl"
fi

# Check if label map exists
if [ ! -f "backend/app/models/label_map.json" ]; then
    echo "❌ Label map not found!"
    exit 1
else
    echo "✓ Label map found: backend/app/models/label_map.json"
fi

echo ""
echo "Starting API server..."
echo "=========================================="
cd backend
uvicorn app.main:app --reload --port 8000
