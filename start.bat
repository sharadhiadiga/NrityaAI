@echo off
REM QUICK START SCRIPT - Bharatanatyam Mudra Detection (Windows)

echo.
echo ==========================================
echo Bharatanatyam Mudra Detection System
echo Quick Start Script
echo ==========================================
echo.

REM Check if training model exists
if not exist "backend\app\models\mudra_model.pkl" (
    echo [!] Model not found. Running training...
    cd backend
    python -m app.services.train_model
    cd ..
    echo [OK] Training complete
) else (
    echo [OK] Model found: backend\app\models\mudra_model.pkl
)

REM Check if label map exists
if not exist "backend\app\models\label_map.json" (
    echo [ERROR] Label map not found!
    exit /b 1
) else (
    echo [OK] Label map found: backend\app\models\label_map.json
)

echo.
echo Starting API server...
echo ==========================================
cd backend
uvicorn app.main:app --reload --port 8000
