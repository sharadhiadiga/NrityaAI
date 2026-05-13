# 🎭 NrityaAI — AI-powered Bharatanatyam learning and analysis platform

## 📌 Overview
NrityaAI is an AI-based system that detects and classifies Bharatanatyam mudras and postures in real-time using computer vision, while also enabling deeper understanding of dance through expression, posture, and storytelling.

## 🚀 Features
- Real-time mudra detection and classification (webcam + image upload)  
- Bhava (facial expression) analysis  
- Posture detection  
- Story interpretation from dance sequences  
- AI chatbot for learning assistance  
- Community section for sharing and learning
  
## 🏗️ Tech Stack
Frontend: React.js, Tailwind CSS  
Backend: Flask, FastAPI  
ML: Python, OpenCV, MediaPipe, TensorFlow/Keras, Scikit-learn  
Database: SQLite, JSON  

## ⚙️ How It Works
1. Capture video input  
2. Detect hand, face, and body landmarks  
3. Extract features  
4. Classify mudra, bhava, and posture  
5. Interpret sequence (story)  
6. Display results on UI  

## 🛠️ Setup
```bash
git clone <your-repo-link>
cd NrityaAI
pip install -r requirements.txt
```
Run backend:
```bash
python app.py
```
Run frontend:
```bash
npm install
npm run dev
```
