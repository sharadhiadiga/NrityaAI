"""
Real-time Mudra Detection from Webcam
Uses OpenCV for webcam capture and MediaPipe + trained model for mudra detection
"""

import cv2
import numpy as np
import logging
from typing import Optional
import argparse
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.services.model_loader import ModelLoader
from backend.app.services.mediapipe_service import MediaPipeHandsService
from backend.app.services.feature_extractor import FeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealtimeMudraDetector:
    """Real-time mudra detection from webcam"""
    
    def __init__(
        self,
        model_path: str = "ml-models/mudra/mudra_model.h5",
        scaler_path: str = "ml-models/mudra/scaler.pkl",
        label_map_path: str = "ml-models/mudra/label_map.json",
        reverse_label_map_path: str = "ml-models/mudra/label_map_reverse.json",
        confidence_threshold: float = 0.7,
        smoothing_window: int = 5
    ):
        """
        Initialize real-time detector
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to scaler
            label_map_path: Path to label map
            reverse_label_map_path: Path to reverse label map
            confidence_threshold: Confidence threshold for predictions
            smoothing_window: Window size for temporal smoothing
        """
        self.model_loader = ModelLoader()
        self.mediapipe_service = MediaPipeHandsService(static_image_mode=False)
        self.feature_extractor = FeatureExtractor()
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.landmarks_history = []
        
        # Load models
        logger.info("Loading models...")
        if not self._load_models(model_path, scaler_path, label_map_path, reverse_label_map_path):
            raise RuntimeError("Failed to load models")
        
        logger.info("✓ Models loaded successfully")
    
    def _load_models(self, model_path, scaler_path, label_map_path, reverse_label_map_path) -> bool:
        """Load all required models and artifacts"""
        try:
            if not self.model_loader.load_model(model_path):
                logger.error("Failed to load model")
                return False
            
            if not self.model_loader.load_scaler(scaler_path):
                logger.error("Failed to load scaler")
                return False
            
            if not self.model_loader.load_label_maps(label_map_path, reverse_label_map_path):
                logger.error("Failed to load label maps")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def detect_and_predict(self, frame: np.ndarray) -> dict:
        """
        Detect hands and predict mudra
        
        Args:
            frame: Input frame from webcam
        
        Returns:
            Dictionary with detection and prediction results
        """
        try:
            # Extract landmarks
            landmarks, confidence = self.mediapipe_service.extract_landmarks_with_confidence(frame)
            
            result = {
                'landmarks': landmarks,
                'confidence_raw': confidence,
                'mudra': None,
                'mudra_confidence': 0.0,
                'all_predictions': [],
                'hand_detected': landmarks is not None,
                'smoothed_prediction': None
            }
            
            if landmarks is None:
                return result
            
            # Add to history for smoothing
            self.landmarks_history.append(landmarks)
            if len(self.landmarks_history) > self.smoothing_window:
                self.landmarks_history.pop(0)
            
            # Use smoothed landmarks if available
            if len(self.landmarks_history) >= 3:
                smoothed_landmarks = self.feature_extractor.smooth_landmarks(self.landmarks_history)
            else:
                smoothed_landmarks = landmarks
            
            # Make prediction
            prediction = self.model_loader.predict(
                smoothed_landmarks,
                confidence_threshold=self.confidence_threshold
            )
            
            if prediction:
                result['mudra'] = prediction.get('mudra')
                result['mudra_confidence'] = prediction.get('confidence', 0.0)
                result['all_predictions'] = prediction.get('all_predictions', [])
                result['smoothed_prediction'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error in detection: {e}")
            return {
                'mudra': None,
                'mudra_confidence': 0.0,
                'error': str(e),
                'hand_detected': False
            }
    
    def run(self, camera_id: int = 0, window_name: str = "NrityaAI - Live Mudra Detection"):
        """
        Run live mudra detection
        
        Args:
            camera_id: Camera device ID
            window_name: Window name for display
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return
        
        logger.info(f"Camera opened successfully")
        logger.info(f"Press 'q' to quit, 's' to save frame, 'r' to reset smoothing")
        
        frame_count = 0
        fps_counter = 0
        fps_time = cv2.getTickCount()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                # Detect and predict
                result = self.detect_and_predict(frame)
                
                # Draw UI elements
                frame = self._draw_ui(frame, result, w, h)
                
                # Calculate FPS
                fps_counter += 1
                if cv2.getTickCount() - fps_time > cv2.getTickFrequency():
                    fps = fps_counter / ((cv2.getTickCount() - fps_time) / cv2.getTickFrequency())
                    fps_counter = 0
                    fps_time = cv2.getTickCount()
                
                # Draw FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display
                cv2.imshow(window_name, frame)
                
                # Keyboard handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Exiting...")
                    break
                elif key == ord('s'):
                    filename = f"mudra_screenshot_{frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Screenshot saved: {filename}")
                elif key == ord('r'):
                    self.landmarks_history = []
                    logger.info("Smoothing reset")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera closed")
    
    def _draw_ui(self, frame: np.ndarray, result: dict, w: int, h: int) -> np.ndarray:
        """Draw UI elements on frame"""
        try:
            # Draw detection status
            if result['hand_detected']:
                status_color = (0, 255, 0)
                status_text = "✓ Hand Detected"
            else:
                status_color = (0, 0, 255)
                status_text = "✗ No Hand"
            
            cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Draw mudra prediction
            if result['mudra'] is not None:
                # Draw box for prediction
                cv2.rectangle(frame, (10, 100), (400, 200), (255, 200, 0), 2)
                
                mudra_text = f"Mudra: {result['mudra']}"
                conf_text = f"Confidence: {result['mudra_confidence']:.2%}"
                
                cv2.putText(frame, mudra_text, (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
                cv2.putText(frame, conf_text, (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
                
                # Draw top predictions
                if result['all_predictions']:
                    y_offset = 220
                    cv2.putText(frame, "Top Predictions:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    for i, pred in enumerate(result['all_predictions'][:3]):
                        y_offset += 25
                        pred_text = f"{i+1}. {pred['mudra']}: {pred['confidence']:.2%}"
                        cv2.putText(frame, pred_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Draw hand confidence
            if result['hand_detected'] and result['confidence_raw'] > 0:
                cv2.putText(
                    frame,
                    f"Hand Confidence: {result['confidence_raw']:.2f}",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (100, 200, 100),
                    1
                )
            
            # Draw circle guide
            center = (w // 2, h // 2)
            radius = min(w, h) // 3
            cv2.circle(frame, center, radius, (150, 150, 0), 2)
            cv2.putText(frame, "Align hand in circle", (center[0] - 100, center[1] - radius - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 0), 1)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing UI: {e}")
            return frame


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Real-time Mudra Detection")
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--model', type=str, default='ml-models/mudra/mudra_model.h5', help='Path to model')
    parser.add_argument('--scaler', type=str, default='ml-models/mudra/scaler.pkl', help='Path to scaler')
    parser.add_argument('--labels', type=str, default='ml-models/mudra/label_map.json', help='Path to label map')
    parser.add_argument('--reverse-labels', type=str, default='ml-models/mudra/label_map_reverse.json', help='Path to reverse label map')
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold (0-1)')
    parser.add_argument('--smoothing', type=int, default=5, help='Smoothing window size')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("NRITYAAI - REAL-TIME MUDRA DETECTION")
    logger.info("=" * 60)
    
    try:
        detector = RealtimeMudraDetector(
            model_path=args.model,
            scaler_path=args.scaler,
            label_map_path=args.labels,
            reverse_label_map_path=args.reverse_labels,
            confidence_threshold=args.confidence,
            smoothing_window=args.smoothing
        )
        
        detector.run(camera_id=args.camera)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
