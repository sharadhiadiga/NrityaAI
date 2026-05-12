"""
Model Loading and Inference Service
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
from typing import Optional, Tuple
from backend.app.services.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handle model loading, inference, and predictions"""
    
    def __init__(self):
        """Initialize model loader"""
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.label_map = None
        self.reverse_label_map = None
    
    def load_model(self, model_path: str) -> bool:
        """
        Load trained Keras model
        
        Args:
            model_path: Path to .h5 model file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            self.model = keras.models.load_model(model_path)
            logger.info(f"Model loaded successfully from: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def load_scaler(self, scaler_path: str) -> bool:
        """
        Load feature scaler
        
        Args:
            scaler_path: Path to pickled scaler file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.feature_extractor.load_scaler(scaler_path)
            return success
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            return False
    
    def load_label_maps(self, label_map_path: str, reverse_label_map_path: str) -> bool:
        """
        Load label mappings
        
        Args:
            label_map_path: Path to label map JSON
            reverse_label_map_path: Path to reverse label map JSON
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.label_map = FeatureExtractor.load_label_map(label_map_path)
            self.reverse_label_map = FeatureExtractor.load_reverse_label_map(reverse_label_map_path)
            
            if self.label_map is None or self.reverse_label_map is None:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error loading label maps: {e}")
            return False
    
    def predict(self, landmarks: np.ndarray, confidence_threshold: float = 0.0) -> Optional[dict]:
        """
        Make prediction on landmarks
        
        Args:
            landmarks: Raw landmark array (126,)
            confidence_threshold: Minimum confidence to return prediction
        
        Returns:
            Dictionary with mudra name, confidence, and prediction details
            or None if prediction fails or confidence is below threshold
        """
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return None
            
            if not self.feature_extractor.validate_landmarks(landmarks):
                logger.error("Invalid landmarks provided")
                return None
            
            # Normalize landmarks
            normalized = self.feature_extractor.normalize_landmarks(landmarks)
            if normalized is None:
                logger.error("Failed to normalize landmarks")
                return None
            
            # Add batch dimension
            input_data = np.expand_dims(normalized, axis=0)
            
            # Get prediction
            predictions = self.model.predict(input_data, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            
            # Check confidence threshold
            if confidence < confidence_threshold:
                logger.warning(f"Confidence {confidence} below threshold {confidence_threshold}")
                return {
                    'mudra': None,
                    'confidence': confidence,
                    'all_predictions': self._get_top_predictions(predictions[0], top_k=3),
                    'status': 'confidence_below_threshold'
                }
            
            # Get mudra name
            mudra_name = self.reverse_label_map.get(predicted_idx, "Unknown")
            
            result = {
                'mudra': mudra_name,
                'confidence': confidence,
                'class_index': int(predicted_idx),
                'all_predictions': self._get_top_predictions(predictions[0], top_k=3),
                'status': 'success'
            }
            
            logger.info(f"Prediction: {mudra_name} (confidence: {confidence:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'mudra': None,
                'confidence': 0.0,
                'status': 'error',
                'error_message': str(e)
            }
    
    def predict_batch(self, landmarks_batch: np.ndarray) -> list:
        """
        Make predictions on batch of landmarks
        
        Args:
            landmarks_batch: Array of shape (batch_size, 126)
        
        Returns:
            List of prediction dictionaries
        """
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return []
            
            # Normalize batch
            normalized_batch = []
            for landmarks in landmarks_batch:
                norm = self.feature_extractor.normalize_landmarks(landmarks)
                if norm is not None:
                    normalized_batch.append(norm)
            
            if not normalized_batch:
                return []
            
            normalized_batch = np.array(normalized_batch)
            
            # Get predictions
            predictions = self.model.predict(normalized_batch, verbose=0)
            
            results = []
            for pred in predictions:
                predicted_idx = np.argmax(pred)
                confidence = float(pred[predicted_idx])
                mudra_name = self.reverse_label_map.get(predicted_idx, "Unknown")
                
                results.append({
                    'mudra': mudra_name,
                    'confidence': confidence,
                    'class_index': int(predicted_idx)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return []
    
    def _get_top_predictions(self, predictions: np.ndarray, top_k: int = 3) -> list:
        """
        Get top K predictions with mudra names and confidences
        
        Args:
            predictions: Prediction array from model
            top_k: Number of top predictions to return
        
        Returns:
            List of top predictions
        """
        try:
            top_indices = np.argsort(predictions)[::-1][:top_k]
            top_predictions = []
            
            for idx in top_indices:
                mudra_name = self.reverse_label_map.get(int(idx), "Unknown")
                confidence = float(predictions[idx])
                top_predictions.append({
                    'mudra': mudra_name,
                    'confidence': confidence
                })
            
            return top_predictions
        except Exception as e:
            logger.error(f"Error getting top predictions: {e}")
            return []
    
    def is_ready(self) -> bool:
        """Check if model and dependencies are loaded"""
        return (
            self.model is not None and
            self.feature_extractor.scaler is not None and
            self.label_map is not None and
            self.reverse_label_map is not None
        )
    
    def get_model_info(self) -> dict:
        """Get information about loaded model"""
        try:
            if self.model is None:
                return {'status': 'Model not loaded'}
            
            return {
                'status': 'ready' if self.is_ready() else 'incomplete',
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape,
                'num_parameters': self.model.count_params(),
                'num_layers': len(self.model.layers),
                'num_classes': len(self.reverse_label_map) if self.reverse_label_map else 0,
                'mudra_classes': list(self.reverse_label_map.values()) if self.reverse_label_map else []
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {'status': 'error', 'error': str(e)}
