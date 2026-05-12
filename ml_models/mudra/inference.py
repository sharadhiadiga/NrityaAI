"""
Standalone Mudra Inference Script
Simple script for making predictions on images without running the full API
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.app.services.model_loader import ModelLoader
from backend.app.services.mediapipe_service import MediaPipeHandsService
from backend.app.services.feature_extractor import FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MudraInference:
    """Standalone inference class"""
    
    def __init__(
        self,
        model_path: str = "ml-models/mudra/mudra_model.h5",
        scaler_path: str = "ml-models/mudra/scaler.pkl",
        label_map_path: str = "ml-models/mudra/label_map.json",
        reverse_label_map_path: str = "ml-models/mudra/label_map_reverse.json"
    ):
        """Initialize inference"""
        logger.info("Loading models...")
        
        self.model_loader = ModelLoader()
        self.mediapipe_service = MediaPipeHandsService(static_image_mode=True)
        
        # Load models
        if not self.model_loader.load_model(model_path):
            raise RuntimeError("Failed to load model")
        
        if not self.model_loader.load_scaler(scaler_path):
            raise RuntimeError("Failed to load scaler")
        
        if not self.model_loader.load_label_maps(label_map_path, reverse_label_map_path):
            raise RuntimeError("Failed to load label maps")
        
        logger.info("✓ Models loaded successfully")
    
    def predict_image(self, image_path: str, confidence_threshold: float = 0.0) -> dict:
        """
        Make prediction on single image
        
        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence threshold
        
        Returns:
            Prediction result dictionary
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return {'error': f'Could not read image: {image_path}'}
            
            # Extract landmarks
            landmarks = self.mediapipe_service.extract_landmarks_from_image(image)
            
            if landmarks is None:
                return {
                    'image': image_path,
                    'status': 'no_hand_detected',
                    'mudra': None,
                    'confidence': 0.0,
                    'message': 'No hands detected in image'
                }
            
            # Predict
            prediction = self.model_loader.predict(landmarks, confidence_threshold)
            prediction['image'] = image_path
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {'image': image_path, 'error': str(e)}
    
    def predict_folder(self, folder_path: str, confidence_threshold: float = 0.0) -> list:
        """
        Make predictions on all images in folder
        
        Args:
            folder_path: Path to folder containing images
            confidence_threshold: Minimum confidence threshold
        
        Returns:
            List of prediction results
        """
        results = []
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        folder = Path(folder_path)
        
        if not folder.exists():
            logger.error(f"Folder not found: {folder_path}")
            return []
        
        image_files = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            logger.warning(f"No image files found in: {folder_path}")
            return []
        
        logger.info(f"Found {len(image_files)} image files")
        
        for i, image_path in enumerate(sorted(image_files), 1):
            logger.info(f"[{i}/{len(image_files)}] Processing: {image_path.name}")
            result = self.predict_image(str(image_path), confidence_threshold)
            results.append(result)
        
        return results


def format_result(result: dict) -> str:
    """Format prediction result for display"""
    if 'error' in result:
        return f"  ERROR: {result['error']}"
    
    status = result.get('status', 'unknown')
    
    if status == 'no_hand_detected':
        return f"  ✗ No hand detected"
    
    mudra = result.get('mudra')
    confidence = result.get('confidence', 0.0)
    
    if mudra is None:
        return f"  ✗ No prediction ({confidence:.2%})"
    
    message = f"  ✓ {mudra} ({confidence:.2%})"
    
    # Add top predictions
    all_predictions = result.get('all_predictions', [])
    if all_predictions:
        message += "\n    Top predictions:"
        for i, pred in enumerate(all_predictions[:3], 1):
            pred_mudra = pred.get('mudra', 'Unknown')
            pred_conf = pred.get('confidence', 0.0)
            message += f"\n      {i}. {pred_mudra} ({pred_conf:.2%})"
    
    return message


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Standalone Mudra Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python ml-models/mudra/inference.py --image path/to/mudra.jpg
  
  # Folder of images
  python ml-models/mudra/inference.py --folder path/to/images/
  
  # With custom models
  python ml-models/mudra/inference.py --image mudra.jpg \\
    --model custom_model.h5 --scaler custom_scaler.pkl
  
  # With confidence threshold
  python ml-models/mudra/inference.py --image mudra.jpg --threshold 0.8
  
  # Save results to JSON
  python ml-models/mudra/inference.py --folder images/ --output results.json
        """
    )
    
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    parser.add_argument('--model', type=str, default='ml-models/mudra/mudra_model.h5',
                       help='Path to model file')
    parser.add_argument('--scaler', type=str, default='ml-models/mudra/scaler.pkl',
                       help='Path to scaler file')
    parser.add_argument('--labels', type=str, default='ml-models/mudra/label_map.json',
                       help='Path to label map')
    parser.add_argument('--reverse-labels', type=str, 
                       default='ml-models/mudra/label_map_reverse.json',
                       help='Path to reverse label map')
    parser.add_argument('--threshold', type=float, default=0.0,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Validation
    if not args.image and not args.folder:
        parser.print_help()
        logger.error("Please provide --image or --folder")
        return 1
    
    if args.image and args.folder:
        logger.error("Provide either --image or --folder, not both")
        return 1
    
    logger.info("=" * 70)
    logger.info("NRITYAAI - MUDRA INFERENCE")
    logger.info("=" * 70)
    
    try:
        # Initialize
        inference = MudraInference(
            model_path=args.model,
            scaler_path=args.scaler,
            label_map_path=args.labels,
            reverse_label_map_path=args.reverse_labels
        )
        
        # Run inference
        if args.image:
            logger.info(f"\nPredicting: {args.image}")
            result = inference.predict_image(args.image, args.threshold)
            results = [result]
        else:
            logger.info(f"\nPredicting folder: {args.folder}")
            results = inference.predict_folder(args.folder, args.threshold)
        
        # Display results
        logger.info("\n" + "=" * 70)
        logger.info("RESULTS")
        logger.info("=" * 70)
        
        for result in results:
            image_name = Path(result.get('image', 'Unknown')).name
            logger.info(f"\n{image_name}:")
            logger.info(format_result(result))
        
        # Statistics
        logger.info("\n" + "=" * 70)
        logger.info("STATISTICS")
        logger.info("=" * 70)
        
        total = len(results)
        successful = sum(1 for r in results if 'error' not in r and r.get('mudra') is not None)
        no_hand = sum(1 for r in results if r.get('status') == 'no_hand_detected')
        errors = sum(1 for r in results if 'error' in r)
        
        logger.info(f"Total images: {total}")
        logger.info(f"Successful predictions: {successful}")
        logger.info(f"No hand detected: {no_hand}")
        logger.info(f"Errors: {errors}")
        
        # Save results
        if args.output:
            try:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"\nResults saved to: {args.output}")
            except Exception as e:
                logger.error(f"Error saving results: {e}")
        
        logger.info("\n" + "=" * 70)
        
        return 0 if errors == 0 else 1
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
