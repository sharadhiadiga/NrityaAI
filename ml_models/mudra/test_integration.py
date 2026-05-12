"""
Integration tests for Mudra Detection Module
Verify all components work correctly
"""

import os
import sys
import numpy as np
import tempfile
import json
import logging
from pathlib import Path

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test all required imports"""
    logger.info("Testing imports...")
    try:
        from backend.app.services.mediapipe_service import MediaPipeHandsService
        from backend.app.services.feature_extractor import FeatureExtractor
        from backend.app.services.model_loader import ModelLoader
        logger.info("✓ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_mediapipe_service():
    """Test MediaPipe service"""
    logger.info("\nTesting MediaPipe service...")
    try:
        from backend.app.services.mediapipe_service import MediaPipeHandsService
        import cv2
        
        service = MediaPipeHandsService()
        
        # Create dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test extraction
        landmarks = service.extract_landmarks_from_image(dummy_image)
        logger.info(f"  Landmarks shape: {landmarks.shape if landmarks is not None else 'None (expected for blank image)'}")
        
        service.close()
        logger.info("✓ MediaPipe service OK")
        return True
    except Exception as e:
        logger.error(f"✗ MediaPipe service failed: {e}")
        return False


def test_feature_extractor():
    """Test feature extractor"""
    logger.info("\nTesting feature extractor...")
    try:
        from backend.app.services.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        
        # Create dummy landmarks
        dummy_landmarks = np.random.randn(126).astype(np.float32)
        
        # Test validation
        is_valid = extractor.validate_landmarks(dummy_landmarks)
        logger.info(f"  Validation result: {is_valid}")
        
        # Test smoothing
        landmarks_seq = [dummy_landmarks for _ in range(3)]
        smoothed = extractor.smooth_landmarks(landmarks_seq)
        logger.info(f"  Smoothed shape: {smoothed.shape}")
        
        logger.info("✓ Feature extractor OK")
        return True
    except Exception as e:
        logger.error(f"✗ Feature extractor failed: {e}")
        return False


def test_model_structure():
    """Test model structure creation"""
    logger.info("\nTesting model structure...")
    try:
        from ml_models.mudra.train import MudraClassificationModel
        
        model = MudraClassificationModel(input_dim=126, num_classes=28)
        model.build_model()
        model.compile_model()
        
        logger.info(f"  Model created with {model.model.count_params()} parameters")
        logger.info(f"  Input shape: {model.model.input_shape}")
        logger.info(f"  Output shape: {model.model.output_shape}")
        
        logger.info("✓ Model structure OK")
        return True
    except Exception as e:
        logger.error(f"✗ Model structure failed: {e}")
        return False


def test_model_loader():
    """Test model loader"""
    logger.info("\nTesting model loader...")
    try:
        from backend.app.services.model_loader import ModelLoader
        
        loader = ModelLoader()
        
        # Check initial state
        logger.info(f"  Initial ready state: {loader.is_ready()}")
        
        logger.info("✓ Model loader OK")
        return True
    except Exception as e:
        logger.error(f"✗ Model loader failed: {e}")
        return False


def test_routes_structure():
    """Test API routes structure"""
    logger.info("\nTesting API routes structure...")
    try:
        from backend.app.routes import mudra
        
        # Check functions exist
        functions = [
            'get_model_loader',
            'get_mediapipe_service',
            'initialize_models',
            'health_check',
            'predict_mudra',
            'predict_batch',
            'get_model_info',
            'setup_models'
        ]
        
        for func_name in functions:
            if not hasattr(mudra, func_name):
                raise AttributeError(f"Missing function: {func_name}")
        
        logger.info(f"  Found all {len(functions)} required endpoints")
        logger.info("✓ Routes structure OK")
        return True
    except Exception as e:
        logger.error(f"✗ Routes structure failed: {e}")
        return False


def test_main_app():
    """Test FastAPI main app"""
    logger.info("\nTesting FastAPI main app...")
    try:
        from backend.app.main import app
        
        # Check app properties
        logger.info(f"  App title: {app.title}")
        logger.info(f"  App version: {app.version}")
        
        # Check routes
        routes = [route.path for route in app.routes]
        logger.info(f"  Total routes: {len(routes)}")
        logger.info(f"  Sample routes: {routes[:3]}")
        
        logger.info("✓ FastAPI app OK")
        return True
    except Exception as e:
        logger.error(f"✗ FastAPI app failed: {e}")
        return False


def test_config_files():
    """Check if configuration files exist"""
    logger.info("\nChecking configuration files...")
    try:
        config_files = {
            "requirements.txt": "ml-models/mudra/requirements.txt",
            "README.md": "ml-models/mudra/README.md",
            "train.py": "ml-models/mudra/train.py",
            "webcam_detection.py": "ml-models/mudra/webcam_detection.py",
        }
        
        missing_files = []
        for name, path in config_files.items():
            full_path = os.path.join(os.path.dirname(__file__), "..", "..", path)
            if not os.path.exists(full_path):
                missing_files.append(name)
                logger.warning(f"  ✗ Missing: {name}")
            else:
                logger.info(f"  ✓ Found: {name}")
        
        if missing_files:
            logger.warning(f"Missing {len(missing_files)} files")
            return False
        
        logger.info("✓ All configuration files present")
        return True
    except Exception as e:
        logger.error(f"✗ Config check failed: {e}")
        return False


def test_dependencies():
    """Test if all dependencies are available"""
    logger.info("\nTesting dependencies...")
    
    dependencies = {
        'fastapi': 'FastAPI',
        'tensorflow': 'TensorFlow',
        'mediapipe': 'MediaPipe',
        'cv2': 'OpenCV',
        'sklearn': 'scikit-learn',
        'numpy': 'NumPy',
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            logger.info(f"  ✓ {name}")
        except ImportError:
            logger.warning(f"  ✗ {name} NOT installed")
            missing.append(name)
    
    if missing:
        logger.warning(f"\nMissing packages: {', '.join(missing)}")
        logger.info("Install with: pip install -r ml-models/mudra/requirements.txt")
        return False
    
    logger.info("✓ All dependencies available")
    return True


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("MUDRA DETECTION MODULE - INTEGRATION TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("MediaPipe Service", test_mediapipe_service),
        ("Feature Extractor", test_feature_extractor),
        ("Model Structure", test_model_structure),
        ("Model Loader", test_model_loader),
        ("Routes Structure", test_routes_structure),
        ("FastAPI App", test_main_app),
        ("Configuration Files", test_config_files),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("=" * 60)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("\n✓ All tests passed! System is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Train model: python ml-models/mudra/train.py")
        logger.info("2. Start API: python -m uvicorn backend.app.main:app --reload")
        logger.info("3. Run webcam: python ml-models/mudra/webcam_detection.py")
        return 0
    else:
        logger.error(f"\n✗ {total - passed} test(s) failed. Please fix issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
