#!/usr/bin/env python
"""
Quick Start Guide for NrityaAI Mudra Detection Module
Automated setup and validation script
"""

import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_banner(text):
    """Print formatted banner"""
    width = 70
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def check_python_version():
    """Check Python version"""
    logger.info("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        logger.error(f"✗ Python 3.8+ required. Found: {version.major}.{version.minor}")
        return False


def check_pip():
    """Check if pip is available"""
    logger.info("Checking pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      capture_output=True, check=True)
        logger.info("✓ pip available")
        return True
    except Exception as e:
        logger.error(f"✗ pip not found: {e}")
        return False


def install_dependencies():
    """Install required packages"""
    logger.info("Installing dependencies...")
    requirements_file = "ml-models/mudra/requirements.txt"
    
    if not os.path.exists(requirements_file):
        logger.error(f"✗ Requirements file not found: {requirements_file}")
        return False
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file],
                      check=True)
        logger.info("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to install dependencies: {e}")
        logger.info("Try running manually: pip install -r ml-models/mudra/requirements.txt")
        return False


def run_tests():
    """Run integration tests"""
    logger.info("\nRunning integration tests...")
    test_file = "ml-models/mudra/test_integration.py"
    
    if not os.path.exists(test_file):
        logger.error(f"✗ Test file not found: {test_file}")
        return False
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=False, check=True)
        logger.info("✓ Tests passed")
        return True
    except subprocess.CalledProcessError:
        logger.error("✗ Some tests failed")
        return False


def display_next_steps():
    """Display next steps"""
    print_banner("NEXT STEPS")
    
    steps = [
        ("1. TRAIN MODEL", [
            "python ml-models/mudra/train.py",
            "",
            "This will:",
            "  • Download Bharatanatyam Mudra Dataset from Kagglehub",
            "  • Extract hand landmarks using MediaPipe",
            "  • Train deep neural network",
            "  • Save model artifacts (model, scaler, label maps)"
        ]),
        ("2. START FASTAPI SERVER", [
            "python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000",
            "",
            "API available at: http://localhost:8000",
            "Docs available at: http://localhost:8000/docs"
        ]),
        ("3. TEST WITH API", [
            "curl http://localhost:8000/api/mudra/health",
            "",
            "Upload image for prediction:",
            "curl -X POST -F 'file=@mudra_image.jpg' \\",
            "  http://localhost:8000/api/mudra/predict-mudra"
        ]),
        ("4. RUN REAL-TIME DETECTION", [
            "python ml-models/mudra/webcam_detection.py",
            "",
            "Controls: q=quit, s=screenshot, r=reset smoothing"
        ]),
    ]
    
    for title, commands in steps:
        print(f"\n{title}:")
        print("-" * 70)
        for line in commands:
            print(f"  {line}")


def main():
    """Main setup function"""
    print_banner("NRITYAAI MUDRA DETECTION - QUICK START SETUP")
    
    # Checks
    checks = [
        ("Python Version", check_python_version),
        ("Pip", check_pip),
    ]
    
    logger.info("\n" + "=" * 70)
    logger.info("ENVIRONMENT CHECKS")
    logger.info("=" * 70)
    
    all_passed = True
    for name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            logger.error(f"✗ {name}: {e}")
            all_passed = False
    
    if not all_passed:
        logger.error("\n✗ Some environment checks failed. Please fix issues above.")
        return 1
    
    # Installation
    logger.info("\n" + "=" * 70)
    logger.info("INSTALLATION")
    logger.info("=" * 70)
    
    print("\nDo you want to install/upgrade dependencies? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        if not install_dependencies():
            logger.warning("Failed to install dependencies. Please install manually.")
            # Don't return 1 here - user can continue with existing environment
    
    # Tests
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING TESTS")
    logger.info("=" * 70)
    
    if not run_tests():
        logger.error("\n⚠ Some tests failed. This might be OK if model files are missing.")
        logger.info("Model files will be created when you train the model.")
    
    # Summary
    display_next_steps()
    
    print_banner("QUICK START COMPLETE")
    print("\nFor detailed documentation, see: ml-models/mudra/README.md\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
