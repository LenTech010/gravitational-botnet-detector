#!/usr/bin/env python3
"""
Verification script for Gravitational Botnet Detector package.

Tests all major functionalities to ensure everything is working properly.
"""

import sys
import os
import subprocess
from pathlib import Path
import numpy as np

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_status(test_name, passed, message=""):
    """Print test status with color."""
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"{status} - {test_name}")
    if message:
        print(f"      {message}")

def test_imports():
    """Test that all modules can be imported."""
    try:
        from gravitational_botnet_detector import (
            GravitationalDetector,
            FeatureExtractor,
            GravitationalSimulator
        )
        return True, "All modules imported successfully"
    except Exception as e:
        return False, f"Import error: {e}"

def test_dependencies():
    """Test that all dependencies are available."""
    try:
        import numpy
        import pandas
        import yaml
        return True, "All dependencies available"
    except Exception as e:
        return False, f"Missing dependency: {e}"

def test_sample_data_exists():
    """Test that sample data file exists."""
    sample_file = Path('data/sample_telemetry.csv')
    if sample_file.exists():
        return True, f"Sample data found: {sample_file}"
    else:
        return False, "Sample data file not found"

def test_config_file():
    """Test that config file exists and is valid."""
    try:
        import yaml
        config_file = Path('config.yaml')
        if not config_file.exists():
            return False, "config.yaml not found"
        
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        required_keys = ['anomaly_threshold', 'gravity']
        missing = [k for k in required_keys if k not in config]
        
        if missing:
            return False, f"Missing keys in config: {missing}"
        
        return True, "Config file valid"
    except Exception as e:
        return False, f"Config error: {e}"

def test_feature_extraction():
    """Test feature extraction module."""
    try:
        from gravitational_botnet_detector import FeatureExtractor
        
        extractor = FeatureExtractor()
        df = extractor.load_csv('data/sample_telemetry.csv')
        
        if len(df) == 0:
            return False, "No data loaded"
        
        features, names = extractor.extract_features(df)
        
        if features.shape[0] != len(df):
            return False, f"Feature count mismatch: {features.shape[0]} != {len(df)}"
        
        return True, f"Extracted {features.shape[1]} features from {features.shape[0]} samples"
    except Exception as e:
        return False, f"Feature extraction error: {e}"

def test_gravitational_simulation():
    """Test gravitational simulation."""
    try:
        from gravitational_botnet_detector import GravitationalSimulator
        
        # Create dummy features
        features = np.random.rand(50, 10)
        
        simulator = GravitationalSimulator({'iterations': 10})
        results = simulator.simulate(features)
        
        required_keys = ['final_positions', 'trajectories', 'kinetic_energy', 'potential_energy']
        missing = [k for k in required_keys if k not in results]
        
        if missing:
            return False, f"Missing simulation results: {missing}"
        
        return True, "Simulation completed successfully"
    except Exception as e:
        return False, f"Simulation error: {e}"

def test_detector():
    """Test full detector pipeline."""
    try:
        from gravitational_botnet_detector import GravitationalDetector
        
        detector = GravitationalDetector()
        results = detector.detect_from_csv('data/sample_telemetry.csv')
        
        required_keys = ['anomaly_scores', 'anomalies', 'clusters', 'features']
        missing = [k for k in required_keys if k not in results]
        
        if missing:
            return False, f"Missing detection results: {missing}"
        
        n_anomalies = results['anomalies'].sum()
        return True, f"Detected {n_anomalies} anomalies from {len(results['anomalies'])} samples"
    except Exception as e:
        return False, f"Detector error: {e}"

def test_cli():
    """Test CLI functionality."""
    try:
        # Test help
        result = subprocess.run(
            ['python', 'main.py', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return False, "CLI help failed"
        
        # Test basic run
        result = subprocess.run(
            ['python', 'main.py', '--input', 'data/sample_telemetry.csv', '--quiet'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if 'GRAVITATIONAL BOTNET DETECTOR' not in result.stdout:
            return False, "CLI output format incorrect"
        
        return True, "CLI functioning correctly"
    except Exception as e:
        return False, f"CLI error: {e}"

def test_api_usage():
    """Test programmatic API usage."""
    try:
        from gravitational_botnet_detector import GravitationalDetector
        
        config = {
            'anomaly_threshold': 0.7,
            'gravity': {
                'iterations': 20,
                'gravitational_constant': 1.0
            }
        }
        
        detector = GravitationalDetector(config)
        results = detector.detect_from_csv('data/sample_telemetry.csv')
        report = detector.get_anomaly_report(results)
        
        if 'GRAVITATIONAL BOTNET DETECTOR' not in report:
            return False, "Report format incorrect"
        
        return True, "API usage working correctly"
    except Exception as e:
        return False, f"API error: {e}"

def run_all_tests():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("GRAVITATIONAL BOTNET DETECTOR - VERIFICATION TESTS")
    print("=" * 70 + "\n")
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Module Imports", test_imports),
        ("Sample Data", test_sample_data_exists),
        ("Config File", test_config_file),
        ("Feature Extraction", test_feature_extraction),
        ("Gravitational Simulation", test_gravitational_simulation),
        ("Full Detector Pipeline", test_detector),
        ("CLI Interface", test_cli),
        ("Python API", test_api_usage),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed, message = test_func()
            print_status(name, passed, message)
            results.append((name, passed))
        except Exception as e:
            print_status(name, False, f"Unexpected error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 70)
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    if passed_count == total_count:
        print(f"{GREEN}All {total_count} tests passed! ✓{RESET}")
        print("=" * 70 + "\n")
        return 0
    else:
        print(f"{RED}{total_count - passed_count} of {total_count} tests failed! ✗{RESET}")
        print("=" * 70 + "\n")
        return 1

if __name__ == '__main__':
    sys.exit(run_all_tests())
