"""
Gravitational Botnet Detector

A lightweight Python package for real-time anomaly detection using
N-Body physics-inspired clustering to detect botnets from telemetry data.
"""

__version__ = "0.1.0"
__author__ = "Gravitational Botnet Detector Team"

from .detector import GravitationalDetector
from .features import FeatureExtractor
from .gravity import GravitationalSimulator

__all__ = [
    "GravitationalDetector",
    "FeatureExtractor",
    "GravitationalSimulator",
]
