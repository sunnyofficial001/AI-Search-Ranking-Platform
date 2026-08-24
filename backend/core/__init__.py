"""
Core Infrastructure Module
===========================
Exports centralized configuration settings and domain exceptions.
"""

from backend.core.config import settings
from backend.core.exceptions import (
    PlatformException,
    ModelNotFoundException,
    FeatureComputationException,
    DatabaseConnectionException,
    InvalidRequestException,
    ExperimentNotFoundException,
    DriftThresholdExceededException,
)

__all__ = [
    "settings",
    "PlatformException",
    "ModelNotFoundException",
    "FeatureComputationException",
    "DatabaseConnectionException",
    "InvalidRequestException",
    "ExperimentNotFoundException",
    "DriftThresholdExceededException",
]
