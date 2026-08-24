"""
Standardized Platform Exception Classes
=========================================
Custom domain exception hierarchy for clean error handling across services and API layers.
"""

from typing import Optional, Dict, Any


class PlatformException(Exception):
    """Base exception for all domain-level platform errors."""
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class ModelNotFoundException(PlatformException):
    """Raised when a requested ML model version or artifact is missing."""
    def __init__(self, model_id: str):
        super().__init__(
            message=f"Model artifact '{model_id}' was not found in registry catalog.",
            status_code=404,
            details={"model_id": model_id}
        )


class FeatureComputationException(PlatformException):
    """Raised when online or offline feature calculation fails validation or bounds."""
    def __init__(self, feature_name: str, reason: str):
        super().__init__(
            message=f"Feature computation failed for '{feature_name}': {reason}",
            status_code=422,
            details={"feature_name": feature_name, "reason": reason}
        )


class DatabaseConnectionException(PlatformException):
    """Raised when database query execution fails."""
    def __init__(self, reason: str):
        super().__init__(
            message=f"Database infrastructure error: {reason}",
            status_code=503,
            details={"reason": reason}
        )


class InvalidRequestException(PlatformException):
    """Raised when request payload violates domain constraints."""
    def __init__(self, message: str):
        super().__init__(message=message, status_code=400)


class ExperimentNotFoundException(PlatformException):
    """Raised when an A/B experiment or MLflow run is missing."""
    def __init__(self, experiment_id: str):
        super().__init__(
            message=f"Experiment '{experiment_id}' was not found.",
            status_code=404,
            details={"experiment_id": experiment_id}
        )


class DriftThresholdExceededException(PlatformException):
    """Raised when statistical PSI/KS drift breaches safety threshold."""
    def __init__(self, feature_name: str, psi: float):
        super().__init__(
            message=f"Critical statistical drift detected for feature '{feature_name}' (PSI={psi:.4f}).",
            status_code=409,
            details={"feature_name": feature_name, "psi": psi}
        )
