"""
Custom exceptions for the artisan-engine API.

This module defines all custom exceptions used throughout the application,
providing structured error handling and clear error messages.
"""


class ArtisanEngineError(Exception):
    """Base exception for all artisan-engine related errors."""

    def __init__(self, message: str, error_code: str = "UNKNOWN_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class ModelNotFoundError(ArtisanEngineError):
    """Raised when a model file cannot be found."""

    def __init__(self, message: str):
        super().__init__(message, "MODEL_NOT_FOUND")


class ModelNotLoadedError(ArtisanEngineError):
    """Raised when attempting to use a model that hasn't been loaded."""

    def __init__(self, message: str):
        super().__init__(message, "MODEL_NOT_LOADED")


class GenerationError(ArtisanEngineError):
    """Raised when structured generation fails."""

    def __init__(self, message: str):
        super().__init__(message, "GENERATION_ERROR")


class ValidationError(ArtisanEngineError):
    """Raised when request validation fails."""

    def __init__(self, message: str):
        super().__init__(message, "VALIDATION_ERROR")


class ConfigurationError(ArtisanEngineError):
    """Raised when there are configuration issues."""

    def __init__(self, message: str):
        super().__init__(message, "CONFIGURATION_ERROR")
