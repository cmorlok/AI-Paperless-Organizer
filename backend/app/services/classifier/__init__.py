"""KI-Klassifizierer: Automatic document classification for Paperless-ngx."""

from app.services.classifier.base_provider import BaseClassifierProvider, ClassificationResult

__all__ = [
    "BaseClassifierProvider",
    "ClassificationResult",
]
