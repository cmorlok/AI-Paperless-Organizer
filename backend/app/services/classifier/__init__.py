"""KI-Klassifizierer: Automatic document classification for Paperless-ngx."""

from app.services.classifier.base_provider import BaseClassifierProvider, ClassificationResult
from app.services.classifier.protocol import DocumentClassifierService
from app.services.classifier.state import AutoClassifyState
from app.services.classifier.auto_classify_loop import auto_classify_loop

__all__ = [
    "BaseClassifierProvider",
    "ClassificationResult",
    "DocumentClassifierService",
    "AutoClassifyState",
    "auto_classify_loop",
]
