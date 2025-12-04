"""
Multimodal Integration Module

Complete multimodal integration with vision, audio, and haptic processing.
Phase 8 of the Elysia roadmap.
"""

from .vision_processor import VisionProcessor, ImageAnalysis
from .audio_processor import AudioProcessor, AudioAnalysis
from .multimodal_fusion import MultimodalFusion, FusionResult, FusionStrategy

__all__ = [
    "VisionProcessor",
    "ImageAnalysis",
    "AudioProcessor",
    "AudioAnalysis",
    "MultimodalFusion",
    "FusionResult",
    "FusionStrategy",
]
