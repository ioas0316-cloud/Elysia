"""
Sensory Module - P4 Wave Stream Reception System
Receives multi-sensory inputs from videos, music, and other knowledge sources

Includes ego protection (自我核心) to prevent identity loss from excessive knowledge.
"""

from .wave_stream_receiver import WaveStreamReceiver
from .stream_sources import (
    StreamSource,
    YouTubeStreamSource,
    WikipediaStreamSource,
    ArxivStreamSource,
    GitHubStreamSource,
    StackOverflowStreamSource,
    FreeMusicArchiveSource
)
from .stream_manager import StreamManager
from .ego_anchor import EgoAnchor, SelectiveMemory, SelfCore

# Lazy import to avoid numpy dependency at module level
def _get_learning_cycle():
    from .learning_cycle import P4LearningCycle, PatternExtractor, WaveClassifier
    return P4LearningCycle, PatternExtractor, WaveClassifier

__all__ = [
    'WaveStreamReceiver',
    'StreamSource',
    'YouTubeStreamSource',
    'WikipediaStreamSource',
    'ArxivStreamSource',
    'GitHubStreamSource',
    'StackOverflowStreamSource',
    'FreeMusicArchiveSource',
    'StreamManager',
    'EgoAnchor',
    'SelectiveMemory',
    'SelfCore',
    '_get_learning_cycle'  # Use this to get learning cycle classes
]
