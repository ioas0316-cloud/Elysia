"""
Sensory Module - P4 Wave Stream Reception System
Receives multi-sensory inputs from videos, music, and other knowledge sources
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

__all__ = [
    'WaveStreamReceiver',
    'StreamSource',
    'YouTubeStreamSource',
    'WikipediaStreamSource',
    'ArxivStreamSource',
    'GitHubStreamSource',
    'StackOverflowStreamSource',
    'FreeMusicArchiveSource',
    'StreamManager'
]
