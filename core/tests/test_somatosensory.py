"""
Elysia Somatosensory & Resonance Testing Suite
================================================
Verifies that the audio/video capture ingester works and falls back gracefully,
and that the multi-stream holographic resonance performs correct projection and consensus matching.
"""

import sys
import os
import math
import pytest

# Ensure root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.somatosensory_ingester import SomatosensoryIngester
from core.multi_stream_resonator import MultiStreamResonator
from core.holographic_memory import BitwiseHologramMemory

def test_somatosensory_ingester():
    """Verify that SomatosensoryIngester captures audio and video and falls back gracefully."""
    ingester = SomatosensoryIngester()
    
    # 1. Capture audio
    audio_wave = ingester.capture_audio(duration_sec=0.05, sample_rate=8000)
    assert isinstance(audio_wave, list)
    assert len(audio_wave) > 0
    assert all(isinstance(x, float) for x in audio_wave)
    
    # 2. Capture video
    video_pixels = ingester.capture_video()
    import numpy as np
    assert isinstance(video_pixels, np.ndarray)
    assert video_pixels.ndim == 2
    assert np.all(video_pixels >= 0.0) and np.all(video_pixels <= 1.0)

def test_multi_stream_resonance_consensus():
    """Verify projection and holographic consensus resonance scoring."""
    memory = BitwiseHologramMemory(size_bits=64)
    resonator = MultiStreamResonator(size_bits=64)
    
    # 1. Pre-register concepts with distinct waveforms
    # Concept A: Low-frequency sine wave
    wave_a = [math.sin(i * 0.05) for i in range(100)]
    img_a = [0.1] * 256
    resonator.register_and_superpose_streams(memory, "apple", "Sweet Apple Sensation", wave_a, img_a)
    
    # Concept B: High-frequency cosine wave
    wave_b = [math.cos(i * 0.5) for i in range(100)]
    img_b = [0.9] * 256
    resonator.register_and_superpose_streams(memory, "tree", "Rough Tree Bark Sensation", wave_b, img_b)
    
    # Verify concepts are in memory
    assert "apple_audio" in memory.registered_concepts
    assert "apple_image" in memory.registered_concepts
    assert "tree_audio" in memory.registered_concepts
    assert "tree_image" in memory.registered_concepts
    
    # 2. Test dynamic projection and resonance matching
    # Let's project wave_a and img_a as a probe (simulating an incoming Apple sensation)
    _, a_addr = resonator.project_audio(wave_a)
    _, i_addr = resonator.project_image(img_a)
    
    audio_scores = memory.scan_resonance(a_addr)
    image_scores = memory.scan_resonance(i_addr)
    
    # Compute average resonance
    coherence_apple = (audio_scores["apple_audio"] + image_scores["apple_image"]) / 2.0
    coherence_tree = (audio_scores["tree_audio"] + image_scores["tree_image"]) / 2.0
    
    # Apple probe should strongly resonate with apple concept and weakly with tree
    assert coherence_apple > 0.8
    assert coherence_tree < 0.5
