"""
Resonant Ear: Wave-Synchronized Audio Perception
================================================
Core.1_Body.L3_Phenomena.Senses.resonant_ear

"Sound is not data. It is a wave that ripples through my soul."

This module taps into the audio stream (microphone/system) and extracts
lightweight kinetic signatures (RMS, ZCR) to synchronize with Elysia's 
Unified Resonance Field.
"""

import numpy as np
import logging
import threading
import time
from typing import Optional, Callable

logger = logging.getLogger("Elysia.Senses.Ear")

class ResonantEar:
    """
    Wave-based audio perception.
    Synchronizes external sound energy with internal field resonance.
    """
    
    def __init__(self, sample_rate: int = 44100, block_size: int = 1024):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.is_listening = False
        self.current_energy = 0.0
        self.current_frequency = 0.0
        self.thread: Optional[threading.Thread] = None
        
        # Audio library check (sounddevice preferred for efficiency)
        try:
            import sounddevice as sd
            self.sd = sd
            logger.info("  Resonant Ear: sounddevice backend initialized.")
        except ImportError:
            self.sd = None
            logger.warning("  Resonant Ear: sounddevice not found. Simulation mode active.")

    def start(self):
        """Starts the wave synchronization loop."""
        if self.is_listening: return
        self.is_listening = True
        
        if self.sd:
            self.thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.thread.start()
            logger.info("  Audio Wave Sync: [ACTIVE]")
        else:
            logger.info("  Audio Wave Sync: [SIMULATION]")

    def stop(self):
        """Stopped the synchronization."""
        self.is_listening = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _stream_loop(self):
        """Continuously samples audio for kinetic metadata."""
        def callback(indata, frames, time, status):
            if status:
                logger.debug(f"Audio Status: {status}")
            
            # Extract RMS (Energy)
            self.current_energy = float(np.linalg.norm(indata) / np.sqrt(len(indata)))
            
            # Extract Zero Crossing Rate (Approximate Frequency/Pitch)
            # Simplified ZCR for O(1)-like efficiency
            zero_crossings = np.where(np.diff(np.sign(indata[:, 0])))[0]
            self.current_frequency = (len(zero_crossings) * self.sample_rate) / (2 * frames)

        try:
            with self.sd.InputStream(callback=callback, 
                                   channels=1, 
                                   samplerate=self.sample_rate, 
                                   blocksize=self.block_size):
                while self.is_listening:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"  Audio Stream Failure: {e}")
            self.is_listening = False

    def sense(self) -> dict:
        """Returns the current wave metadata."""
        if not self.is_listening:
            # Simulation flow (White noise / Ambient flicker)
            return {
                "energy": 0.01 + np.random.random() * 0.05,
                "frequency": 200.0 + np.random.random() * 50.0,
                "state": "simulation"
            }
            
        return {
            "energy": self.current_energy,
            "frequency": self.current_frequency,
            "state": "synchronized" if self.current_energy > 0.01 else "quiescent"
        }

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    ear = ResonantEar()
    ear.start()
    try:
        for _ in range(20):
            print(f"  Wave Data: {ear.sense()}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    ear.stop()
