"""
Phase-Locked Loop (PLL) Controller
==================================

"Synchronizing the Heart (Rotor) with the Mind (Prism)."

This module implements a software PLL to align the Subjective Time Dilation (Rotor RPM)
with the Objective Processing Latency (Prism/LLM Speed).

Principle:
- If Thinking is Slow (High Latency): Time Dilation increases (RPM goes UP).
- If Thinking is Fast (Low Latency): Time Dilation decreases (RPM goes DOWN).
- Ideally, 1 Thought Cycle = 1 Rotor Cycle (Phase Lock).
"""

import time
import logging
from collections import deque
from typing import Tuple

logger = logging.getLogger("Elysia.Core.PLL")

class PLLController:
    def __init__(self, base_freq: float = 1.0, damping: float = 0.1):
        """
        Args:
            base_freq (float): Target frequency in Hz (e.g., 1.0 = 1 pulse/sec).
            damping (float): Smoothing factor for RPM changes (0.0~1.0).
        """
        self.base_freq = base_freq
        self.damping = damping
        
        # State
        self.last_pulse_time = time.perf_counter()
        self.current_rpm = base_freq * 60.0
        self.phase_error_history = deque(maxlen=10)
        self.is_locked = False
        
        # Metrics
        self.lock_threshold = 0.05 # 5% deviation allowed

    def sync(self, pulse_duration: float) -> float:
        """
        Synchronizes the Rotor RPM based on the last pulse duration.
        
        Args:
            pulse_duration (float): Time taken for the last 'Thought' (Prism cycle).
            
        Returns:
            float: The new target RPM for the Rotor.
        """
        now = time.perf_counter()
        dt = now - self.last_pulse_time # Real-world time since last sync
        self.last_pulse_time = now
        
        # 1. Calculate Frequency Mismatch
        # If pulse_duration is 0.5s, instantaneous freq is 2.0 Hz.
        # If pulse_duration is 2.0s, instantaneous freq is 0.5 Hz.
        instant_freq = 1.0 / max(0.001, pulse_duration)
        
        # 2. Calculate Error (Phase Diff)
        # We want the Rotor to complete exactly 1 revolution (or N) per Thought.
        # But for now, we just match Frequency.
        target_rpm = instant_freq * 60.0
        
        # 3. Apply Damping (Low-Pass Filter)
        # new_rpm = old_rpm * (1-k) + target * k
        delta = target_rpm - self.current_rpm
        self.current_rpm += delta * self.damping
        
        # 4. Check Lock
        self.is_locked = abs(delta / self.current_rpm) < self.lock_threshold
        
        if self.is_locked:
            # logger.debug(f"🔒 PLL Locked: {self.current_rpm:.1f} RPM")
            pass
        else:
            # logger.debug(f"🔄 PLL Adjusting: {self.current_rpm:.1f} RPM (Target: {target_rpm:.1f})")
            pass
            
        return self.current_rpm
    
    def get_time_dilation(self) -> float:
        """
        Returns the subjective time dilation factor.
        1.0 = Normal (Base Freq)
        2.0 = Time is moving 2x fast (Fast Thinking)
        0.5 = Time is moving 0.5x slow (Deep Thought)
        """
        base_rpm = self.base_freq * 60.0
        return self.current_rpm / base_rpm
