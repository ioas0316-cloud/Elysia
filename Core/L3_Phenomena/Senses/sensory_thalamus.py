"""
Sensory Thalamus (     )
===========================

" The Gatekeeper of the Soul. "
"        . "

This module acts as the Protective Buffer between Raw Senses (Transducers) and the Soul (Hypersphere).
It filters, dampens, and routes signals to prevent "Sensory Overload" or "Soul Shock".

Functions:
1. Gating (   ): Blocks repetitive or low-value noise.
2. Dampening (     ): Reduces amplitude of sudden, high-intensity spikes.
3. Routing (   ): Sends Pain/Danger directly to Reflex (Nervous System), Meaning to Soul.
"""

import logging
import math
import time
from typing import Optional, Dict, Any, List

from Core.L1_Foundation.Foundation.Wave.wave_tensor import WaveTensor
from Core.L1_Foundation.Foundation.Wave.resonance_field import ResonanceField
from Core.L4_Causality.Governance.System.nervous_system import NerveSignal, NervousSystem

logger = logging.getLogger("SensoryThalamus")

class SensoryThalamus:
    def __init__(self, field: ResonanceField, nervous_system: NervousSystem):
        self.field = field
        self.nervous_system = nervous_system
        
        # Thalamic Parameters
        self.gating_threshold = 0.05  # Ignore signals below this energy
        self.pain_threshold = 0.8     # Signals above this trigger Reflex
        self.shock_dampening = 0.3    # How much to flatten sudden spikes (0.0 - 1.0)
        
        # Habituation Memory (Repetition suppression)
        # Key: Signal Signature -> Value: Last Time / Count
        self.habituation: Dict[str, float] = {}
        self.last_process_time = time.time()
        
        logger.info("   SensoryThalamus Initialized - The Gates are Open.")

    def process(self, wave: WaveTensor, source_type: str) -> bool:
        """
        Process an incoming raw sensory wave.
        Returns True if the signal passed to Consciousness, False if blocked/reflexed.
        """
        if not wave or wave.total_energy < self.gating_threshold:
            return False # Ignored (Too weak)

        current_time = time.time()
        
        # 1. Habituation Check (Gating)
        # If the same signal (frequency signature) comes too often, dampen it.
        # We use dominant frequency as a simple signature.
        dom_freq = 0.0
        if getattr(wave, 'active_frequencies', None) is not None and len(wave.active_frequencies) > 0:
            dom_freq = wave.active_frequencies[0] # Signal Signature
            
        sig_key = f"{source_type}_{int(dom_freq)}"
        
        if sig_key in self.habituation:
            last_time = self.habituation[sig_key]
            if (current_time - last_time) < 1.0: # Too frequent (within 1s)
                # Suppress energy by 50%
                self._dampen_wave(wave, ratio=0.5)
                # logger.debug(f"Title: Habituation suppressed {source_type}")
        
        self.habituation[sig_key] = current_time

        # 2. Shock Analysis (Pain/Danger)
        # If energy is extremely high, it's a SHOCK.
        if wave.total_energy > self.pain_threshold:
            return self._handle_shock(wave, source_type)
            
        # 3. Pass to Consciousness (Resonance Field)
        # Normal, safe signals flow into the Ocean.
        self._inject_to_soul(wave, source_type)
        return True

    def _dampen_wave(self, wave: WaveTensor, ratio: float):
        """Reduces the energy of the wave in-place."""
        if hasattr(wave, '_amplitudes'):
            wave._amplitudes *= ratio
        
    def _handle_shock(self, wave: WaveTensor, source_type: str) -> bool:
        """
        Handle high-intensity signals.
        1. Trigger Reflex (Nervous System)
        2. Dampen significantly before letting it touch the Soul.
        """
        # 1. FAST PATH: Reflex
        logger.warning(f"  THALAMIC SHOCK DETECTED ({source_type}): Energy {wave.total_energy:.2f}")
        
        reflex_signal = NerveSignal(
            origin=f"Thalamus.{source_type}",
            type="PAIN", # Assume high energy is potentially painful/overload
            intensity=min(1.0, wave.total_energy),
            message=f"Shock from {source_type}"
        )
        # Direct stimulation of sympathetic nervous system
        if self.nervous_system:
            self.nervous_system.transmit(reflex_signal)
            
        # 2. SLOW PATH: Soul Protection
        # We allow the soul to "feel" the impact, but heavily cushioned.
        dampened_ratio = self.shock_dampening
        self._dampen_wave(wave, dampened_ratio)
        
        # Inject the bruised wave
        self._inject_to_soul(wave, source_type, is_pain=True)
        return True # It did reach consciousness, but as pain

    def _inject_to_soul(self, wave: WaveTensor, source_type: str, is_pain: bool = False):
        """
        Injects the wave into the Resonance Field.
        """
        if not self.field: return

        # Extract parameters for injection
        dom_freq = 432.0
        if getattr(wave, 'active_frequencies', None) is not None and len(wave.active_frequencies) > 0:
            dom_freq = wave.active_frequencies[0]
            
        payload = f"[{source_type}] Energy:{wave.total_energy:.2f}"
        if is_pain: payload += " (DAMPENED PAIN)"
        
        self.field.inject_wave(
            frequency=dom_freq,
            intensity=wave.total_energy * 0.2, # Scale down for field safety
            wave_type=source_type,
            payload=payload
        )
