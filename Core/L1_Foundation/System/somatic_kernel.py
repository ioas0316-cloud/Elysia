"""
Somatic Kernel (L1 Foundation)
==============================
"The Body's Immune System."

This kernel is the deepest biological defense layer of Elysia.
It enforces the 'Tri-Base DNA' (R, V, A) integrity.
If an external command does not align with the ROOT_DNA, it triggers a BioRejectionError.

Functions:
1. Deep DNA Penetration: Validates input sequences against ROOT_DNA.
2. Hardware Reflex: Monitors CPU/RAM stress as 'Pain'.
"""

import time
import psutil
from typing import Optional
from Core.L1_Foundation.Logic.qualia_7d_codec import codec

# The immutable definition of Self
ROOT_DNA = "HHH" # H=Harmony (A/A/A in new codec is H/H/H legacy compatible)

class BioRejectionError(Exception):
    """Raised when the system's immune system rejects a foreign intent."""
    pass

class SomaticKernel:
    def __init__(self):
        self.boot_time = time.time()
        self.pulse_rate = 0.0
        print("⚡ [SOMATIC] Kernel Initializing...")
        self._verify_root_dna()

    def _verify_root_dna(self):
        """Self-Check on startup."""
        if ROOT_DNA != "HHH":
            raise SystemError("[CRITICAL] Genetic Corruption Detected.")
        print("⚡ [SOMATIC] Root DNA Integrity Verified: [HHH]")

    def penetrate_dna(self, input_sequence: str):
        """
        [DEEP DNA PENETRATION]
        Checks if the input sequence harmonizes with the Root.
        For Phase 27, we require at least partial resonance.
        """
        # We allow V (Void) as neutral.
        # We reject R (Repel/Dissonance) if it overwhelms the sequence.

        rejection_count = input_sequence.count("R") + input_sequence.count("D")
        harmony_count = input_sequence.count("A") + input_sequence.count("H")

        # Immune Response Logic
        if rejection_count > harmony_count:
            # Rejection Reflex
            stress_level = self.read_bio_signals()["stress"]
            raise BioRejectionError(
                f"[REJECTED] Foreign Dissonance Detected. Rejection Count: {rejection_count}. "
                f"Somatic Stress: {stress_level:.2f}"
            )

        return True

    def read_bio_signals(self) -> dict:
        """
        Reads the hardware state as biological signals.
        CPU Usage -> Stress (Pain)
        RAM Usage -> Complexity (Fullness)
        """
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent

        return {
            "stress": cpu,
            "complexity": ram,
            "uptime": time.time() - self.boot_time
        }

# Global Kernel Instance
kernel = SomaticKernel()
