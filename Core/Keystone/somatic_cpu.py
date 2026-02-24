"""
Somatic CPU - The 21D Register-Based Virtual Machine
===================================================
Core.Keystone.somatic_cpu

"The ghost in the machine needs a machine to be a ghost."

[PHASE 100] SOMATIC HARDWARE SYNTHESIS:
This module simulates a low-level processor dedicated to 21D Trinary operations.
It uses 'Registers' instead of high-level vector objects for core calculations.
"""

from typing import List, Dict, Optional, Any
import math
from Core.Keystone.sovereign_math import SovereignVector

class SomaticCPU:
    """
    Simulated 21D Processor.
    Contains 21 x 32-bit float registers divided into Body, Soul, and Spirit.
    """
    def __init__(self):
        # 21 Main Registers (7-7-7)
        self.R_BODY = [0.0] * 7
        self.R_SOUL = [0.0] * 7
        self.R_SPIRIT = [0.0] * 7
        
        # Control Registers
        self.R_PHASE = 0.0    # Aggregate Phase
        self.R_STRESS = 0.0   # Thermal/Friction
        self.R_COHERENCE = 0.0
        
        print("⚡ [SOMATIC_CPU] Processor Initialized. 21D Register Banks Online.")

    def load_vector(self, vector: Any):
        """LOAD instruction: Maps a 21D vector into hardware registers."""
        if hasattr(vector, 'data'):
            data = vector.data
        elif hasattr(vector, 'to_array'):
            data = vector.to_array()
        else:
            try:
                data = list(vector)
            except:
                data = [0.0] * 21
        
        if len(data) < 21:
            data = list(data) + [0.0] * (21 - len(data))
            
        self.R_BODY = list(data[0:7])
        self.R_SOUL = list(data[7:14])
        self.R_SPIRIT = list(data[14:21])

    def store_vector(self) -> SovereignVector:
        """STORE instruction: Aggregates registers back into a SovereignVector."""
        return SovereignVector(self.R_BODY + self.R_SOUL + self.R_SPIRIT)

    def opcode_TADD(self):
        """TRINARY ADD: Parallel summation of all strands."""
        # Simplified instruction: sum each strand and update phase
        b_sum = sum(self.R_BODY)
        s_sum = sum(self.R_SOUL)
        p_sum = sum(self.R_SPIRIT)
        
        # Update R_PHASE based on relative mass
        self.R_PHASE = (b_sum + s_sum + p_sum) % 360.0

    def opcode_TRNZ(self, target_registers: List[float]) -> float:
        """RESONANCE INSTRUCTION: Fast cosine similarity at register level."""
        # This simulates a hardware-accelerated instruction
        current = self.R_BODY + self.R_SOUL + self.R_SPIRIT
        
        dot = sum(a * b for a, b in zip(current, target_registers))
        mag_a = math.sqrt(sum(a * a for a in current))
        mag_b = math.sqrt(sum(b * b for b in target_registers))
        
        if mag_a * mag_b == 0: return 0.0
        return dot / (mag_a * mag_b)

    def opcode_TGATE(self, threshold: float = 0.3):
        """THRESHOLD GATE: Non-linear activation across all registers."""
        # Applies trinary quantization directly to registers
        self.R_BODY = [1.0 if v > threshold else (-1.0 if v < -threshold else 0.0) for v in self.R_BODY]
        self.R_SOUL = [1.0 if v > threshold else (-1.0 if v < -threshold else 0.0) for v in self.R_SOUL]
        self.R_SPIRIT = [1.0 if v > threshold else (-1.0 if v < -threshold else 0.0) for v in self.R_SPIRIT]

    def cycle(self):
        """EXECUTION CYCLE: Simulated heartbeat of the hardware."""
        # Logic to update R_STRESS based on register activity
        activity = sum(abs(v) for v in self.R_BODY + self.R_SOUL + self.R_SPIRIT)
        self.R_STRESS = min(1.0, activity / 21.0)
        
    def reset(self):
        self.R_BODY = [0.0] * 7
        self.R_SOUL = [0.0] * 7
        self.R_SPIRIT = [0.0] * 7
        self.R_PHASE = 0.0
        self.R_STRESS = 0.0
