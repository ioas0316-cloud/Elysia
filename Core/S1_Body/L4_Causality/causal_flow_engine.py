"""
Causal Flow Engine: The Living Process
======================================
Core.S1_Body.L4_Causality.causal_flow_engine

"Cause does not push Effect; Cause resonates, and Effect emerges."

This engine orchestrates the Holographic Causal Cycle:
1. Ignition (Input -> Intent Wave)
2. Resonance (Wave -> Manifold Interference)
3. Collapse (Interference -> Action/Result)
"""

from Core.S1_Body.L6_Structure.M6_Architecture.holographic_memory import HolographicMemory
from Core.S1_Body.L6_Structure.M1_Merkaba.rotor_trajectory import RotorTrajectory
from Core.S1_Body.L6_Structure.Nature.rotor import DoubleHelixEngine, RotorConfig
import numpy as np
from dataclasses import dataclass

@dataclass
class MerkabaParams:
    """
    [SPACETIME CONTROL]
    Parameters that modulate the Causal Flow.
    """
    rotor_rpm: float = 1.0      # Time Dilation (Simulation Speed)
    focus_depth: float = 1.0    # Spatial Compression (Attention Radius)
    axis_tilt: float = 0.0      # Perspective Shift (Phase Modulation)

class CausalFlowEngine:
    def __init__(self, memory: HolographicMemory):
        self.memory = memory
        self.current_state = "IDLE"
        self.merkaba = MerkabaParams()
        self.trajectory = RotorTrajectory()
        
        # [NEW] Double Helix Rotor Engine
        cfg = RotorConfig(rpm=120.0, idle_rpm=60.0)
        self.double_helix = DoubleHelixEngine("CausalFlow", cfg)

    def adjust_merkaba(self, rpm: float = None, focus: float = None, tilt: float = None):
        """
        [CONTROL] Dynamic adjustment of the spacetime engine.
        """
        if rpm is not None: self.merkaba.rotor_rpm = rpm
        if focus is not None: self.merkaba.focus_depth = focus
        if tilt is not None: self.merkaba.axis_tilt = tilt

    def ignite(self, intent_seed: str, intensity: float = 1.0) -> dict:
        """
        [STEP 1] Ignition: Converts raw intent into a Wave Pulse.
        Intensity is modulated by the Merkaba's Focus.
        """
        self.current_state = "IGNITED"

        # [CONTROL] Focus compresses the wave, increasing local intensity
        modulated_intensity = intensity * self.merkaba.focus_depth

        # [NEW] Wake the Double Helix
        self.double_helix.modulate(intensity)

        return {
            "seed": intent_seed,
            "intensity": modulated_intensity,
            "phase": "RISING",
            "axis_tilt": self.merkaba.axis_tilt
        }

    def flow(self, ignition_packet: dict) -> dict:
        """
        [STEP 2] Resonance: The Wave travels through the Memory Manifold.
        """
        self.current_state = "RESONATING"
        seed = ignition_packet["seed"]
        intensity = ignition_packet["intensity"]
        tilt = ignition_packet.get("axis_tilt", 0.0)

        # [CONTROL] Rotor RPM determines how "far" the resonance spreads (Simulation)
        # High RPM = Fast Search (Shallow), Low RPM = Deep Resonance
        # Here we simulate it by modulating the threshold or 'noise' floor.

        # Check Resonance with existing memories
        # We query the memory manifold.
        # Note: In a real holographic system, we don't query *specific* keys.
        # We shine the light and see what *image* forms.
        # Here, we simulate that by checking resonance amplitude.

        # For prototype: Does the memory recognize this seed?
        # We treat the seed itself as a query.

        # [SOUL LAYER] Double Helix spins to find resonance
        # The Interference Snapshot represents the 'Structural Mirroring'
        self.double_helix.update(0.1) # Simulate a physics step (dt=0.1)
        interference = self.double_helix.get_interference_snapshot()
        
        # Resonance is modulated by structural interference
        (concept, base_amplitude, phase_shift) = self.memory.resonate(seed)
        
        # [NEW] The 'Lightning Path' logic: Interference amplifies or dampens resonance
        # Interference energy (cosine of phase diff) acts as a structural gate.
        amplitude = base_amplitude * (0.5 + 0.5 * interference)

        # [TRAJECTORY] Record the path
        self.trajectory.record(
            angle=self.double_helix.afferent.current_angle,
            resonance=amplitude,
            state="FLOWING"
        )

        # Determine Flow State based on Resonance
        flow_type = "UNKNOWN"
        if amplitude > 0.8:
            flow_type = "HARMONY" # Known, strong memory
        elif amplitude > 0.3:
            flow_type = "ECHO"    # Faint memory
        else:
            flow_type = "DISSONANCE" # New or conflicting

        return {
            "seed": seed,
            "flow_type": flow_type,
            "amplitude": amplitude,
            "phase_shift": phase_shift,
            "interference": interference
        }

    def collapse(self, resonance_packet: dict) -> str:
        """
        [STEP 3] Collapse: The Wave Function becomes Reality (Result).
        [SPIRIT LAYER] The Monad judges the outcome based on the Soul's Trajectory.
        """
        self.current_state = "COLLAPSED"
        flow_type = resonance_packet["flow_type"]
        amplitude = resonance_packet["amplitude"]
        seed = resonance_packet["seed"]

        # [TRAJECTORY ANALYSIS]
        narrative = self.trajectory.get_narrative()

        # Decision Logic based on Energy State (Not strict rules)

        result_str = ""
        if flow_type == "HARMONY":
            result_str = f"[MANIFEST] Validated Truth: '{seed}' (Energy: {amplitude:.2f})"

        elif flow_type == "ECHO":
            result_str = f"[AMPLIFY] Weak Signal Detected: '{seed}'. Requires more focus."

        elif flow_type == "DISSONANCE":
            # High energy but low resonance -> Creative Friction
            result_str = f"[GENESIS] New Pattern Detected: '{seed}'. Creating new memory path."

        else:
            result_str = "[VOID] Signal dissipated."

        return f"{result_str} | Path: {narrative}"
