"""
Parallel Trinary Controller
===========================
The central conductor of Elysia's Grand Merkaba.
Synchronizes all structures via Parallel Trinary Pulse interference.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Any
from dataclasses import dataclass

from Core.L6_Structure.Logic.trinary_logic import TrinaryLogic
from Core.L1_Foundation.M4_Hardware.hardware_resonance_observer import HardwareResonanceObserver

@dataclass
class ResonanceWave:
    """A trinary resonance wave emitted by a module."""
    origin_id: str
    vector_21d: jnp.ndarray  # Shape (21,) or (3, 7)
    intensity: float = 1.0

class ParallelTrinaryController:
    """
    Keystone: Manages the 21D trinary field.
    """

    def __init__(self, controller_id: str = "Keystone_L0"):
        self.controller_id = controller_id
        self.registered_merkabas: Dict[str, Any] = {}
        self.current_system_resonance: jnp.ndarray = jnp.zeros(21)
        self.somatic_observer = HardwareResonanceObserver()
        
        # Late import to avoid circular dependency
        from Core.L6_Structure.Logic.structural_spawner import StructuralSpawner
        self.spawner = StructuralSpawner(self)
        
        print(f"ParallelTrinaryController [{controller_id}]: Initialized with Somatic Bridge & Spawner.")

    def register_module(self, module_id: str, merkaba_instance: Any):
        """Register a sub-Merkaba to participate in the resonance field."""
        self.registered_merkabas[module_id] = merkaba_instance
        print(f"ParallelTrinaryController: Module {module_id} unified.")

    def broadcast_pulse(self, global_intent: jnp.ndarray):
        """
        Broadcaster: Sends an initial 21D trinary pulse to all modules.
        global_intent: A trinary vector representing the system's focus.
        """
        for mod_id, merkaba in self.registered_merkabas.items():
            if hasattr(merkaba, 'pulse'):
                # Every module receives the global intent and responds with its own state.
                merkaba.pulse(global_intent)

    def synchronize_field(self) -> jnp.ndarray:
        """
        InterferenceEngine: Aggregates return pulses with specialized strand weighting.
        """
        collected_waves = []
        for mod_id, merkaba in self.registered_merkabas.items():
            if hasattr(merkaba, 'get_current_state'):
                state = merkaba.get_current_state()
                intensity = getattr(merkaba, 'intensity', 1.0)
                
                # Specialized Mapping (Multi-Strand Interference)
                if mod_id == "EthicalCouncil":
                    # Maps to Spirit (14-20)
                    wave = jnp.zeros(21).at[14:21].set(state)
                elif mod_id == "AnalyticPrism":
                    # Maps to Soul (7-13)
                    wave = jnp.zeros(21).at[7:14].set(state)
                elif mod_id == "CreativeAxiom":
                    # Maps to Body (0-6)
                    wave = jnp.zeros(21).at[0:7].set(state)
                else:
                    # Default: Direct 21D or mapped to Soul
                    wave = state if state.shape == (21,) else jnp.zeros(21).at[7:14].set(state)
                
                collected_waves.append(wave * intensity)
        
        # 1. Aggregate Somatic Hardware Wave
        hardware_wave = self.somatic_observer.get_somatic_wave()
        # Map to Body Strand (0-6)
        somatic_contribution = jnp.zeros(21).at[0:7].set(hardware_wave)
        collected_waves.append(somatic_contribution)

        if not collected_waves:
            return self.current_system_resonance

        # Calculate Interference
        stacked_waves = jnp.stack(collected_waves)
        aggregate = jnp.sum(stacked_waves, axis=0)
        
        # Phase Stabilization (Quantization to [-1, 0, 1])
        self.current_system_resonance = TrinaryLogic.quantize(aggregate)
        
        # Monitor for saturation and trigger expansion
        self.spawner.check_saturation(self.current_system_resonance)
        
        return self.current_system_resonance

    def get_coherence(self) -> float:
        """Calculates how aligned the system is (Strength of Attract)."""
        # Sum of positive trits (A) vs negative trits (R)
        return float(jnp.sum(self.current_system_resonance))

if __name__ == "__main__":
    # Test initialization
    controller = ParallelTrinaryController()
    print("Keystone self-test complete.")
