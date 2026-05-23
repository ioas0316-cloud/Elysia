"""
Parallel Trinary Controller
===========================
The central conductor of Elysia's Grand Merkaba.
Synchronizes all structures via Parallel Trinary Pulse interference.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import jax
import jax.numpy as jnp

from Core.System.hardware_resonance_observer import HardwareResonanceObserver
from Core.System.trinary_logic import TrinaryLogic


@dataclass
class ResonanceWave:
    """A trinary resonance wave emitted by a module."""
    origin_id: str
    vector_21d: jnp.ndarray  # Shape (21,) or (3, 7)
    intensity: float = 1.0


@dataclass
class CausalEvent:
    """Structured causal trace event for phase-by-phase auditability."""
    event_id: str
    stage: str
    module_id: str
    input_vector_hash: str
    output_vector_hash: str
    intensity: float
    timestamp: float


@dataclass
class HyperPhaseSnapshot:
    """4D+ projection state to prevent collapse into point/line-only abstractions."""
    bands_7x4: jnp.ndarray
    phase_coherence: float
    field_torque: float
    collapse_pressure: float
    timestamp: float


class ParallelTrinaryController:
    """
    Keystone: Manages the 21D trinary field and its 4D+ hyperphase projection.
    """

    def __init__(self, controller_id: str = "Keystone_L0"):
        self.controller_id = controller_id
        self.registered_merkabas: Dict[str, Any] = {}
        self.current_system_resonance: jnp.ndarray = jnp.zeros(21)
        self.current_hyperphase_bands: jnp.ndarray = jnp.zeros((7, 4))
        self.latest_hyperphase: HyperPhaseSnapshot = HyperPhaseSnapshot(
            bands_7x4=self.current_hyperphase_bands,
            phase_coherence=0.0,
            field_torque=0.0,
            collapse_pressure=0.0,
            timestamp=time.time(),
        )
        self.causal_events: List[CausalEvent] = []
        self.somatic_observer = HardwareResonanceObserver()

        # Late import to avoid circular dependency
        from Core.System.structural_spawner import StructuralSpawner

        self.spawner = StructuralSpawner(self)

        print(f"ParallelTrinaryController [{controller_id}]: Initialized with Somatic Bridge & Spawner.")

    @staticmethod
    def _hash_vector(vector: jnp.ndarray) -> str:
        arr = jnp.asarray(vector).reshape(-1)
        serial = ",".join(f"{float(value):.6f}" for value in arr.tolist())
        return hashlib.sha256(serial.encode("utf-8")).hexdigest()[:16]

    def _record_event(
        self,
        stage: str,
        module_id: str,
        input_vector: jnp.ndarray,
        output_vector: jnp.ndarray,
        intensity: float = 1.0,
    ):
        timestamp = time.time()
        event_id = f"{stage}:{module_id}:{int(timestamp * 1_000_000)}"
        self.causal_events.append(
            CausalEvent(
                event_id=event_id,
                stage=stage,
                module_id=module_id,
                input_vector_hash=self._hash_vector(input_vector),
                output_vector_hash=self._hash_vector(output_vector),
                intensity=float(intensity),
                timestamp=timestamp,
            )
        )

    def _project_to_hyperphase(self, aggregate_21d: jnp.ndarray) -> jnp.ndarray:
        """Project 21D manifold to seven 4D bands: [resistance, void, flow, phase-energy]."""
        bands = jnp.reshape(aggregate_21d, (7, 3))
        resistance = jnp.clip(-bands[:, 0], 0.0, None)
        voidness = jnp.abs(bands[:, 1])
        flow = jnp.clip(bands[:, 2], 0.0, None)
        phase_energy = jnp.linalg.norm(bands, axis=1)
        return jnp.stack([resistance, voidness, flow, phase_energy], axis=1)

    def evolve_hyperphase(self, aggregate_21d: jnp.ndarray, dt: float = 0.1) -> HyperPhaseSnapshot:
        """
        Evolve 4D+ hyperphase instead of collapsing all dynamics into scalar field sums.
        """
        projected = self._project_to_hyperphase(aggregate_21d)
        evolved = (1.0 - dt) * self.current_hyperphase_bands + dt * projected
        self.current_hyperphase_bands = evolved

        coherence = float(jnp.mean(evolved[:, 3]))
        torque = float(jnp.sum(jnp.abs(evolved[:, 2] - evolved[:, 0])))
        collapse_pressure = float(jnp.mean(jnp.abs(evolved[:, 1] - evolved[:, 3])))

        snapshot = HyperPhaseSnapshot(
            bands_7x4=evolved,
            phase_coherence=coherence,
            field_torque=torque,
            collapse_pressure=collapse_pressure,
            timestamp=time.time(),
        )
        self.latest_hyperphase = snapshot
        return snapshot

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
            if hasattr(merkaba, "pulse"):
                merkaba.pulse(global_intent)
                self._record_event(
                    stage="Intent Ingest",
                    module_id=mod_id,
                    input_vector=global_intent,
                    output_vector=global_intent,
                    intensity=getattr(merkaba, "intensity", 1.0),
                )

    def synchronize_field(self) -> jnp.ndarray:
        """
        InterferenceEngine: Aggregates return pulses with specialized strand weighting.
        """
        collected_waves = []
        for mod_id, merkaba in self.registered_merkabas.items():
            if hasattr(merkaba, "get_current_state"):
                state = merkaba.get_current_state()
                intensity = getattr(merkaba, "intensity", 1.0)

                if mod_id == "EthicalCouncil":
                    wave = jnp.zeros(21).at[14:21].set(state)
                elif mod_id == "AnalyticPrism":
                    wave = jnp.zeros(21).at[7:14].set(state)
                elif mod_id == "CreativeAxiom":
                    wave = jnp.zeros(21).at[0:7].set(state)
                else:
                    wave = state if state.shape == (21,) else jnp.zeros(21).at[7:14].set(state)

                collected_waves.append(wave * intensity)
                self._record_event(
                    stage="Resonance Interference",
                    module_id=mod_id,
                    input_vector=state,
                    output_vector=wave,
                    intensity=intensity,
                )

        hardware_wave = self.somatic_observer.get_somatic_wave()
        somatic_contribution = jnp.zeros(21).at[0:7].set(hardware_wave)
        collected_waves.append(somatic_contribution)
        self._record_event(
            stage="Resonance Interference",
            module_id="SomaticObserver",
            input_vector=hardware_wave,
            output_vector=somatic_contribution,
            intensity=1.0,
        )

        if not collected_waves:
            return self.current_system_resonance

        stacked_waves = jnp.stack(collected_waves)
        aggregate = jnp.sum(stacked_waves, axis=0)

        hyperphase = self.evolve_hyperphase(aggregate)
        self._record_event(
            stage="Hyperphase Evolution",
            module_id=self.controller_id,
            input_vector=aggregate,
            output_vector=jnp.asarray(hyperphase.bands_7x4).reshape(-1),
            intensity=hyperphase.phase_coherence,
        )

        self.current_system_resonance = TrinaryLogic.quantize(aggregate)
        self.spawner.check_saturation(self.current_system_resonance)
        self._record_event(
            stage="Sovereign Feedback",
            module_id=self.controller_id,
            input_vector=aggregate,
            output_vector=self.current_system_resonance,
            intensity=1.0,
        )

        return self.current_system_resonance

    def get_hyperphase_metrics(self) -> Dict[str, float]:
        return {
            "phase_coherence": self.latest_hyperphase.phase_coherence,
            "field_torque": self.latest_hyperphase.field_torque,
            "collapse_pressure": self.latest_hyperphase.collapse_pressure,
        }

    def export_causal_trace(self, path: str):
        """Export causal events as JSONL for deterministic replay and audit."""
        with open(path, "w", encoding="utf-8") as trace_file:
            for event in self.causal_events:
                trace_file.write(json.dumps(event.__dict__, ensure_ascii=False) + "\n")

    def get_coherence(self) -> float:
        """Calculates how aligned the system is (Strength of Attract)."""
        return float(jnp.sum(self.current_system_resonance))


if __name__ == "__main__":
    controller = ParallelTrinaryController()
    print("Keystone self-test complete.")
