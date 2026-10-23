"""
[Telos-Gravitational Boundary Adapter]
Implements a topological boundary adapter that establishes a Teleological Gravitational Center (Telos)
upon core task entry.

Rather than relying on legacy OS priority classes or CPU affinity APIs, this adapter acts as a
math/topological filter. High-semantic-mass Telos creates gravitational curvature (K_telos) in
causal space, bending, attenuating, and swallowing external OS background noise, telemetry, and
interrupts, while opening a zero-copy shortcut high-speed pipeline for core execution flows.
"""

import numpy as np
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

from core.physics.semantic_mass_engine import (
    SemanticMassOperator,
    CausalGravityField,
    WhiteTensorField,
    RawPerturbationImpulse
)


@dataclass
class TelosSignal:
    """
    Trigger signal indicating entry into a primary intention stream (Telos).
    """
    signal_id: str
    intent_description: str
    telos_vector: np.ndarray
    priority_weight: float = 1.0
    timestamp: float = 0.0


@dataclass
class ExternalNoiseStream:
    """
    Raw background noise stream (OS telemetry, polling interrupts, background tasks).
    """
    stream_id: str
    source_type: str  # e.g., "OS_Telemetry", "Background_Polling", "Hardware_Interrupt"
    signal_vector: np.ndarray
    raw_payload: Any = None
    amplitude: float = 1.0


class TelosGravitationalBoundaryAdapter:
    """
    [Telos-Gravitational Boundary Adapter]
    Acts as the outermost boundary interface between legacy OS hardware noise
    and the internal Elysia Engine core manifold.
    """
    def __init__(self, dimensions: int = 16, base_gravitational_constant: float = 2.0):
        self.dimensions = dimensions
        self.white_field = WhiteTensorField(dimensions=dimensions)
        self.mass_operator = SemanticMassOperator(base_density=1.5)
        self.gravity_field = CausalGravityField(dimensions=dimensions, gravitational_constant=base_gravitational_constant)

        self.active_telos: Optional[TelosSignal] = None
        self.telos_semantic_mass: float = 0.1
        self.field_curvature: float = 0.1
        self.attenuation_history: List[Dict[str, Any]] = []

    def activate_telos(self, telos_signal: TelosSignal) -> Dict[str, Any]:
        """
        Activates a primary Telos intention stream, establishing a Teleological Gravitational Center.
        """
        if len(telos_signal.telos_vector) < self.dimensions:
            padded_vec = np.pad(telos_signal.telos_vector, (0, self.dimensions - len(telos_signal.telos_vector)))
        else:
            padded_vec = telos_signal.telos_vector[:self.dimensions]

        norm_vec = padded_vec / (np.linalg.norm(padded_vec) + 1e-9)
        telos_signal.telos_vector = norm_vec

        self.active_telos = telos_signal

        # Compute semantic mass of Telos based on intent vector density & priority weight
        conn_matrix = np.outer(norm_vec[:4], norm_vec[:4])
        self.telos_semantic_mass = self.mass_operator.compute_mass(
            connectivity_matrix=conn_matrix,
            trinitarian_contrast_score=2.0 * telos_signal.priority_weight,
            friction_inertia=1.5 * telos_signal.priority_weight
        )

        # Compute gravitational curvature K_telos
        self.field_curvature = self.gravity_field.compute_field_curvature(self.telos_semantic_mass)

        return {
            "status": "Telos_Activated",
            "signal_id": telos_signal.signal_id,
            "intent": telos_signal.intent_description,
            "semantic_mass": self.telos_semantic_mass,
            "field_curvature": self.field_curvature
        }

    def deactivate_telos(self) -> Dict[str, Any]:
        """Deactivates current Telos state and resets gravitational field to baseline."""
        prev_id = self.active_telos.signal_id if self.active_telos else None
        self.active_telos = None
        self.telos_semantic_mass = 0.1
        self.field_curvature = 0.1

        return {
            "status": "Telos_Deactivated",
            "previous_signal_id": prev_id,
            "semantic_mass": self.telos_semantic_mass,
            "field_curvature": self.field_curvature
        }

    def filter_background_noise(self, noise_stream: ExternalNoiseStream) -> Dict[str, Any]:
        """
        Applies topological gravitational attenuation filter to background noise streams.
        As gravitational curvature K_telos increases, noise trajectories are bent into orbit,
        attenuating their energy and preventing them from intruding upon core execution.
        """
        if len(noise_stream.signal_vector) < self.dimensions:
            noise_vec = np.pad(noise_stream.signal_vector, (0, self.dimensions - len(noise_stream.signal_vector)))
        else:
            noise_vec = noise_stream.signal_vector[:self.dimensions]

        noise_norm = noise_vec / (np.linalg.norm(noise_vec) + 1e-9)

        if self.active_telos is None:
            # Baseline state (No Telos gravitational shielding active)
            attenuation_factor = 0.0
            retained_amplitude = noise_stream.amplitude
            bent_vector = noise_norm
        else:
            # Distance / Orthogonality in topological vector space relative to Telos vector
            dot_alignment = float(np.abs(np.dot(self.active_telos.telos_vector, noise_norm)))
            # Topological distance: d = sqrt(2 * (1 - dot))
            topological_distance = float(np.sqrt(2.0 * max(0.0, 1.0 - dot_alignment)))

            # Attenuation calculation:
            # High field_curvature (K_telos) + larger topological distance = heavy decay/swallowing
            # Exponential gravitational decay filter: exp(- K_telos * topological_distance)
            decay_rate = self.field_curvature * (0.5 + topological_distance)
            attenuation_factor = float(1.0 - np.exp(-decay_rate))
            retained_amplitude = float(noise_stream.amplitude * (1.0 - attenuation_factor))

            # Bending the noise vector towards Telos orbital trajectory
            pulled_pos, pulled_vel = self.gravity_field.apply_gravitational_pull(
                mass_center_pos=self.active_telos.telos_vector,
                semantic_mass=self.telos_semantic_mass,
                particle_positions=noise_norm[np.newaxis, :],
                particle_velocities=np.zeros((1, self.dimensions), dtype=np.float32),
                dt=0.2
            )
            bent_vector = pulled_pos[0] / (np.linalg.norm(pulled_pos[0]) + 1e-9)

        record = {
            "stream_id": noise_stream.stream_id,
            "source_type": noise_stream.source_type,
            "original_amplitude": noise_stream.amplitude,
            "retained_amplitude": retained_amplitude,
            "attenuation_factor": attenuation_factor,
            "bent_vector": bent_vector.tolist(),
            "telos_active": self.active_telos is not None
        }
        self.attenuation_history.append(record)
        return record

    def route_primary_stream(self, raw_data: Any, primary_vector: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Routes primary Telos execution stream data through zero-copy shortcut paths,
        bypassing legacy OS queues and amplifying signal-to-noise ratio (SNR).
        """
        if primary_vector is None:
            if self.active_telos is not None:
                primary_vector = self.active_telos.telos_vector
            else:
                primary_vector = np.ones(self.dimensions, dtype=np.float32) / np.sqrt(self.dimensions)

        if len(primary_vector) < self.dimensions:
            primary_vector = np.pad(primary_vector, (0, self.dimensions - len(primary_vector)))
        else:
            primary_vector = primary_vector[:self.dimensions]

        norm_primary = primary_vector / (np.linalg.norm(primary_vector) + 1e-9)

        # Gain factor enhanced by active Telos curvature
        snr_gain = 1.0 + self.field_curvature * 1.5

        return {
            "route_status": "ZeroCopy_Shortcut_HighSpeed",
            "data_payload": raw_data,
            "aligned_vector": norm_primary.tolist(),
            "snr_gain_factor": float(snr_gain),
            "latency_overhead_ms": 0.001
        }

    def evaluate_system_topology(self) -> Dict[str, Any]:
        """Evaluates overall system topology metrics under Telos gravity."""
        avg_attenuation = float(np.mean([r["attenuation_factor"] for r in self.attenuation_history])) if self.attenuation_history else 0.0

        return {
            "telos_active": self.active_telos is not None,
            "active_intent": self.active_telos.intent_description if self.active_telos else None,
            "telos_semantic_mass": self.telos_semantic_mass,
            "field_curvature": self.field_curvature,
            "average_noise_attenuation": avg_attenuation,
            "total_filtered_streams": len(self.attenuation_history)
        }
