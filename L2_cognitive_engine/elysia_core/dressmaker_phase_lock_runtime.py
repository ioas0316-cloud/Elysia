"""
Dressmaker Fabric Wave Spectrum & Scale-Free Rotor Phase-Locking Runtime
========================================================================
Implements the continuous phase-locking mechanism between Dressmaker fabric wave spectra
(weave pattern, seam resistance, tension tensor, fold frequencies) and scale-free character
body rotors across different game worlds (e.g., Action Game, Explorer World, VR Simulation).

Instead of format conversion (Export/Import) or manual skeleton rigging, fabrics and bodies
resonate and lock phase over ELYSIA's scale-free rotor manifold.
"""

from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from core.knowledge_base.knowledge_bus_slot import KnowledgeBusSlot


@dataclass
class FabricWaveSpectrum:
    """
    Represents a Dressmaker fabric as a continuous wave spectrum in phase space.
    """
    fabric_id: str
    name: str
    weave_density: float          # Frequency base (Hz)
    tension_tensor: np.ndarray    # 3x3 anisotropic physical stress tensor
    seam_resistance: float        # Damping coefficient / boundary stiffness
    fold_frequency_spectrum: np.ndarray  # Harmonics of fabric folds (64-point array)
    chromatic_signature: np.ndarray      # Red/Blue/Yellow [Flux, Order, Entropy]

    @classmethod
    def create_silk(cls, fabric_id: str = "FABRIC_SILK_01") -> 'FabricWaveSpectrum':
        """Factory for light, smooth silk fabric."""
        sample_points = 64
        t = np.linspace(0, 4 * np.pi, sample_points)
        folds = np.sin(t) * 0.8 + 0.3 * np.cos(3 * t)
        tension = np.array([
            [1.2, 0.1, 0.0],
            [0.1, 0.8, 0.0],
            [0.0, 0.0, 0.5]
        ])
        return cls(
            fabric_id=fabric_id,
            name="천상의 비단 (Celestial Silk)",
            weave_density=45.0,
            tension_tensor=tension,
            seam_resistance=0.15,
            fold_frequency_spectrum=folds,
            chromatic_signature=np.array([0.2, 0.7, 0.1])  # High Order
        )

    @classmethod
    def create_armored_leather(cls, fabric_id: str = "FABRIC_LEATHER_01") -> 'FabricWaveSpectrum':
        """Factory for stiff, heavy armored leather."""
        sample_points = 64
        t = np.linspace(0, 4 * np.pi, sample_points)
        folds = 0.3 * np.sin(0.5 * t) + 0.1 * np.cos(2 * t)
        tension = np.array([
            [8.5, 0.5, 0.2],
            [0.5, 7.0, 0.3],
            [0.2, 0.3, 5.5]
        ])
        return cls(
            fabric_id=fabric_id,
            name="용비늘 가죽 갑옷 (Dragon Scale Leather)",
            weave_density=120.0,
            tension_tensor=tension,
            seam_resistance=0.85,
            fold_frequency_spectrum=folds,
            chromatic_signature=np.array([0.7, 0.2, 0.1])  # High Flux/Stiffness
        )


@dataclass
class CharacterBodyRotor:
    """
    Represents a game character's body topology as a Clifford / Scale-Free Rotor Layer.
    """
    character_id: str
    world_origin: str            # e.g., 'ActionGame_Valhalla', 'Explorer_Cyberpunk'
    body_mass: float
    rotor_angles: np.ndarray     # [Euler/Rotor angles for body articulation]
    body_surface_curvature: np.ndarray  # Curvature tensor or spectrum (64-point array)
    phase_velocity: float = 1.0


class DressmakerPhaseLockRuntime:
    """
    ELYSIA Scale-Free Phase-Locking Engine.
    Phase-locks FabricWaveSpectrum with CharacterBodyRotor without format conversion or rigging.
    """

    def __init__(self, knowledge_bus: Optional[KnowledgeBusSlot] = None):
        self.knowledge_bus = knowledge_bus or KnowledgeBusSlot()
        self.active_phase_couplings: Dict[str, Dict[str, Any]] = {}

    def phase_lock_fabric_to_body(
        self,
        fabric: FabricWaveSpectrum,
        body: CharacterBodyRotor
    ) -> Dict[str, Any]:
        """
        Executes phase-locking (Phi_fabric <==> Phi_body)
        between fabric wave spectrum and character body rotor.
        """
        # 1. Fetch macro knowledge constraints if available
        knowledge_spectrum = self.knowledge_bus.generate_phase_constraint_spectrum(
            sample_points=len(fabric.fold_frequency_spectrum)
        )

        # 2. Compute Phase Difference Matrix
        fabric_harmonics = fabric.fold_frequency_spectrum
        body_curvature = body.body_surface_curvature

        # Scale alignment via Rotor Phase Coupling
        coupling_field = (fabric_harmonics + knowledge_spectrum) * np.cos(body_curvature)
        phase_mismatch = np.linalg.norm(fabric_harmonics - body_curvature)

        # Phase-Locking Attractor convergence
        # Phase coherence score C in [0.0, 1.0]
        coherence_score = np.exp(-0.2 * phase_mismatch / (1.0 + fabric.seam_resistance))
        coherence_score = float(np.clip(coherence_score, 0.0, 1.0))

        # 3. Dynamic Tension Tensor Adaptation
        # The fabric's stress tensor adapts smoothly to the body rotor curvature
        adapted_tension = fabric.tension_tensor * (1.0 - 0.3 * coherence_score)

        coupling_key = f"{fabric.fabric_id}__LOCK_TO__{body.character_id}"
        coupling_record = {
            "coupling_key": coupling_key,
            "fabric_id": fabric.fabric_id,
            "fabric_name": fabric.name,
            "character_id": body.character_id,
            "world_origin": body.world_origin,
            "phase_coherence_score": coherence_score,
            "phase_mismatch": float(phase_mismatch),
            "adapted_tension_trace": float(np.trace(adapted_tension)),
            "phase_locked_waveform": coupling_field.tolist(),
            "status": "PHASE_LOCKED" if coherence_score > 0.4 else "PHASE_DIVERGENT"
        }

        self.active_phase_couplings[coupling_key] = coupling_record
        return coupling_record

    def simulate_movement_resonance(
        self,
        coupling_key: str,
        motion_impulse: float,
        time_steps: int = 10
    ) -> List[float]:
        """
        Simulates the dynamic propagation of wrinkles and tension during character movement.
        Returns the resonance trajectory across time steps.
        """
        if coupling_key not in self.active_phase_couplings:
            raise KeyError(f"Coupling '{coupling_key}' not found in active phase couplings.")

        record = self.active_phase_couplings[coupling_key]
        coherence = record["phase_coherence_score"]

        trajectory = []
        current_energy = motion_impulse

        for t in range(time_steps):
            # Energy dampening via seam resistance and phase locking
            damping = 0.15 + (1.0 - coherence) * 0.2
            current_energy = current_energy * (1.0 - damping) + 0.05 * np.sin(0.5 * t)
            trajectory.append(float(current_energy))

        return trajectory
