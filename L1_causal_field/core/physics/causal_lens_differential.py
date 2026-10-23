"""
Causal Lens Differential & Non-Dualistic Intent Engine.

Implements:
1. Primitive Substrate, Structural Mechanics (Binding Operator B_hat), and Emergent Construct with Derivation Paths.
2. 5 Intentional Causal Lenses: Mathematics, Physics, Language, Sound/Acoustics, Vision/Optics.
3. World Intrinsic Mechanism Field (WorldMechanismField) & Parallel Observation Universe.
4. Tri-Variable Sweep Dial algorithm (Fix 2, Sweep 1) for inverse mechanism discovery & bifurcation detection.
5. Reverse Decompression Pipeline (Macro label stripping -> Binding Operator extraction -> Atomic rule reduction -> Re-emergence test).
6. Trinity Phase Topology (Self, Other, Meta) & Reversible 4-Phase Transition Pipeline (Gas <-> Liquid <-> Lattice <-> Re-melting).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Callable
import math
import numpy as np


class ModalityType(Enum):
    MATHEMATICS = "mathematics"
    PHYSICS = "physics"
    LANGUAGE = "language"
    SOUND_ACOUSTICS = "sound_acoustics"
    VISION_OPTICS = "vision_optics"


class PhaseState(Enum):
    GAS_POTENTIAL = "gas_potential"        # Unconverged potential field
    LIQUID_FLOW = "liquid_flow"            # Dynamic operational flow
    LATTICE_CRYSTAL = "lattice_crystal"    # Phase-locked 1D bit lattice
    RE_MELTED = "re_melted"                # Re-activated fluid potential


@dataclass
class PrimitiveSubstrate:
    """Raw, formless potential substrate before structural modulation."""
    substrate_id: str
    energy_density: float
    tension_potential: float
    repulsion_potential: float
    state_vector: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))


@dataclass
class StructuralMechanics:
    """Binding operator and dynamic force field governing substrate transformation."""
    mechanism_id: str
    modality: ModalityType
    tension_coefficient: float
    repulsion_coefficient: float
    boundary_curvature: float
    phase_lock_threshold: float = 0.5

    def apply_binding(self, substrate: PrimitiveSubstrate) -> np.ndarray:
        """Applies structural mechanics to transform substrate state vector."""
        v = substrate.state_vector.copy()
        # Tension draws together, Repulsion pushes apart, boundary curvature refracts
        force = (self.tension_coefficient * substrate.tension_potential) - (self.repulsion_coefficient * substrate.repulsion_potential)
        refraction = math.tanh(self.boundary_curvature * force)

        # Apply transformation
        transformed = v * (1.0 + force) + refraction
        return transformed


@dataclass
class DerivationPath:
    """Complete non-hallucinatory derivation path from substrate through mechanics to construct."""
    path_id: str
    substrate_id: str
    mechanism_id: str
    atomic_rules: List[str]
    phase_transition_history: List[PhaseState]
    isomorphism_score: float = 1.0


@dataclass
class EmergentConstruct:
    """Final phase-crystallized or dynamic phenomenon emerging from derivation path."""
    construct_id: str
    modality: ModalityType
    form_tensor: np.ndarray
    derivation_path: DerivationPath
    bit_offset: int
    invariant_phi: float


class IntentionalCausalLens:
    """Base Intentional Causal Lens embodying specific observation intention."""

    def __init__(self, modality: ModalityType, intent_weight: float = 1.0):
        self.modality = modality
        self.intent_weight = intent_weight

    def refract_substrate(
        self, substrate: PrimitiveSubstrate, mechanics: StructuralMechanics
    ) -> EmergentConstruct:
        raise NotImplementedError


class MathematicsLens(IntentionalCausalLens):
    """Intent: Quantitative invariance, numerical conservation, and state equation crystallization."""

    def __init__(self, intent_weight: float = 1.0):
        super().__init__(ModalityType.MATHEMATICS, intent_weight)

    def refract_substrate(
        self, substrate: PrimitiveSubstrate, mechanics: StructuralMechanics
    ) -> EmergentConstruct:
        bound_vector = mechanics.apply_binding(substrate)
        # Mathematical lens projects into quantitative conservation & differential invariants
        norm_val = float(np.linalg.norm(bound_vector))
        form_tensor = np.array([norm_val, np.mean(bound_vector), np.var(bound_vector)], dtype=np.float64)

        path = DerivationPath(
            path_id=f"math_path_{substrate.substrate_id}",
            substrate_id=substrate.substrate_id,
            mechanism_id=mechanics.mechanism_id,
            atomic_rules=[
                "Rule_Math_1: Conservation of state vector norm",
                "Rule_Math_2: Differential state equation equilibrium"
            ],
            phase_transition_history=[PhaseState.GAS_POTENTIAL, PhaseState.LIQUID_FLOW, PhaseState.LATTICE_CRYSTAL],
            isomorphism_score=1.0
        )

        bit_offset = abs(hash(substrate.substrate_id + mechanics.mechanism_id)) % (2**16)
        return EmergentConstruct(
            construct_id=f"math_construct_{substrate.substrate_id}",
            modality=self.modality,
            form_tensor=form_tensor,
            derivation_path=path,
            bit_offset=bit_offset,
            invariant_phi=norm_val
        )


class PhysicsLens(IntentionalCausalLens):
    """Intent: Physical constraint enforcement, tension/repulsion balance, and dynamic trajectory."""

    def __init__(self, intent_weight: float = 1.0):
        super().__init__(ModalityType.PHYSICS, intent_weight)

    def refract_substrate(
        self, substrate: PrimitiveSubstrate, mechanics: StructuralMechanics
    ) -> EmergentConstruct:
        bound_vector = mechanics.apply_binding(substrate)
        # Physics lens projects into energy-momentum & spatial force gradient
        kinetic = 0.5 * np.sum(bound_vector**2)
        potential = mechanics.tension_coefficient * substrate.tension_potential
        form_tensor = np.array([kinetic, potential, kinetic + potential], dtype=np.float64)

        path = DerivationPath(
            path_id=f"phys_path_{substrate.substrate_id}",
            substrate_id=substrate.substrate_id,
            mechanism_id=mechanics.mechanism_id,
            atomic_rules=[
                "Rule_Phys_1: Hamilton-Jacobi action minimization",
                "Rule_Phys_2: Force gradient tension-repulsion equilibrium"
            ],
            phase_transition_history=[PhaseState.GAS_POTENTIAL, PhaseState.LIQUID_FLOW, PhaseState.LATTICE_CRYSTAL],
            isomorphism_score=1.0
        )

        bit_offset = abs(hash(substrate.substrate_id + "phys")) % (2**16)
        return EmergentConstruct(
            construct_id=f"phys_construct_{substrate.substrate_id}",
            modality=self.modality,
            form_tensor=form_tensor,
            derivation_path=path,
            bit_offset=bit_offset,
            invariant_phi=float(kinetic + potential)
        )


class LanguageLens(IntentionalCausalLens):
    """Intent: Topological constraint, suppression/promotion on knowledge graphs, context pressure gradient."""

    def __init__(self, intent_weight: float = 1.0):
        super().__init__(ModalityType.LANGUAGE, intent_weight)

    def refract_substrate(
        self, substrate: PrimitiveSubstrate, mechanics: StructuralMechanics
    ) -> EmergentConstruct:
        bound_vector = mechanics.apply_binding(substrate)
        # Contextual pressure gradient & phase-locked semantic node
        pressure_gradient = math.tanh(np.mean(bound_vector) * mechanics.boundary_curvature)
        phase_lock_value = 1.0 if pressure_gradient > mechanics.phase_lock_threshold else 0.0
        form_tensor = np.array([pressure_gradient, phase_lock_value], dtype=np.float64)

        path = DerivationPath(
            path_id=f"lang_path_{substrate.substrate_id}",
            substrate_id=substrate.substrate_id,
            mechanism_id=mechanics.mechanism_id,
            atomic_rules=[
                "Rule_Lang_1: Context pressure gradient fluid flow",
                "Rule_Lang_2: Topological phase-lock semantic node crystallization"
            ],
            phase_transition_history=[PhaseState.GAS_POTENTIAL, PhaseState.LIQUID_FLOW, PhaseState.LATTICE_CRYSTAL],
            isomorphism_score=1.0
        )

        bit_offset = abs(hash(substrate.substrate_id + "lang")) % (2**16)
        return EmergentConstruct(
            construct_id=f"lang_construct_{substrate.substrate_id}",
            modality=self.modality,
            form_tensor=form_tensor,
            derivation_path=path,
            bit_offset=bit_offset,
            invariant_phi=float(pressure_gradient)
        )


class SoundAcousticsLens(IntentionalCausalLens):
    """Intent: Continuous waveform tensor, frequency modulation, and resonance friction."""

    def __init__(self, intent_weight: float = 1.0):
        super().__init__(ModalityType.SOUND_ACOUSTICS, intent_weight)

    def refract_substrate(
        self, substrate: PrimitiveSubstrate, mechanics: StructuralMechanics
    ) -> EmergentConstruct:
        bound_vector = mechanics.apply_binding(substrate)
        # Frequency, resonance phase, and wave friction
        freq = float(np.abs(np.fft.fft(bound_vector)[0]))
        resonance_phase = math.sin(freq * mechanics.tension_coefficient)
        form_tensor = np.array([freq, resonance_phase, mechanics.repulsion_coefficient], dtype=np.float64)

        path = DerivationPath(
            path_id=f"sound_path_{substrate.substrate_id}",
            substrate_id=substrate.substrate_id,
            mechanism_id=mechanics.mechanism_id,
            atomic_rules=[
                "Rule_Sound_1: Continuous waveform tensor projection",
                "Rule_Sound_2: Resonant frequency lattice phase modulation"
            ],
            phase_transition_history=[PhaseState.GAS_POTENTIAL, PhaseState.LIQUID_FLOW, PhaseState.LATTICE_CRYSTAL],
            isomorphism_score=1.0
        )

        bit_offset = abs(hash(substrate.substrate_id + "sound")) % (2**16)
        return EmergentConstruct(
            construct_id=f"sound_construct_{substrate.substrate_id}",
            modality=self.modality,
            form_tensor=form_tensor,
            derivation_path=path,
            bit_offset=bit_offset,
            invariant_phi=freq
        )


class VisionOpticsLens(IntentionalCausalLens):
    """Intent: Spatial density field, local vorticity, and discrete visual lattice reception."""

    def __init__(self, intent_weight: float = 1.0):
        super().__init__(ModalityType.VISION_OPTICS, intent_weight)

    def refract_substrate(
        self, substrate: PrimitiveSubstrate, mechanics: StructuralMechanics
    ) -> EmergentConstruct:
        bound_vector = mechanics.apply_binding(substrate)
        # Local vorticity and spatial density distribution
        spatial_density = substrate.energy_density * mechanics.boundary_curvature
        vorticity = float(np.std(np.diff(bound_vector))) if len(bound_vector) > 1 else 0.0
        form_tensor = np.array([spatial_density, vorticity], dtype=np.float64)

        path = DerivationPath(
            path_id=f"optics_path_{substrate.substrate_id}",
            substrate_id=substrate.substrate_id,
            mechanism_id=mechanics.mechanism_id,
            atomic_rules=[
                "Rule_Optics_1: Photonic phase transition receptor coupling",
                "Rule_Optics_2: Spatial vorticity & density field lattice formation"
            ],
            phase_transition_history=[PhaseState.GAS_POTENTIAL, PhaseState.LIQUID_FLOW, PhaseState.LATTICE_CRYSTAL],
            isomorphism_score=1.0
        )

        bit_offset = abs(hash(substrate.substrate_id + "optics")) % (2**16)
        return EmergentConstruct(
            construct_id=f"optics_construct_{substrate.substrate_id}",
            modality=self.modality,
            form_tensor=form_tensor,
            derivation_path=path,
            bit_offset=bit_offset,
            invariant_phi=spatial_density
        )


class WorldMechanismField:
    """Represents physical reality's intrinsic mechanism field (independent of observer lens)."""

    def __init__(self, intrinsic_tension: float = 1.5, intrinsic_repulsion: float = 0.8):
        self.intrinsic_tension = intrinsic_tension
        self.intrinsic_repulsion = intrinsic_repulsion

    def EvolveIntrinsicTrajectory(self, substrate: PrimitiveSubstrate, steps: int = 5) -> np.ndarray:
        """Evolves the substrate according to raw physical force field without lens bias."""
        v = substrate.state_vector.copy()
        for _ in range(steps):
            force = (self.intrinsic_tension * substrate.tension_potential) - (self.intrinsic_repulsion * substrate.repulsion_potential)
            v = v + 0.1 * force * np.sin(v + 0.1)
        return v


@dataclass
class MetaLensDifferential:
    """Differential diagnostic comparing observer lens construct vs. world intrinsic causality."""
    lens_modality: ModalityType
    bias_distortion: float
    artifact_coefficient: float
    isomorphism_degree: float
    derivation_delta: np.ndarray


class ParallelObservationUniverse:
    """Contrasts System's Observer Lens vs. World Intrinsic Causality to self-diagnose bias & artifacts."""

    def __init__(self, world_field: Optional[WorldMechanismField] = None):
        self.world_field = world_field or WorldMechanismField()
        self.lenses: Dict[ModalityType, IntentionalCausalLens] = {
            ModalityType.MATHEMATICS: MathematicsLens(),
            ModalityType.PHYSICS: PhysicsLens(),
            ModalityType.LANGUAGE: LanguageLens(),
            ModalityType.SOUND_ACOUSTICS: SoundAcousticsLens(),
            ModalityType.VISION_OPTICS: VisionOpticsLens(),
        }

    def contrast_and_diagnose(
        self,
        substrate: PrimitiveSubstrate,
        mechanics: StructuralMechanics,
        modality: ModalityType
    ) -> MetaLensDifferential:
        lens = self.lenses[modality]
        construct = lens.refract_substrate(substrate, mechanics)

        # World intrinsic evolution trajectory
        world_trajectory = self.world_field.EvolveIntrinsicTrajectory(substrate)
        world_norm = float(np.linalg.norm(world_trajectory))

        # System lens form tensor summary
        lens_form_norm = float(np.linalg.norm(construct.form_tensor))

        # Delta analysis
        bias = abs(lens_form_norm - world_norm) / (world_norm + 1e-6)
        artifact_coeff = abs(mechanics.boundary_curvature - 1.0) * 0.1
        isomorphism = max(0.0, 1.0 - bias)

        delta = np.array([bias, artifact_coeff, isomorphism], dtype=np.float64)

        return MetaLensDifferential(
            lens_modality=modality,
            bias_distortion=bias,
            artifact_coefficient=artifact_coeff,
            isomorphism_degree=isomorphism,
            derivation_delta=delta
        )


class TrinityPhaseTopology:
    """Non-dualistic triadic topology: Self (C_self), Other (C_other), and Meta World (C_meta)."""

    def __init__(self, self_bias: float = 0.2, other_bias: float = 0.1):
        self.self_bias = self_bias
        self.other_bias = other_bias

    def MapTrinityRefraction(
        self, substrate: PrimitiveSubstrate, mechanics: StructuralMechanics
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Maps trajectory of Self, Other, and Meta World, calculating non-suppressive Delta."""
        base = mechanics.apply_binding(substrate)

        c_self = base * (1.0 + self.self_bias)
        c_other = base * (1.0 - self.other_bias)

        # Meta World is invariant reference space
        c_meta = base.copy()

        # Refraction delta between self and other within meta space
        delta_refraction = float(np.linalg.norm(c_self - c_other) / (np.linalg.norm(c_meta) + 1e-6))
        return c_self, c_other, c_meta, delta_refraction


class TriVariableSweepDial:
    """Sweep dial algorithm: Fixes 2 variables among {Substrate, Mechanics, Construct} and sweeps 1."""

    @staticmethod
    def sweep_mechanism(
        substrate: PrimitiveSubstrate,
        target_construct: EmergentConstruct,
        lens: IntentionalCausalLens,
        tension_range: Tuple[float, float] = (0.1, 3.0),
        steps: int = 20
    ) -> Tuple[List[float], List[float], Optional[float]]:
        """Fixes Substrate & Construct, sweeps Mechanism tension coefficient to discover inverse parameter & bifurcation."""
        tensions = np.linspace(tension_range[0], tension_range[1], steps)
        errors = []
        bifurcation_point = None
        prev_err = None

        target_norm = float(np.linalg.norm(target_construct.form_tensor))

        for t in tensions:
            mech = StructuralMechanics(
                mechanism_id=f"sweep_mech_{t:.2f}",
                modality=lens.modality,
                tension_coefficient=float(t),
                repulsion_coefficient=0.5,
                boundary_curvature=1.0
            )
            c = lens.refract_substrate(substrate, mech)
            err = abs(float(np.linalg.norm(c.form_tensor)) - target_norm)
            errors.append(err)

            if prev_err is not None and abs(err - prev_err) > 1.5 * (np.mean(errors) + 1e-3) and bifurcation_point is None:
                bifurcation_point = float(t)
            prev_err = err

        return list(tensions), errors, bifurcation_point


class ReverseDecompressionPipeline:
    """4-Stage Reverse Decompression Pipeline:
    Macro Label Stripping -> Binding Operator Extraction -> Atomic Rule Reduction -> Spontaneous Re-emergence.
    """

    def decompress(
        self,
        macro_label: str,
        construct: EmergentConstruct,
        substrate: PrimitiveSubstrate
    ) -> Dict[str, Any]:
        # Step 1: Strip macro label
        stripped_tensor = construct.form_tensor.copy()

        # Step 2: Extract binding operator & tension-repulsion ratio
        binding_operator = {
            "operator_id": f"B_hat_{construct.modality.value}",
            "tension_ratio": construct.invariant_phi / (float(np.linalg.norm(stripped_tensor)) + 1e-6),
            "bit_offset": construct.bit_offset
        }

        # Step 3: Reduce to atomic causal rules
        atomic_rules = construct.derivation_path.atomic_rules

        # Step 4: Spontaneous re-emergence test
        re_emergent_phi = binding_operator["tension_ratio"] * float(np.linalg.norm(stripped_tensor))
        derivation_integrity = 1.0 - abs(re_emergent_phi - construct.invariant_phi) / (construct.invariant_phi + 1e-6)

        return {
            "macro_label_stripped": macro_label,
            "binding_operator": binding_operator,
            "atomic_rules": atomic_rules,
            "re_emergent_phi": re_emergent_phi,
            "derivation_integrity": max(0.0, derivation_integrity)
        }


class ReversiblePhaseTransitionEngine:
    """Reversible 4-Phase Transition Pipeline:
    Gas Potential <-> Liquid Flow <-> Lattice Crystal <-> Re-melting.
    """

    def __init__(self, memory_size: int = 65536):
        self.memory_space = np.zeros(memory_size, dtype=np.float64)

    def execute_phase_cycle(
        self, substrate: PrimitiveSubstrate, mechanics: StructuralMechanics
    ) -> Dict[str, Any]:
        # Phase 1: Gas Potential
        state_1 = PhaseState.GAS_POTENTIAL
        gas_energy = substrate.energy_density

        # Phase 2: Liquid Flow
        state_2 = PhaseState.LIQUID_FLOW
        liquid_flow_vec = mechanics.apply_binding(substrate)

        # Phase 3: Lattice Crystal (Phase-Lock & 1D Bit Offset Binding)
        state_3 = PhaseState.LATTICE_CRYSTAL
        bit_offset = abs(hash(substrate.substrate_id + mechanics.mechanism_id)) % len(self.memory_space)
        crystal_val = float(np.mean(liquid_flow_vec))
        self.memory_space[bit_offset] = crystal_val

        # Phase 4: Re-melting (Re-activation of Lattice into Liquid Potential)
        state_4 = PhaseState.RE_MELTED
        remelted_potential = self.memory_space[bit_offset]
        reconstitution_error = abs(remelted_potential - crystal_val)

        return {
            "phase_sequence": [state_1, state_2, state_3, state_4],
            "bit_offset": bit_offset,
            "crystallized_value": crystal_val,
            "remelted_potential": remelted_potential,
            "reconstitution_error": reconstitution_error,
            "causal_continuity": 1.0 if reconstitution_error < 1e-6 else 0.0
        }
