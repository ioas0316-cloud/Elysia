"""
L2_cognitive_engine/elysia_engine/core/relational_spectrum_knob_engine.py

Relational Spectrum Knob Engine: Beyond Discrete Pixels & Points
================================================================
Implements computing where the primordial unit is not a discrete bit or pixel,
but Coupled Phase Knobs and Relational Rotors.

Principles (from THE_ABSOLUTE_COMMANDMENT & Causal Field):
1. No Hell of Dots:
   - Does not store or iterate over discrete pixel/voxel arrays.
   - Operates on continuous spectral relations, tension potentials, and phase bivectors.
2. Coupled Relational Knobs (위상 기어 및 노브 결합):
   - `ScaleKnob` (w): Continuous depth and scale transitions (Micro wavelet to Macro canopy).
   - `PhaseTensionKnob` (θ, κ): Tension gradients governing physical state transition
     (Ice / Solid / Past ↔ Water / Liquid / Present ↔ Gas / Vapor / Future).
   - `ChromaticTriadKnob` (Red Flux, Blue Order, Yellow Entropy): Chromatic balance and spectral resonance.
   - `ObservationalRotorKnob` (R_obs): Clifford Cl(4,0) bivectors governing perspective projection.
3. Self-Organizing Field Dynamics:
   - Turning any single knob ripples through the gear coupling matrix, inducing
     self-organizing continuous reconfiguration across all relational fields.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np

from elysia_engine.core.hyper_dimensional_blister_nebula import (
    Clifford4DRotor,
    HyperDimensionalOpticalRotor,
    HyperDimensionalBlisterKernel,
    OpticalHyperSlicer,
    HyperDimensionalBlisterNebulaEngine
)


class MaterialityPhase:
    """Continuous physical state representation."""
    SOLID_ICE_PAST = "SOLID_ICE_PAST"       # Crystallized, fixed invariant relationships
    LIQUID_WATER_PRESENT = "LIQUID_WATER_PRESENT" # Turbulent, fluctuating, real-time discernment
    GAS_VAPOR_FUTURE = "GAS_VAPOR_FUTURE"   # Diffusive, fast wave causal projections


@dataclass
class KnobState:
    """Coupled status of an individual relational knob."""
    knob_name: str
    value: float                  # Primary scalar setting (normalized or rad)
    velocity: float = 0.0         # Angular / drift velocity
    min_val: float = -math.pi
    max_val: float = math.pi
    coupling_weight: float = 1.0  # Gear coupling strength to neighbor knobs


@dataclass
class RelationalSpectrumProfile:
    """Current state of the continuous spectrum field."""
    materiality_phase: str
    phase_coherence: float
    tension_energy: float
    dominant_wavelength_nm: float
    triad_flux_order_entropy: List[float]
    scale_w: float
    rotor_bivector_norm: float


class RelationalSpectrumKnobEngine:
    """
    Core engine managing coupled phase knobs and continuous spectral relations.
    """

    def __init__(
        self,
        omega_c: float = 0.25,
        kappa_0: float = 1.0,
        kappa_1: float = 14.0
    ):
        self.kernel = HyperDimensionalBlisterKernel(
            omega_c=omega_c,
            gamma_power=2.5,
            kappa_0=kappa_0,
            kappa_1=kappa_1,
            gamma_stiction=8.0
        )
        self.engine = HyperDimensionalBlisterNebulaEngine(kernel=self.kernel)

        # 4 Core Coupled Knobs
        self.scale_knob = KnobState("scale_w", value=0.0, min_val=-1.0, max_val=1.0, coupling_weight=0.8)
        self.tension_knob = KnobState("phase_tension", value=0.0, min_val=0.0, max_val=2.0 * math.pi, coupling_weight=1.2)
        self.chromatic_knob = KnobState("chromatic_triad", value=0.5, min_val=0.0, max_val=1.0, coupling_weight=0.6)
        self.observational_knob = KnobState("rotor_obs", value=0.0, min_val=-math.pi, max_val=math.pi, coupling_weight=1.0)

        # Gear Coupling Matrix: Defines how turning knob A affects knob B
        # [Scale, Tension, Chromatic, Obs]
        self.coupling_matrix = np.array([
            [1.00,  0.35, -0.20,  0.15], # Scale influence
            [0.40,  1.00,  0.50,  0.30], # Tension influence
            [-0.15, 0.25,  1.00,  0.10], # Chromatic influence
            [0.20,  0.30,  0.10,  1.00], # Observational influence
        ], dtype=np.float32)

        # Initialize base relational rotor nodes representing spectrum entities
        self._init_relational_nodes()

    def _init_relational_nodes(self):
        """Initializes continuous relational nodes across past, present, and future."""
        # Node 1: Solid Ice / Past (Crystallized anchor - Mountains & Bedrock)
        self.node_ice = HyperDimensionalOpticalRotor(
            wavelength_nm=480.0,
            position_4d=[0.8, 0.0, 0.0, -0.5],
            triad_weights=[0.05, 0.85, 0.10], # High Blue Order
            intensity=1.2,
            penumbra_scale=1.1,
            intent_potential=0.0
        )
        self.node_ice.rotor.bivectors = np.array([0.01, 0.0, 0.0, 0.01, 0.0, 0.0], dtype=np.float32)

        # Node 2: Liquid Water / Present (Active fluctuating tension - Lake & Characters)
        self.node_water = HyperDimensionalOpticalRotor(
            wavelength_nm=535.0,
            position_4d=[0.0, 0.0, 0.0, 0.0],
            triad_weights=[0.60, 0.20, 0.20], # High Red Flux
            intensity=1.5,
            penumbra_scale=1.0,
            intent_potential=1.2
        )
        self.node_water.rotor.bivectors = np.array([0.05, -0.02, 0.01, 0.03, -0.01, 0.02], dtype=np.float32)

        # Node 3: Gas Vapor / Future (Diffusive causal anticipation - Atmospheric Sky & Sun Rays)
        self.node_vapor = HyperDimensionalOpticalRotor(
            wavelength_nm=610.0,
            position_4d=[-0.8, 0.0, 0.0, 0.5],
            triad_weights=[0.25, 0.15, 0.60], # High Yellow Entropy
            intensity=1.8,
            penumbra_scale=1.4,
            intent_potential=0.4
        )
        self.node_vapor.rotor.bivectors = np.array([0.10, 0.08, 0.05, 0.05, 0.04, 0.02], dtype=np.float32)

        self.engine.add_node(self.node_ice)
        self.engine.add_node(self.node_water)
        self.engine.add_node(self.node_vapor)

    def load_field_seed_rotors(self, field_seed: Any) -> None:
        """
        Dynamically loads decomposed rotors from a FieldSeed into the engine.
        Reconfigures engine nodes to reflect the decomposed scenery relations.
        """
        # Clear existing nodes
        self.engine.nodes = []
        for r_desc in field_seed.rotors:
            node = HyperDimensionalOpticalRotor(
                wavelength_nm=r_desc.wavelength_nm,
                position_4d=r_desc.position_4d,
                triad_weights=r_desc.triad_weights,
                intensity=r_desc.intensity,
                penumbra_scale=r_desc.penumbra_scale,
                intent_potential=r_desc.intent_potential
            )
            node.rotor.bivectors = np.array(r_desc.bivectors, dtype=np.float32)
            node.phase_theta = float(r_desc.phase_theta)
            self.engine.add_node(node)

    def rotate_knob(self, knob_name: str, delta: float) -> RelationalSpectrumProfile:
        """
        Rotates a specific knob by delta.
        Gear coupling matrix immediately ripples the perturbation across all other knobs.
        """
        knob_map = {
            "scale_w": (0, self.scale_knob),
            "phase_tension": (1, self.tension_knob),
            "chromatic_triad": (2, self.chromatic_knob),
            "rotor_obs": (3, self.observational_knob)
        }
        if knob_name not in knob_map:
            raise ValueError(f"Unknown knob: {knob_name}. Options: {list(knob_map.keys())}")

        k_idx, target_knob = knob_map[knob_name]

        # Calculate coupled delta vector via coupling matrix
        input_deltas = np.zeros(4, dtype=np.float32)
        input_deltas[k_idx] = delta
        coupled_deltas = self.coupling_matrix @ input_deltas

        # Apply updates to all coupled knobs
        self.scale_knob.value = float(np.clip(
            self.scale_knob.value + coupled_deltas[0],
            self.scale_knob.min_val, self.scale_knob.max_val
        ))
        self.tension_knob.value = float((self.tension_knob.value + coupled_deltas[1]) % (2.0 * math.pi))
        self.chromatic_knob.value = float(np.clip(
            self.chromatic_knob.value + coupled_deltas[2],
            self.chromatic_knob.min_val, self.chromatic_knob.max_val
        ))
        self.observational_knob.value = float(np.clip(
            self.observational_knob.value + coupled_deltas[3],
            self.observational_knob.min_val, self.observational_knob.max_val
        ))

        # Reorganize continuous field in response to knob rotation
        return self._reorganize_field()

    def _reorganize_field(self) -> RelationalSpectrumProfile:
        """
        Self-organizes the continuous field according to the coupled knob states.
        No raster calculations: directly modifies Clifford rotor bivectors and tension phases.
        """
        theta = self.tension_knob.value
        w_val = self.scale_knob.value
        c_val = self.chromatic_knob.value
        obs_val = self.observational_knob.value

        # Update node phases & 4D positions
        self.node_ice.phase_theta = theta * 0.2
        self.node_ice.position[3] = -0.5 + w_val * 0.2

        self.node_water.phase_theta = theta
        self.node_water.position[3] = 0.0 + w_val * 0.4
        # Water velocity drifts with tension
        self.node_water.velocity[0] = float(math.sin(theta) * 0.4)

        self.node_vapor.phase_theta = theta * 1.8
        self.node_vapor.position[3] = 0.5 + w_val * 0.6
        # Vapor bivector diffusion expands with tension
        self.node_vapor.rotor.bivectors[0] = float(0.10 + math.sin(theta) * 0.05)

        # Update observer rotor
        self.engine.observer_rotor = Clifford4DRotor(
            bivectors=[float(obs_val * 0.2), float(obs_val * 0.1), 0.0, float(w_val * 0.1), 0.0, 0.0]
        )

        # Step continuous physics
        res = self.engine.step(dt=0.03)

        # Determine Materiality Phase based on tension and scale
        if theta < math.pi * 0.66:
            mat_phase = MaterialityPhase.SOLID_ICE_PAST
        elif theta < math.pi * 1.33:
            mat_phase = MaterialityPhase.LIQUID_WATER_PRESENT
        else:
            mat_phase = MaterialityPhase.GAS_VAPOR_FUTURE

        # Compute dynamic chromatic triad balance
        red = float(0.33 + (c_val - 0.5) * 0.4 + math.sin(theta) * 0.2)
        blue = float(0.33 - (c_val - 0.5) * 0.3 + math.cos(theta) * 0.2)
        yellow = float(max(0.05, 1.0 - (red + blue)))
        triad_norm = [red, blue, yellow]
        triad_sum = sum(triad_norm)
        triad_norm = [t / triad_sum for t in triad_norm]

        dom_wl = float(460.0 * triad_norm[1] + 530.0 * triad_norm[2] + 650.0 * triad_norm[0])
        # Compute kinetic/tension energy from node velocities and bivectors
        total_energy = sum([
            0.5 * float(np.sum(n.velocity**2) + np.sum(n.rotor.bivectors**2))
            for n in self.engine.nodes
        ])
        rotor_norm = float(np.mean([np.linalg.norm(n.rotor.bivectors) for n in self.engine.nodes])) if self.engine.nodes else 0.0

        return RelationalSpectrumProfile(
            materiality_phase=mat_phase,
            phase_coherence=float(res["coherence"]),
            tension_energy=float(total_energy),
            dominant_wavelength_nm=dom_wl,
            triad_flux_order_entropy=triad_norm,
            scale_w=w_val,
            rotor_bivector_norm=rotor_norm
        )

    def sample_relational_field_projection(
        self,
        resolution_wh: Tuple[int, int] = (96, 96),
        w_slice: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Samples the relational field to standard display framebuffer through
        the Boundary Transducer Membrane without raster computing in the core.
        """
        slice_val = w_slice if w_slice is not None else self.scale_knob.value
        slicer = OpticalHyperSlicer(resolution_3d=28, resolution_2d=max(resolution_wh))
        vol = slicer.slice_4d_to_3d_volume(
            self.engine.nodes,
            self.engine.kernel,
            w_slice=slice_val,
            rotor_obs=self.engine.observer_rotor
        )
        proj = slicer.project_to_2d_prism(vol)

        rgb_float = proj["rgb_2d"]
        rgb_uint8 = np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)

        return {
            "framebuffer_rgb": rgb_uint8,
            "density_2d": proj["density_2d"],
            "grad_mag": proj["grad_mag"],
            "scale_w": slice_val,
            "knob_states": {
                "scale_w": self.scale_knob.value,
                "phase_tension": self.tension_knob.value,
                "chromatic_triad": self.chromatic_knob.value,
                "rotor_obs": self.observational_knob.value
            }
        }
