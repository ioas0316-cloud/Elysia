"""
Project Elysia: Cognitive OS Organ Adapters & World Field Simulators
===================================================================
Implements media/software environment simulators as active sensory-motor organ
adapters for the Cognitive OS architecture.

Rather than isolated external software applications, Blender, CAD, Photoshop,
Game Engines, Particle/Fluid Sandboxes, and Non-Euclidean Spacetime Fields are
modeled as active sensory organs (망막, 촉각, 고유수용성 감각기관, 인과 물리 장).

Key Concepts:
1. CognitiveOSOrganAdapter (감각신경기관 어댑터):
   Base class bridging external field states/software APIs with Perceptual Probes.
2. ParticleFluidSandboxOrgan (입자/유체 상전이 샌드박스 장):
   Local FDTD / cellular density interactions & phase transitions (fluid-solid).
3. NonEuclideanSpacetimeOrgan (비유클리드 위상 장):
   Spacetime metric deformation, curvature $K_c$, and Clifford rotor rotation.
4. BlenderMeshOrgan (블렌더 3D 위상 촉각):
   3D surface topology, vertex deformation, curvature friction.
5. CADConstraintOrgan (CAD 구조적 인과 제약):
   Mechanical constraint manifold & degree-of-freedom tension.
6. PhotoshopRetinaOrgan (포토샵 시각/색채 망막):
   Chromatic spectrum field (Red Flux, Blue Order, Yellow Entropy).
7. GameEnginePhysicsOrgan (게임엔진 실시간 인과 물리):
   Action-reaction collision dynamics & state transition topology.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


@dataclass
class OrganSensation:
    """Sensation payload returned by a Cognitive OS Organ Adapter."""
    organ_id: str
    organ_type: str
    sensation_vector: np.ndarray # High-dimensional sensory state
    resistance_friction: float # Physical/informational friction (\mathcal{E})
    field_curvature: float # Local metric/topological curvature
    echo_latency: float # Indirect control echo latency / gradient of agency
    metadata: Dict[str, Any] = field(default_factory=dict)


class CognitiveOSOrganAdapter(ABC):
    """
    Base abstraction for external software/world fields operating as
    Elysia's Sensory-Motor Organs (감각신경기관).
    """
    def __init__(self, organ_id: str, organ_type: str, dimension: int = 5):
        self.organ_id = organ_id
        self.organ_type = organ_type
        self.dimension = dimension
        self.state_vector = np.zeros(dimension, dtype=np.float32)
        self.intrinsic_friction: float = 0.05
        self.echo_latency: float = 0.1 # Distance in causal gravity

    @abstractmethod
    def sense_and_intervene(
        self,
        probe_action: np.ndarray,
        causal_gravity: float = 1.0
    ) -> OrganSensation:
        """
        Receives probe action from Elysia, updates organ internal state,
        computes resistance friction and echo latency, and returns OrganSensation.
        """
        pass


class ParticleFluidSandboxOrgan(CognitiveOSOrganAdapter):
    """
    [입자/유체 상전이 샌드박스 장 Organ]
    Simulates local FDTD particle density, fluid-solid phase transitions,
    and friction from local neighbor interactions.
    """
    def __init__(self, organ_id: str = "fluid_sandbox_01", dimension: int = 5, grid_size: int = 16):
        super().__init__(organ_id=organ_id, organ_type="particle_fluid_sandbox", dimension=dimension)
        self.grid_size = grid_size
        self.density_field = np.random.uniform(0.1, 0.9, size=(grid_size, grid_size)).astype(np.float32)
        self.temperature = 0.5 # Phase transition parameter (solid < 0.3, fluid 0.3-0.8, gas > 0.8)

    def sense_and_intervene(
        self,
        probe_action: np.ndarray,
        causal_gravity: float = 1.0
    ) -> OrganSensation:
        action_vec = np.array(probe_action, dtype=np.float32).flatten()[:self.dimension]
        if len(action_vec) < self.dimension:
            action_vec = np.pad(action_vec, (0, self.dimension - len(action_vec)))

        # Update temperature / fluid state based on action energy
        action_energy = float(np.linalg.norm(action_vec))
        self.temperature = float(0.9 * self.temperature + 0.1 * action_energy)

        # Local FDTD density diffusion step
        kernel = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]], dtype=np.float32)
        # Simplified 2D convolution for density update
        padded = np.pad(self.density_field, 1, mode='wrap')
        new_density = np.zeros_like(self.density_field)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                new_density[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
        self.density_field = new_density

        # Density variance measures structural friction & phase state
        density_var = float(np.var(self.density_field))
        friction = float(density_var * 2.0 + abs(self.temperature - 0.5) * 0.5)

        # Extract sensory vector from density moments
        self.state_vector = np.array([
            np.mean(self.density_field),
            density_var,
            self.temperature,
            float(np.max(self.density_field)),
            float(np.min(self.density_field))
        ], dtype=np.float32)

        return OrganSensation(
            organ_id=self.organ_id,
            organ_type=self.organ_type,
            sensation_vector=self.state_vector.copy(),
            resistance_friction=friction,
            field_curvature=density_var,
            echo_latency=self.echo_latency * (1.0 + friction),
            metadata={
                "temperature": self.temperature,
                "phase_state": "solid" if self.temperature < 0.3 else ("fluid" if self.temperature < 0.8 else "gas")
            }
        )


class NonEuclideanSpacetimeOrgan(CognitiveOSOrganAdapter):
    """
    [비유클리드 위상 장 Organ]
    Renders curved spacetime metric deformation, Clifford rotor rotations,
    and topological holonomy shifts.
    """
    def __init__(self, organ_id: str = "noneuclidean_field_01", dimension: int = 5):
        super().__init__(organ_id=organ_id, organ_type="noneuclidean_spacetime", dimension=dimension)
        # Clifford rotor angle components (bivectors)
        self.rotor_bivectors = np.zeros(3, dtype=np.float32)
        self.curvature_k = 0.1 # Gaussian curvature

    def sense_and_intervene(
        self,
        probe_action: np.ndarray,
        causal_gravity: float = 1.0
    ) -> OrganSensation:
        action_vec = np.array(probe_action, dtype=np.float32).flatten()[:self.dimension]
        if len(action_vec) < self.dimension:
            action_vec = np.pad(action_vec, (0, self.dimension - len(action_vec)))

        # Clifford Rotor rotation shift along bivector components
        bivector_delta = action_vec[:3] * 0.2
        self.rotor_bivectors += bivector_delta
        # Keep rotor bivectors bounded in [-pi, pi]
        self.rotor_bivectors = np.remainder(self.rotor_bivectors + np.pi, 2 * np.pi) - np.pi

        # Curvature dynamic shift
        self.curvature_k = float(np.sin(self.rotor_bivectors[0]) * np.cos(self.rotor_bivectors[1]))

        # Holonomy friction arises from path-dependence in curved space
        holonomy_friction = float(abs(self.curvature_k) * np.linalg.norm(self.rotor_bivectors))

        self.state_vector = np.array([
            np.cos(self.rotor_bivectors[0]),
            np.sin(self.rotor_bivectors[1]),
            self.rotor_bivectors[2],
            self.curvature_k,
            float(np.linalg.norm(self.rotor_bivectors))
        ], dtype=np.float32)

        return OrganSensation(
            organ_id=self.organ_id,
            organ_type=self.organ_type,
            sensation_vector=self.state_vector.copy(),
            resistance_friction=holonomy_friction,
            field_curvature=abs(self.curvature_k),
            echo_latency=self.echo_latency * (1.0 + abs(self.curvature_k)),
            metadata={
                "rotor_bivectors": self.rotor_bivectors.tolist(),
                "gaussian_curvature": self.curvature_k
            }
        )


class BlenderMeshOrgan(CognitiveOSOrganAdapter):
    """
    [블렌더 3D 위상 촉각 Organ]
    Simulates 3D mesh surface topology, vertex displacement, and geometric curvature friction.
    """
    def __init__(self, organ_id: str = "blender_mesh_organ", dimension: int = 5):
        super().__init__(organ_id=organ_id, organ_type="blender_3d_mesh", dimension=dimension)
        self.mesh_vertices = np.random.normal(0, 1, size=(20, 3)).astype(np.float32)

    def sense_and_intervene(
        self,
        probe_action: np.ndarray,
        causal_gravity: float = 1.0
    ) -> OrganSensation:
        action_vec = np.array(probe_action, dtype=np.float32).flatten()[:self.dimension]
        if len(action_vec) < self.dimension:
            action_vec = np.pad(action_vec, (0, self.dimension - len(action_vec)))

        # Sculpt mesh vertices with action displacement
        disp = action_vec[:3] * 0.05
        self.mesh_vertices += disp

        # Compute mesh surface normal variance and topological tension
        norms = np.linalg.norm(self.mesh_vertices, axis=1)
        mean_radius = float(np.mean(norms))
        surface_roughness = float(np.std(norms))
        friction = float(surface_roughness * 1.5)

        self.state_vector = np.array([
            mean_radius,
            surface_roughness,
            float(np.max(norms)),
            float(np.min(norms)),
            float(np.sum(disp))
        ], dtype=np.float32)

        return OrganSensation(
            organ_id=self.organ_id,
            organ_type=self.organ_type,
            sensation_vector=self.state_vector.copy(),
            resistance_friction=friction,
            field_curvature=surface_roughness,
            echo_latency=self.echo_latency * 1.2,
            metadata={"num_vertices": len(self.mesh_vertices), "mean_radius": mean_radius}
        )


class CADConstraintOrgan(CognitiveOSOrganAdapter):
    """
    [CAD 구조적 인과 제약 Organ]
    Simulates mechanical degree-of-freedom constraints and structural tolerance tension.
    """
    def __init__(self, organ_id: str = "cad_constraint_organ", dimension: int = 5):
        super().__init__(organ_id=organ_id, organ_type="cad_constraints", dimension=dimension)
        self.dof_constraints = np.array([1.0, 0.5, 0.2, 0.8, 0.0], dtype=np.float32)
        self.tolerance_error = 0.01

    def sense_and_intervene(
        self,
        probe_action: np.ndarray,
        causal_gravity: float = 1.0
    ) -> OrganSensation:
        action_vec = np.array(probe_action, dtype=np.float32).flatten()[:self.dimension]
        if len(action_vec) < self.dimension:
            action_vec = np.pad(action_vec, (0, self.dimension - len(action_vec)))

        # Mechanical force against fixed DOF constraints produces reaction friction
        constraint_violation = np.abs(action_vec - self.dof_constraints)
        friction = float(np.sum(constraint_violation) * 0.8)

        self.state_vector = np.clip(action_vec * self.dof_constraints, -2.0, 2.0)

        return OrganSensation(
            organ_id=self.organ_id,
            organ_type=self.organ_type,
            sensation_vector=self.state_vector.copy(),
            resistance_friction=friction,
            field_curvature=float(np.max(constraint_violation)),
            echo_latency=self.echo_latency * 1.5, # High mechanical rigidity latency
            metadata={"dof_active": int(np.sum(self.dof_constraints > 0))}
        )


class PhotoshopRetinaOrgan(CognitiveOSOrganAdapter):
    """
    [포토샵 시각/색채 망막 Organ]
    Renders chromatic spectrum field (Red Flux, Blue Order, Yellow Entropy).
    """
    def __init__(self, organ_id: str = "photoshop_retina_organ", dimension: int = 5):
        super().__init__(organ_id=organ_id, organ_type="photoshop_chromatic_retina", dimension=dimension)
        self.chromatic_triad = np.array([0.33, 0.33, 0.34], dtype=np.float32) # [Red, Blue, Yellow]

    def sense_and_intervene(
        self,
        probe_action: np.ndarray,
        causal_gravity: float = 1.0
    ) -> OrganSensation:
        action_vec = np.array(probe_action, dtype=np.float32).flatten()[:self.dimension]
        if len(action_vec) < self.dimension:
            action_vec = np.pad(action_vec, (0, self.dimension - len(action_vec)))

        # Shift Red (Flux), Blue (Order), Yellow (Entropy) balance
        self.chromatic_triad += action_vec[:3] * 0.1
        self.chromatic_triad = np.maximum(0.01, self.chromatic_triad)
        self.chromatic_triad /= np.sum(self.chromatic_triad)

        # Chromatic entropy mismatch produces visual friction
        entropy = float(-np.sum(self.chromatic_triad * np.log(self.chromatic_triad + 1e-6)))
        friction = float(abs(entropy - 1.0) * 0.5)

        self.state_vector = np.array([
            self.chromatic_triad[0], # Red (Flux)
            self.chromatic_triad[1], # Blue (Order)
            self.chromatic_triad[2], # Yellow (Entropy)
            entropy,
            friction
        ], dtype=np.float32)

        return OrganSensation(
            organ_id=self.organ_id,
            organ_type=self.organ_type,
            sensation_vector=self.state_vector.copy(),
            resistance_friction=friction,
            field_curvature=entropy,
            echo_latency=self.echo_latency * 0.8, # Fast optical feedback
            metadata={"red_flux": self.chromatic_triad[0], "blue_order": self.chromatic_triad[1], "yellow_entropy": self.chromatic_triad[2]}
        )


class GameEnginePhysicsOrgan(CognitiveOSOrganAdapter):
    """
    [게임엔진 실시간 인과 물리 Organ]
    Simulates real-time action-reaction collision, rigid body momentum,
    and state transition topology.
    """
    def __init__(self, organ_id: str = "game_engine_physics_organ", dimension: int = 5):
        super().__init__(organ_id=organ_id, organ_type="game_engine_physics", dimension=dimension)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.collision_impulse = 0.0

    def sense_and_intervene(
        self,
        probe_action: np.ndarray,
        causal_gravity: float = 1.0
    ) -> OrganSensation:
        action_vec = np.array(probe_action, dtype=np.float32).flatten()[:self.dimension]
        if len(action_vec) < self.dimension:
            action_vec = np.pad(action_vec, (0, self.dimension - len(action_vec)))

        # Accelerate character/entity in game physics state machine
        self.velocity = 0.8 * self.velocity + 0.2 * action_vec[:3]
        speed = float(np.linalg.norm(self.velocity))

        # Random collision resistance when speed is high
        self.collision_impulse = float(speed * (0.2 if np.random.rand() > 0.3 else 0.8))
        friction = self.collision_impulse

        self.state_vector = np.array([
            self.velocity[0],
            self.velocity[1],
            self.velocity[2],
            speed,
            self.collision_impulse
        ], dtype=np.float32)

        return OrganSensation(
            organ_id=self.organ_id,
            organ_type=self.organ_type,
            sensation_vector=self.state_vector.copy(),
            resistance_friction=friction,
            field_curvature=self.collision_impulse,
            echo_latency=self.echo_latency * 1.0,
            metadata={"speed": speed, "collision_impulse": self.collision_impulse}
        )
