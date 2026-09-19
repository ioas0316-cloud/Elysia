"""
transformative_gear_engine.py
==============================
Meta-Ontological Engine implementing:
[Component Data Plane × Transformative Gear = Emergent Volume]
and Compressed Causal Seals with Phase Re-ignition.

Core Concepts:
1. ComponentDataPlane:
   Micro-level state tensors (positions, momentum, chromatic/phase spectrum).
2. TransformativeGear:
   Dynamic transformation matrix/tensor operator governed by Cognitive Temperature (T_cog),
   Attention Density, and Degrees of Freedom, driving Phase Transitions (Gas <-> Liquid <-> Solid).
3. EmergentCausalVolume:
   Spatiotemporal causal volume / phase space volume with conservation laws,
   structural deformation, and shear stress metrics.
4. CompressedCausalSeal:
   Macro-level compressed seal (concept/symbol) capable of Phase Re-ignition
   (decompressing into micro data planes and dynamic gears with initial voltage).
5. Cognitive Temperature & Isomorphic Feedback Loop:
   Thermodynamic cognitive feedback driven by prediction error (Delta), angular vector mismatch,
   and structural friction (Arousal & Metabolic Heat).
"""

import math
from typing import Dict, Any, Tuple, Optional, Union
import torch
import numpy as np


class PhaseState:
    GAS = "gas"        # Superposition / Unconstrained / High DOF
    LIQUID = "liquid"  # Fluid Reasoning / Adaptable coupling / Flow
    SOLID = "solid"    # Crystallized Memory / Rigid constraint / High Density


class ComponentDataPlane:
    """
    [Component Data Plane (구성요소 데이터 평면)]
    Micro-level state tensor containing raw position, momentum, phase/chromatic spectrum,
    and topological connectivity for N particles or voxels.
    """
    def __init__(
        self,
        positions: Union[torch.Tensor, np.ndarray],
        velocities: Optional[Union[torch.Tensor, np.ndarray]] = None,
        chromatic_spectrum: Optional[Union[torch.Tensor, np.ndarray]] = None,
        device: str = "cpu"
    ):
        self.device = device
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions).float()
        self.positions = positions.to(self.device)  # Shape: (N, D)

        self.num_nodes, self.dim = self.positions.shape

        if velocities is None:
            self.velocities = torch.zeros_like(self.positions, device=self.device)
        else:
            if isinstance(velocities, np.ndarray):
                velocities = torch.from_numpy(velocities).float()
            self.velocities = velocities.to(self.device)

        if chromatic_spectrum is None:
            # Chromatic spectrum: Red/Blue/Yellow (Flux, Order, Entropy)
            self.chromatic_spectrum = torch.ones((self.num_nodes, 3), device=self.device) * 0.333
        else:
            if isinstance(chromatic_spectrum, np.ndarray):
                chromatic_spectrum = torch.from_numpy(chromatic_spectrum).float()
            self.chromatic_spectrum = chromatic_spectrum.to(self.device)

    def clone(self) -> 'ComponentDataPlane':
        return ComponentDataPlane(
            positions=self.positions.clone(),
            velocities=self.velocities.clone(),
            chromatic_spectrum=self.chromatic_spectrum.clone(),
            device=self.device
        )

    def compute_center_of_mass(self) -> torch.Tensor:
        return torch.mean(self.positions, dim=0)

    def compute_total_momentum(self) -> torch.Tensor:
        return torch.sum(self.velocities, dim=0)


class TransformativeGear:
    """
    [Transformative Gear (변환 기어)]
    Dynamic transformation tensor operator acting on ComponentDataPlane.
    Controls phase transitions (Gas <-> Liquid <-> Solid) based on Cognitive Temperature,
    Attention Density, and Degrees of Freedom.
    """
    def __init__(
        self,
        dim: int = 3,
        cognitive_temperature: float = 1.0,  # T_cog
        attention_density: float = 1.0,      # Attention pressure
        degrees_of_freedom: float = 1.0,     # Constraint index (1.0 = gas, 0.5 = liquid, 0.0 = solid)
        device: str = "cpu"
    ):
        self.dim = dim
        self.device = device
        self.cognitive_temperature = cognitive_temperature
        self.attention_density = attention_density
        self.degrees_of_freedom = degrees_of_freedom

        # Gear transformation matrix G (dim x dim)
        self.gear_matrix = torch.eye(dim, device=self.device)
        self.update_phase_state()

    def update_phase_state(self) -> str:
        """
        Determines current PhaseState from cognitive_temperature and degrees_of_freedom.
        High T_cog (> 2.0) -> GAS
        Medium T_cog (0.5 - 2.0) -> LIQUID
        Low T_cog (< 0.5) -> SOLID
        """
        if self.cognitive_temperature > 2.0 or self.degrees_of_freedom > 0.8:
            self.current_phase = PhaseState.GAS
        elif self.cognitive_temperature < 0.5 and self.degrees_of_freedom < 0.3:
            self.current_phase = PhaseState.SOLID
        else:
            self.current_phase = PhaseState.LIQUID
        return self.current_phase

    def construct_gear_matrix(self, angular_velocity: float = 0.1, dt: float = 0.1) -> torch.Tensor:
        """
        Constructs the transformation gear tensor matrix based on current phase and temperature.
        """
        theta = angular_velocity * dt * self.cognitive_temperature
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        if self.dim == 2:
            rot = torch.tensor([
                [cos_t, -sin_t],
                [sin_t,  cos_t]
            ], device=self.device)
        else:
            # 3D rotation around z-axis + thermal shear
            rot = torch.tensor([
                [cos_t, -sin_t, 0.0],
                [sin_t,  cos_t, 0.0],
                [0.0,    0.0,   1.0]
            ], device=self.device)

        # Scale shear/damping by phase state viscosity
        if self.current_phase == PhaseState.SOLID:
            viscosity_scale = 0.05  # Rigid
        elif self.current_phase == PhaseState.LIQUID:
            viscosity_scale = 0.5   # Adaptive flow
        else:
            viscosity_scale = 1.2   # Free dispersion

        self.gear_matrix = rot * viscosity_scale
        return self.gear_matrix

    def transform_data_plane(
        self,
        data_plane: ComponentDataPlane,
        dt: float = 0.1
    ) -> ComponentDataPlane:
        """
        Applies gear transformation matrix G to component positions and velocities.
        """
        self.update_phase_state()
        G = self.construct_gear_matrix(angular_velocity=1.0, dt=dt)

        new_positions = torch.matmul(data_plane.positions, G)

        # Velocity update influenced by phase viscosity and temperature
        thermal_noise = torch.randn_like(data_plane.velocities) * (self.cognitive_temperature * 0.05)
        new_velocities = torch.matmul(data_plane.velocities, G) + thermal_noise

        if self.current_phase == PhaseState.SOLID:
            # Damped motion in solid phase
            new_velocities = new_velocities * 0.2
        elif self.current_phase == PhaseState.LIQUID:
            new_velocities = new_velocities * 0.85

        return ComponentDataPlane(
            positions=new_positions,
            velocities=new_velocities,
            chromatic_spectrum=data_plane.chromatic_spectrum.clone(),
            device=self.device
        )


class EmergentCausalVolume:
    """
    [Emergent Causal Volume (인과적 현상 체적)]
    Spatiotemporal causal volume generated by DataPlane passing through TransformativeGear.
    Computes volumetric metrics, shear stress, momentum conservation, and topological bounding volume.
    """
    def __init__(
        self,
        data_plane: ComponentDataPlane,
        gear: TransformativeGear,
        time_step: float = 0.0
    ):
        self.data_plane = data_plane
        self.gear = gear
        self.time_step = time_step
        self.volume_metric, self.bounding_box = self._calculate_volumetric_metrics()
        self.shear_stress = self._calculate_shear_stress()

    def _calculate_volumetric_metrics(self) -> Tuple[float, torch.Tensor]:
        pos = self.data_plane.positions
        min_coords, _ = torch.min(pos, dim=0)
        max_coords, _ = torch.max(pos, dim=0)
        extents = torch.clamp(max_coords - min_coords, min=1e-5)
        vol = float(torch.prod(extents).item())
        bbox = torch.stack([min_coords, max_coords])
        return vol, bbox

    def _calculate_shear_stress(self) -> float:
        vel = self.data_plane.velocities
        if vel.numel() == 0 or len(vel) < 2:
            return 0.0
        # Variance of velocity vectors represents internal shear stress
        vel_mean = torch.mean(vel, dim=0)
        shear = float(torch.mean(torch.sum((vel - vel_mean) ** 2, dim=-1)).item())
        return shear


class CompressedCausalSeal:
    """
    [Compressed Causal Seal (입체적 인장)]
    Compressed macro-representation of an EmergentCausalVolume.
    Supports Phase Re-ignition (Decompression): Applies voltage / excitation energy
    to re-ignite ComponentDataPlane and TransformativeGear into an active EmergentCausalVolume.
    """
    def __init__(
        self,
        seal_id: str,
        macro_vector: torch.Tensor,
        stored_gear_params: Dict[str, float],
        reference_volume: float
    ):
        self.seal_id = seal_id
        self.macro_vector = macro_vector  # Compact latent tensor
        self.stored_gear_params = stored_gear_params
        self.reference_volume = reference_volume

    @classmethod
    def compress_volume(
        cls,
        seal_id: str,
        volume: EmergentCausalVolume
    ) -> 'CompressedCausalSeal':
        """
        Compresses an EmergentCausalVolume into a CompressedCausalSeal.
        """
        pos = volume.data_plane.positions
        vel = volume.data_plane.velocities
        chroma = volume.data_plane.chromatic_spectrum

        # Compact latent representation: [mean_pos, mean_vel, mean_chroma, volume, T_cog]
        mean_pos = torch.mean(pos, dim=0)
        mean_vel = torch.mean(vel, dim=0)
        mean_chroma = torch.mean(chroma, dim=0)

        macro_vector = torch.cat([
            mean_pos,
            mean_vel,
            mean_chroma,
            torch.tensor([volume.volume_metric, volume.gear.cognitive_temperature], device=pos.device)
        ])

        stored_params = {
            "cognitive_temperature": volume.gear.cognitive_temperature,
            "attention_density": volume.gear.attention_density,
            "degrees_of_freedom": volume.gear.degrees_of_freedom,
            "num_nodes": volume.data_plane.num_nodes,
            "dim": volume.data_plane.dim
        }

        return cls(
            seal_id=seal_id,
            macro_vector=macro_vector,
            stored_gear_params=stored_params,
            reference_volume=volume.volume_metric
        )

    def decompress_reignite(
        self,
        excitation_voltage: float = 1.0,
        device: str = "cpu"
    ) -> Tuple[ComponentDataPlane, TransformativeGear, EmergentCausalVolume]:
        """
        [Phase Re-ignition (위상 재점화)]
        Unseals the compressed seal by applying excitation voltage.
        Re-ignites micro ComponentDataPlane and dynamic TransformativeGear.
        """
        dim = int(self.stored_gear_params["dim"])
        num_nodes = int(self.stored_gear_params["num_nodes"])

        mean_pos = self.macro_vector[:dim]
        mean_vel = self.macro_vector[dim:2*dim]
        mean_chroma = self.macro_vector[2*dim:2*dim+3]

        # Re-generate particle positions with spatial spread scaled by excitation voltage
        spread = 0.5 * excitation_voltage
        pos_noise = torch.randn((num_nodes, dim), device=device) * spread
        reignited_pos = mean_pos.unsqueeze(0) + pos_noise

        vel_noise = torch.randn((num_nodes, dim), device=device) * (spread * 0.5)
        reignited_vel = mean_vel.unsqueeze(0) + vel_noise

        chroma = mean_chroma.unsqueeze(0).repeat(num_nodes, 1)

        data_plane = ComponentDataPlane(
            positions=reignited_pos,
            velocities=reignited_vel,
            chromatic_spectrum=chroma,
            device=device
        )

        # Re-ignited cognitive temperature reflects excitation_voltage
        reignited_temp = self.stored_gear_params["cognitive_temperature"] * excitation_voltage
        gear = TransformativeGear(
            dim=dim,
            cognitive_temperature=reignited_temp,
            attention_density=self.stored_gear_params["attention_density"],
            degrees_of_freedom=self.stored_gear_params["degrees_of_freedom"],
            device=device
        )

        volume = EmergentCausalVolume(data_plane=data_plane, gear=gear)
        return data_plane, gear, volume


class TransformativeGearEngine:
    """
    [Transformative Gear Engine]
    Full Meta-Ontological Engine managing ComponentDataPlane, TransformativeGear,
    EmergentCausalVolume, and CompressedCausalSeals.
    Features cognitive thermodynamic feedback loop driven by prediction error delta (Delta),
    angular vector mismatch, and attention density.
    """
    def __init__(self, dim: int = 3, device: str = "cpu"):
        self.dim = dim
        self.device = device
        self.gear = TransformativeGear(dim=dim, cognitive_temperature=1.0, device=device)
        self.seals: Dict[str, CompressedCausalSeal] = {}

    def step(
        self,
        data_plane: ComponentDataPlane,
        observed_target_plane: Optional[ComponentDataPlane] = None,
        input_torque_vector: Optional[torch.Tensor] = None,
        dt: float = 0.1
    ) -> Dict[str, Any]:
        """
        Executes one full step of the engine:
        1. Forward gear transformation: DataPlane -> EmergentVolume
        2. Prediction error calculation vs Observed Target
        3. Cognitive Temperature (T_cog) & Phase Transition update
        4. Isomorphic feedback adjustment
        """
        # 1. Forward transformation
        next_data_plane = self.gear.transform_data_plane(data_plane, dt=dt)
        emergent_volume = EmergentCausalVolume(next_data_plane, self.gear)

        prediction_error = 0.0
        angular_mismatch = 0.0

        # 2. Prediction error & Directional resistance calculation
        if observed_target_plane is not None:
            # Position prediction error Delta
            diff = next_data_plane.positions - observed_target_plane.positions.to(self.device)
            prediction_error = float(torch.mean(torch.norm(diff, dim=-1)).item())

            # Angular directional mismatch (inertia vs input)
            curr_mom = torch.mean(next_data_plane.velocities, dim=0)
            target_mom = torch.mean(observed_target_plane.velocities.to(self.device), dim=0)

            norm_c = torch.norm(curr_mom)
            norm_t = torch.norm(target_mom)
            if norm_c > 1e-5 and norm_t > 1e-5:
                cos_sim = torch.dot(curr_mom, target_mom) / (norm_c * norm_t)
                cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
                angular_mismatch = float((1.0 - cos_sim).item())

        # Torque vector impact
        if input_torque_vector is not None:
            torque_impact = float(torch.norm(input_torque_vector).item())
            angular_mismatch += torque_impact * 0.1

        # 3. Cognitive Temperature Update (Metabolic / Friction Heating)
        # Heat = Prediction Error + Angular Mismatch Friction
        thermal_friction = prediction_error * 0.5 + angular_mismatch * 1.2
        cooling_factor = 0.92  # Relaxation back to equilibrium

        new_temp = (self.gear.cognitive_temperature * cooling_factor) + thermal_friction
        self.gear.cognitive_temperature = max(0.1, min(new_temp, 5.0))

        # Update degrees of freedom based on temperature
        if self.gear.cognitive_temperature > 2.0:
            self.gear.degrees_of_freedom = min(1.0, self.gear.degrees_of_freedom + 0.1)
        elif self.gear.cognitive_temperature < 0.5:
            self.gear.degrees_of_freedom = max(0.0, self.gear.degrees_of_freedom - 0.1)
        else:
            self.gear.degrees_of_freedom = 0.5

        phase_state = self.gear.update_phase_state()

        return {
            "next_data_plane": next_data_plane,
            "emergent_volume": emergent_volume,
            "volume_metric": emergent_volume.volume_metric,
            "shear_stress": emergent_volume.shear_stress,
            "prediction_error": prediction_error,
            "angular_mismatch": angular_mismatch,
            "cognitive_temperature": self.gear.cognitive_temperature,
            "degrees_of_freedom": self.gear.degrees_of_freedom,
            "phase_state": phase_state
        }

    def seal_concept(self, seal_id: str, volume: EmergentCausalVolume) -> CompressedCausalSeal:
        seal = CompressedCausalSeal.compress_volume(seal_id, volume)
        self.seals[seal_id] = seal
        return seal

    def reignite_seal(
        self,
        seal_id: str,
        excitation_voltage: float = 1.0
    ) -> Tuple[ComponentDataPlane, TransformativeGear, EmergentCausalVolume]:
        if seal_id not in self.seals:
            raise KeyError(f"Seal ID '{seal_id}' not found in engine seals repository.")
        seal = self.seals[seal_id]
        return seal.decompress_reignite(excitation_voltage=excitation_voltage, device=self.device)
