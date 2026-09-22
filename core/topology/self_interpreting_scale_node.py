"""
Self-Interpreting Scale Node & Causal Schema Protocol Module.

Implements the scale coupling equation:
    Structure(n+1) = f_coupling(Structure(n))
along with reversible phase transitions (Gas <-> Liquid <-> Ice),
spontaneous coupling via Variational Free Energy minimization,
self-interpreting Ice Block Causal Schema Headers with inverse protocols f_{coupling}^{-1},
and Limit Map registration for reconstruction anomalies.
"""

from enum import Enum
import math
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np


class PhaseState(Enum):
    GAS = "gas"         # High entropy, dispersed noise, uncoupled
    LIQUID = "liquid"   # Active inference, dynamic variable coupling, searching for equilibrium
    ICE = "ice"         # Crystallized macro structure (n+1), FLOPs -> 0, zero-computation invariant


class CausalSchemaHeader:
    """
    Self-Interpreting Ice Block Causal Schema Header.

    Contains the inverse protocol f_{coupling}^{-1} and metadata allowing
    O(1) self-deconstruction without requiring external parsers or decoders.
    """
    def __init__(
        self,
        scale: int,
        coupling_params: Dict[str, float],
        phase_offset: float,
        normal_alignment: float,
        reconstruction_error: float = 0.0,
        sub_node_ids: Optional[List[str]] = None,
        structural_resistance: float = 0.0,
        relationship_density: float = 1.0,
    ):
        self.scale = scale  # Scale level n
        self.coupling_params = coupling_params  # alpha, beta, gamma
        self.phase_offset = phase_offset  # Delta theta_ij at coupling moment
        self.normal_alignment = normal_alignment  # n_i . n_j
        self.reconstruction_error = reconstruction_error  # delta_recon
        self.sub_node_ids = sub_node_ids or []
        self.structural_resistance = structural_resistance
        self.relationship_density = relationship_density

    def inverse_protocol(self) -> Dict[str, Any]:
        """
        Returns the inverse protocol f_{coupling}^{-1} instructions for self-deconstruction.
        """
        return {
            "action": "deconstruct",
            "scale_from": self.scale,
            "scale_to": self.scale - 1,
            "phase_offset": self.phase_offset,
            "normal_alignment": self.normal_alignment,
            "coupling_params": self.coupling_params,
            "sub_node_ids": self.sub_node_ids,
            "reconstruction_error": self.reconstruction_error
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scale": self.scale,
            "coupling_params": self.coupling_params,
            "phase_offset": self.phase_offset,
            "normal_alignment": self.normal_alignment,
            "reconstruction_error": self.reconstruction_error,
            "sub_node_ids": self.sub_node_ids,
            "structural_resistance": self.structural_resistance,
            "relationship_density": self.relationship_density,
            "inverse_protocol": self.inverse_protocol(),
        }


class LimitMap:
    """
    Registry for unexplained variant phenomena where reconstruction error delta_recon
    exceeds epsilon_limit despite low variational free energy during coupling.
    """
    def __init__(self, epsilon_limit: float = 0.05):
        self.epsilon_limit = epsilon_limit
        self.records: List[Dict[str, Any]] = []

    def register_anomaly(
        self,
        node_id: str,
        scale: int,
        delta_recon: float,
        free_energy: float,
        details: Dict[str, Any]
    ):
        record = {
            "node_id": node_id,
            "scale": scale,
            "delta_recon": delta_recon,
            "free_energy": free_energy,
            "epsilon_limit": self.epsilon_limit,
            "details": details
        }
        self.records.append(record)
        return record


class SelfInterpretingScaleNode:
    """
    Scale Coupling Node representing Structure(n) or Structure(n+1).

    Equipped with a 2D Level-Set boundary field Phi(x, y), phase angle theta,
    chromatic spectrum, and local temperature T_local.
    """
    def __init__(
        self,
        node_id: str,
        scale: int = 0,
        center: Tuple[float, float] = (0.0, 0.0),
        radius: float = 1.0,
        phase_angle: float = 0.0,
        chromatic_spectrum: Optional[np.ndarray] = None,
        local_temperature: float = 1.0,
        phase_state: PhaseState = PhaseState.GAS
    ):
        self.node_id = node_id
        self.scale = scale
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.phase_angle = float(phase_angle)  # theta_i
        self.chromatic_spectrum = (
            chromatic_spectrum if chromatic_spectrum is not None
            else np.array([1.0, 0.0, 0.0], dtype=np.float64)  # RGB / Flux, Order, Entropy
        )
        self.local_temperature = float(local_temperature)  # T_local
        self.phase_state = phase_state

        # Parent / Child hierarchy
        self.sub_nodes: List['SelfInterpretingScaleNode'] = []
        self.header: Optional[CausalSchemaHeader] = None

        # Flops counter (0 when in ICE state)
        self.flops_executed = 0

    def level_set(self, point: Tuple[float, float]) -> float:
        """
        Implicit Level-Set Function Phi(x, y) = distance(point, center) - radius.
        Phi < 0: inside boundary
        Phi = 0: boundary
        Phi > 0: outside boundary
        """
        p = np.array(point, dtype=np.float64)
        return float(np.linalg.norm(p - self.center) - self.radius)

    def boundary_normal(self, point: Tuple[float, float]) -> np.ndarray:
        """
        Normal vector n_i = grad(Phi_i) / ||grad(Phi_i)||.
        """
        p = np.array(point, dtype=np.float64)
        diff = p - self.center
        norm = np.linalg.norm(diff)
        if norm < 1e-9:
            return np.array([1.0, 0.0], dtype=np.float64)
        return diff / norm

    def calculate_free_energy(
        self,
        other: 'SelfInterpretingScaleNode',
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5
    ) -> Tuple[float, float, float]:
        """
        Calculates Variational Free Energy F_{ij} between this node and another node:
        F_{ij} = int_{Gamma_{ij}} [ alpha*(1 - n_i . n_j) + beta*(1 - cos(Delta theta_{ij})) ] d_sigma + gamma * T_{local}

        Returns:
            (F_{ij}, normal_alignment, phase_offset)
        """
        center_dist = np.linalg.norm(self.center - other.center)
        overlap_dist = (self.radius + other.radius) - center_dist

        if overlap_dist < -1e-5:
            return float('inf'), 0.0, math.pi

        direction = (other.center - self.center)
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-9:
            dir_unit = np.array([1.0, 0.0])
        else:
            dir_unit = direction / dir_norm

        contact_point = self.center + dir_unit * self.radius

        n_i = self.boundary_normal(contact_point)
        n_j = -other.boundary_normal(contact_point)

        normal_alignment = float(np.dot(n_i, n_j))

        phase_offset = abs(self.phase_angle - other.phase_angle) % (2 * math.pi)

        spatial_term = alpha * (1.0 - normal_alignment)
        temporal_term = beta * (1.0 - math.cos(phase_offset))

        avg_temperature = 0.5 * (self.local_temperature + other.local_temperature)
        thermal_term = gamma * avg_temperature

        overlap_measure = max(0.5, float(overlap_dist + 0.5))
        F_ij = (spatial_term + temporal_term) * overlap_measure + thermal_term

        if self.phase_state != PhaseState.ICE:
            self.flops_executed += 10
        if other.phase_state != PhaseState.ICE:
            other.flops_executed += 10

        return F_ij, normal_alignment, phase_offset

    def attempt_spontaneous_coupling(
        self,
        other: 'SelfInterpretingScaleNode',
        f_threshold: float = 2.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5,
        epsilon_limit: float = 0.05,
        limit_map: Optional[LimitMap] = None,
        artificial_reconstruction_error: Optional[float] = None
    ) -> Optional['SelfInterpretingScaleNode']:
        """
        Attempts spontaneous coupling of two Structure(n) nodes into a Structure(n+1) node.

        If F_{ij} < F_{threshold}:
            Fuses into parent node Structure(n+1) with levelset Phi_{n+1} = min(Phi_i, Phi_j).
            State transitions to ICE (crystallized macro structure).
            Generates embedded CausalSchemaHeader with inverse protocol f_{coupling}^{-1}.
            Verifies reconstruction error delta_recon. If > epsilon_limit, registers in LimitMap.
        Else:
            Nodes remain uncoupled or disperse as GAS if temperature is high.
        """
        F_ij, normal_alignment, phase_offset = self.calculate_free_energy(
            other, alpha=alpha, beta=beta, gamma=gamma
        )

        if F_ij >= f_threshold:
            if self.local_temperature > 2.0:
                self.phase_state = PhaseState.GAS
            if other.local_temperature > 2.0:
                other.phase_state = PhaseState.GAS
            return None

        self.phase_state = PhaseState.ICE
        other.phase_state = PhaseState.ICE

        parent_scale = max(self.scale, other.scale) + 1
        new_center = 0.5 * (self.center + other.center)
        new_radius = max(self.radius, other.radius) + 0.5 * np.linalg.norm(self.center - other.center)
        new_phase_angle = 0.5 * (self.phase_angle + other.phase_angle)
        new_chromatic = 0.5 * (self.chromatic_spectrum + other.chromatic_spectrum)
        new_temp = min(self.local_temperature, other.local_temperature) * 0.5

        parent_node = SelfInterpretingScaleNode(
            node_id=f"Structure_scale_{parent_scale}_{self.node_id}_{other.node_id}",
            scale=parent_scale,
            center=new_center,
            radius=new_radius,
            phase_angle=new_phase_angle,
            chromatic_spectrum=new_chromatic,
            local_temperature=new_temp,
            phase_state=PhaseState.ICE
        )

        parent_node.sub_nodes = [self, other]

        if artificial_reconstruction_error is not None:
            delta_recon = artificial_reconstruction_error
        else:
            delta_recon = self._calculate_reconstruction_error(parent_node, [self, other])

        coupling_params = {"alpha": alpha, "beta": beta, "gamma": gamma, "f_threshold": f_threshold}
        header = CausalSchemaHeader(
            scale=parent_scale,
            coupling_params=coupling_params,
            phase_offset=phase_offset,
            normal_alignment=normal_alignment,
            reconstruction_error=delta_recon,
            sub_node_ids=[self.node_id, other.node_id],
            structural_resistance=F_ij,
            relationship_density=1.0 / (1.0 + delta_recon)
        )
        parent_node.header = header

        if delta_recon > epsilon_limit and limit_map is not None:
            limit_map.register_anomaly(
                node_id=parent_node.node_id,
                scale=parent_scale,
                delta_recon=delta_recon,
                free_energy=F_ij,
                details={
                    "header": header.to_dict(),
                    "sub_nodes": [self.node_id, other.node_id]
                }
            )

        return parent_node

    def self_deconstruct(self) -> Tuple[List['SelfInterpretingScaleNode'], Dict[str, Any]]:
        """
        O(1) Self-Deconstruction using embedded CausalSchemaHeader's inverse protocol f_{coupling}^{-1}.
        No external parser required.
        """
        if not self.header or not self.sub_nodes:
            return [self], {"action": "atomic_no_op", "scale": self.scale}

        inv_protocol = self.header.inverse_protocol()

        restored_sub_nodes = []
        for sub in self.sub_nodes:
            sub.phase_state = PhaseState.LIQUID
            restored_sub_nodes.append(sub)

        deconstruction_log = {
            "status": "success",
            "inverse_protocol_applied": inv_protocol,
            "restored_count": len(restored_sub_nodes)
        }

        self.phase_state = PhaseState.LIQUID
        return restored_sub_nodes, deconstruction_log

    def apply_thermal_perturbation(
        self,
        temperature_delta: float,
        noise_amplitude: float = 0.0
    ):
        """
        Applies external thermal or perturbation noise.
        Triggers reversible phase transition:
        - ICE -> LIQUID -> GAS under high temperature/noise.
        - GAS -> LIQUID -> ICE under cooling.
        """
        self.local_temperature = max(0.0, self.local_temperature + temperature_delta)

        if noise_amplitude > 0.0:
            self.phase_angle = (self.phase_angle + np.random.normal(0, noise_amplitude)) % (2 * math.pi)
            self.center += np.random.normal(0, noise_amplitude, size=2)

        if self.local_temperature > 3.0 or noise_amplitude > 1.0:
            if self.phase_state == PhaseState.ICE and self.sub_nodes:
                self.phase_state = PhaseState.LIQUID
                for sub in self.sub_nodes:
                    sub.phase_state = PhaseState.LIQUID
                    sub.local_temperature = self.local_temperature
            else:
                self.phase_state = PhaseState.GAS
        elif self.local_temperature < 0.5 and self.phase_state != PhaseState.GAS:
            if self.header is not None or self.sub_nodes:
                self.phase_state = PhaseState.ICE
                for sub in self.sub_nodes:
                    sub.phase_state = PhaseState.ICE
                    sub.local_temperature = self.local_temperature

    def _calculate_reconstruction_error(
        self,
        parent: 'SelfInterpretingScaleNode',
        sub_nodes: List['SelfInterpretingScaleNode']
    ) -> float:
        """
        Computes reconstruction error delta_recon = ||Structure(n+1) - f_{coupling}(Structure(n))||
        """
        parent_area = math.pi * (parent.radius ** 2)
        sub_area_sum = sum(math.pi * (s.radius ** 2) for s in sub_nodes)
        area_diff = abs(parent_area - sub_area_sum) / max(1e-5, parent_area)

        avg_sub_phase = float(np.mean([s.phase_angle for s in sub_nodes]))
        phase_diff = abs(parent.phase_angle - avg_sub_phase) / (2 * math.pi)

        delta_recon = float(0.7 * area_diff + 0.3 * phase_diff)
        return delta_recon
