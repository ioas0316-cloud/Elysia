"""
Role Field Synergy Engine (역할군 삼각 분업장)

Translates WoW Tank/Dealer/Healer raid triad into continuous causal field dynamics.

External collision Ω_ext is distributed across 3 role fields:
  - Mitigation (Tank): Absorbs and dampens friction stress via exponential damping
  - Action (Dealer): Drives system state toward target phase via gradient flow
  - Restoration (Healer): Corrects decoherence via Kuramoto-style phase locking

Energy conservation law: E_tank + E_deal + E_heal = E_total at all times.

Continuities maintained:
  - Relationship: 3 fields form coupled potential triad, not independent modules
  - Connectivity: Energy flows between fields through distribution weights (continuous)
  - Mobility: Shock energy has momentum that transfers across fields
  - Informational Continuity: Distribution weights evolve smoothly between cycles
  - Chromatic: Tank=Blue(Order/Resistance), Dealer=Red(Flux), Healer=Yellow→Blue(Entropy→Order)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


class RoleFieldType(Enum):
    MITIGATION = auto()   # Tank - absorbs external friction stress
    ACTION = auto()       # Dealer - drives state toward target phase
    RESTORATION = auto()  # Healer - corrects decoherence and maintains scale alignment


@dataclass
class RoleField:
    """A single role field within the Tank/Dealer/Healer triad.
    All numeric parameters are mutable initial values that evolve dynamically."""
    field_type: RoleFieldType
    energy: float                          # Current energy level
    absorption_capacity: float             # Tank: how much shock it can absorb
    action_rate: float                     # Dealer: speed of state transition
    restoration_bandwidth: float           # Healer: decoherence correction bandwidth
    damping_coefficient: float             # Continuous damping factor
    accumulated_strain: float = 0.0        # Total strain absorbed over lifetime
    efficiency: float = 1.0               # Evolves based on performance history
    cycle_count: int = 0                  # Number of synergy cycles participated in
    chromatic_signature: np.ndarray = None  # [Red, Blue, Yellow] role color

    def __post_init__(self):
        if self.chromatic_signature is None:
            if self.field_type == RoleFieldType.MITIGATION:
                self.chromatic_signature = np.array([0.1, 0.7, 0.2], dtype=np.float32)  # Blue-dominant
            elif self.field_type == RoleFieldType.ACTION:
                self.chromatic_signature = np.array([0.7, 0.1, 0.2], dtype=np.float32)  # Red-dominant
            else:
                self.chromatic_signature = np.array([0.2, 0.4, 0.4], dtype=np.float32)  # Blue+Yellow blend


@dataclass
class SynergyCycleReport:
    """Report from a single synergy cycle execution."""
    shock_energy_total: float
    energy_distribution: Dict[RoleFieldType, float]
    residual_after_mitigation: float
    state_displacement: float
    decoherence_corrected: float
    energy_conservation_error: float
    distribution_weights: torch.Tensor
    role_field_states: Dict[RoleFieldType, Dict[str, float]]


class RoleFieldSynergyEngine(nn.Module):
    """
    [역할군 삼각 분업장 (Role Field Synergy Engine)]

    외부 충돌(Ω_ext)에 대응하는 Tank/Dealer/Healer 역학장 시스템.
    에너지 보존 법칙 하에서 3개 역할장이 동시 갱신되며,
    분배 비율은 고정값이 아닌 가변적 초기값으로 매 사이클마다 적응합니다.

    Energy conservation: E_tank + E_deal + E_heal = E_total (enforced every cycle)
    """

    def __init__(
        self,
        dimension: int = 64,
        initial_total_energy: float = 100.0,
        dtype=torch.float32
    ):
        super().__init__()
        self.dimension = dimension
        self.dtype = dtype

        # Mutable initial: energy distribution starts at 1/3:1/3:1/3 but evolves
        self.distribution_weights = nn.Parameter(
            torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=dtype),
            requires_grad=False
        )

        # Total energy is conserved across all 3 fields
        self.total_energy = initial_total_energy

        # Role field alignment tensors (learn which shock directions each role handles best)
        self.mitigation_alignment = nn.Parameter(torch.randn(dimension, dtype=dtype) * 0.1)
        self.action_alignment = nn.Parameter(torch.randn(dimension, dtype=dtype) * 0.1)
        self.restoration_alignment = nn.Parameter(torch.randn(dimension, dtype=dtype) * 0.1)

        # Initialize 3 role fields with mutable initial values
        self.role_fields: Dict[RoleFieldType, RoleField] = {
            RoleFieldType.MITIGATION: RoleField(
                field_type=RoleFieldType.MITIGATION,
                energy=initial_total_energy / 3.0,
                absorption_capacity=10.0,    # Mutable initial
                action_rate=0.0,
                restoration_bandwidth=0.0,
                damping_coefficient=0.85,    # Mutable initial
            ),
            RoleFieldType.ACTION: RoleField(
                field_type=RoleFieldType.ACTION,
                energy=initial_total_energy / 3.0,
                absorption_capacity=0.0,
                action_rate=0.1,             # Mutable initial
                restoration_bandwidth=0.0,
                damping_coefficient=0.5,     # Mutable initial
            ),
            RoleFieldType.RESTORATION: RoleField(
                field_type=RoleFieldType.RESTORATION,
                energy=initial_total_energy / 3.0,
                absorption_capacity=0.0,
                action_rate=0.0,
                restoration_bandwidth=0.3,   # Mutable initial
                damping_coefficient=0.7,     # Mutable initial
            ),
        }

        # Adaptation rate for distribution weight evolution (mutable initial)
        self.weight_adaptation_rate = 0.05

    def distribute_shock(
        self,
        omega_ext: torch.Tensor
    ) -> Dict[RoleFieldType, torch.Tensor]:
        """
        Distributes external collision energy across 3 role fields.

        Distribution is NOT uniform 1/3 — it's driven by:
        1. Current distribution_weights (evolving)
        2. Shock direction alignment with each field's tensor

        Returns per-field shock energy vectors.
        """
        omega_ext = omega_ext.to(self.dtype)
        shock_magnitude = torch.norm(omega_ext)

        # Compute alignment scores: how well each field is oriented to handle this shock
        align_mit = torch.abs(torch.dot(omega_ext / (shock_magnitude + 1e-8), self.mitigation_alignment))
        align_act = torch.abs(torch.dot(omega_ext / (shock_magnitude + 1e-8), self.action_alignment))
        align_res = torch.abs(torch.dot(omega_ext / (shock_magnitude + 1e-8), self.restoration_alignment))

        # Combine base weights with alignment (softmax for valid distribution)
        raw_weights = self.distribution_weights + 0.1 * torch.stack([align_mit, align_act, align_res])
        effective_weights = torch.softmax(raw_weights, dim=0)

        # Distribute shock energy
        distributed: Dict[RoleFieldType, torch.Tensor] = {}
        for i, role_type in enumerate([RoleFieldType.MITIGATION, RoleFieldType.ACTION, RoleFieldType.RESTORATION]):
            distributed[role_type] = omega_ext * effective_weights[i]

        return distributed

    def mitigation_step(
        self,
        shock_energy: torch.Tensor
    ) -> torch.Tensor:
        """
        Tank (완충장): Absorbs and dampens friction stress.

        Uses exponential damping: E_residual = E_shock * exp(-damping * absorption_capacity)
        Returns residual unabsorbed energy that passes through to the rest of the system.
        """
        tank = self.role_fields[RoleFieldType.MITIGATION]
        shock_mag = float(torch.norm(shock_energy).item())

        # Exponential damping — higher capacity + higher damping = more absorption
        absorption_factor = 1.0 - float(np.exp(-tank.damping_coefficient * tank.absorption_capacity))
        absorbed = shock_energy * absorption_factor * tank.efficiency

        residual = shock_energy - absorbed

        # Update tank state
        tank.accumulated_strain += shock_mag * absorption_factor
        tank.energy -= shock_mag * absorption_factor * 0.01  # Small energy cost

        # Capacity adapts: absorbing more strain builds capacity over time
        tank.absorption_capacity += shock_mag * 0.001  # Mutable evolution

        # Efficiency degrades slightly under sustained strain, recovers when idle
        if shock_mag > 5.0:
            tank.efficiency *= 0.999
        else:
            tank.efficiency = min(1.0, tank.efficiency * 1.001)

        tank.cycle_count += 1
        return residual

    def action_step(
        self,
        current_state: torch.Tensor,
        target_phase: torch.Tensor
    ) -> torch.Tensor:
        """
        Dealer (변환장): Drives state toward target phase via gradient flow.

        Uses gradient descent toward target: new_state = state + action_rate * (target - state)
        The action_rate is a mutable initial that evolves based on distance-to-target history.
        """
        dealer = self.role_fields[RoleFieldType.ACTION]

        # Direction toward target
        direction = target_phase - current_state
        distance = torch.norm(direction)

        if distance < 1e-8:
            return current_state

        # Normalized step with adaptive rate
        step = direction * dealer.action_rate * dealer.efficiency

        # Prevent overshooting
        step_magnitude = torch.norm(step)
        if step_magnitude > distance:
            step = direction  # Snap to target

        new_state = current_state + step

        # Adapt action_rate: increase when making progress, decrease when overshooting
        dealer.action_rate *= (1.0 + 0.01 * min(1.0, float(distance.item())))

        # Energy cost proportional to work done (F · d)
        work = float(torch.norm(step).item())
        dealer.energy -= work * 0.01

        dealer.cycle_count += 1
        return new_state

    def restoration_step(
        self,
        state: torch.Tensor,
        reference_phase: torch.Tensor,
        decoherence_metric: float
    ) -> torch.Tensor:
        """
        Healer (복원장): Corrects phase decoherence via Kuramoto-style phase locking.

        Applies phase coupling: d(phi)/dt = bandwidth * sin(phi_ref - phi_state)
        Only corrects within the restoration_bandwidth window.
        """
        healer = self.role_fields[RoleFieldType.RESTORATION]

        if decoherence_metric < 1e-8:
            healer.cycle_count += 1
            return state

        # Phase extraction (using atan2 on consecutive pairs)
        phase_state = torch.atan2(state, torch.roll(state, 1) + 1e-8)
        phase_ref = torch.atan2(reference_phase, torch.roll(reference_phase, 1) + 1e-8)

        # Kuramoto coupling: correction proportional to sin(phase difference)
        phase_diff = phase_ref - phase_state
        kuramoto_correction = healer.restoration_bandwidth * torch.sin(phase_diff) * healer.efficiency

        # Apply correction only where decoherence exceeds threshold
        correction_mask = (torch.abs(phase_diff) > 0.01).float()
        kuramoto_correction = kuramoto_correction * correction_mask

        # Reconstruct corrected state
        corrected_magnitude = torch.norm(state, dim=-1, keepdim=True) if state.dim() > 0 else torch.norm(state)
        corrected_phase = phase_state + kuramoto_correction
        restored = corrected_magnitude * torch.cos(corrected_phase)

        # Blend: don't fully replace, smoothly interpolate
        blend_factor = min(1.0, decoherence_metric * healer.restoration_bandwidth)
        result = (1.0 - blend_factor) * state + blend_factor * restored

        # Energy cost
        correction_work = float(torch.norm(kuramoto_correction).item())
        healer.energy -= correction_work * 0.01

        # Bandwidth adapts: successful corrections increase bandwidth
        healer.restoration_bandwidth += 0.001 * decoherence_metric

        healer.cycle_count += 1
        return result

    def _enforce_energy_conservation(self):
        """
        Enforces E_tank + E_deal + E_heal = E_total.
        Redistributes energy to maintain conservation while respecting current weights.
        """
        current_sum = sum(rf.energy for rf in self.role_fields.values())
        if abs(current_sum) < 1e-8:
            # Fallback: redistribute equally
            for rf in self.role_fields.values():
                rf.energy = self.total_energy / 3.0
            return

        # Scale all energies proportionally to conserve total
        scale_factor = self.total_energy / current_sum
        for rf in self.role_fields.values():
            rf.energy *= scale_factor

    def adapt_distribution_weights(self, cycle_report: SynergyCycleReport):
        """
        Evolves the distribution ratio based on field utilization.

        Over-utilized fields get more energy; under-utilized get less.
        Uses softmax normalization to maintain valid probability distribution.
        """
        # Compute utilization: strain absorbed / energy allocated
        utilizations = []
        for role_type in [RoleFieldType.MITIGATION, RoleFieldType.ACTION, RoleFieldType.RESTORATION]:
            rf = self.role_fields[role_type]
            allocated = cycle_report.energy_distribution.get(role_type, 1e-8)
            utilization = rf.accumulated_strain / (allocated + 1e-8)
            utilizations.append(utilization)

        util_tensor = torch.tensor(utilizations, dtype=self.dtype)

        # Shift weights toward more utilized fields
        with torch.no_grad():
            self.distribution_weights.data += self.weight_adaptation_rate * util_tensor
            # Re-normalize via softmax
            self.distribution_weights.data = torch.softmax(self.distribution_weights.data, dim=0)

    def synergy_cycle(
        self,
        omega_ext: torch.Tensor,
        current_state: torch.Tensor,
        target_phase: torch.Tensor
    ) -> SynergyCycleReport:
        """
        Full synergy cycle: distribute → mitigate → act → restore.

        1. Distribute external shock across 3 role fields
        2. Tank absorbs/dampens the shock
        3. Dealer drives state toward target
        4. Healer corrects any decoherence
        5. Enforce energy conservation
        6. Adapt distribution weights

        Returns comprehensive cycle report.
        """
        omega_ext = omega_ext.to(self.dtype)
        current_state = current_state.to(self.dtype)
        target_phase = target_phase.to(self.dtype)

        shock_total = float(torch.norm(omega_ext).item())

        # 1. Distribute
        distributed = self.distribute_shock(omega_ext)
        energy_dist = {k: float(torch.norm(v).item()) for k, v in distributed.items()}

        # 2. Mitigate (Tank absorbs its portion)
        residual = self.mitigation_step(distributed[RoleFieldType.MITIGATION])
        residual_mag = float(torch.norm(residual).item())

        # Residual shock perturbs the current state
        perturbed_state = current_state + residual * 0.1

        # 3. Act (Dealer drives toward target)
        pre_action_dist = float(torch.norm(perturbed_state - target_phase).item())
        acted_state = self.action_step(perturbed_state, target_phase)
        post_action_dist = float(torch.norm(acted_state - target_phase).item())
        displacement = pre_action_dist - post_action_dist

        # 4. Restore (Healer corrects decoherence)
        decoherence = float(torch.norm(acted_state - target_phase).item())
        restored_state = self.restoration_step(acted_state, target_phase, decoherence)
        decoherence_corrected = decoherence - float(torch.norm(restored_state - target_phase).item())

        # 5. Enforce energy conservation
        self._enforce_energy_conservation()

        # 6. Build report
        report = SynergyCycleReport(
            shock_energy_total=shock_total,
            energy_distribution=energy_dist,
            residual_after_mitigation=residual_mag,
            state_displacement=displacement,
            decoherence_corrected=max(0.0, decoherence_corrected),
            energy_conservation_error=abs(
                sum(rf.energy for rf in self.role_fields.values()) - self.total_energy
            ),
            distribution_weights=self.distribution_weights.clone(),
            role_field_states={
                rt: {
                    "energy": rf.energy,
                    "efficiency": rf.efficiency,
                    "strain": rf.accumulated_strain,
                    "cycles": rf.cycle_count,
                }
                for rt, rf in self.role_fields.items()
            }
        )

        # 7. Adapt weights for next cycle
        self.adapt_distribution_weights(report)

        return report

    def get_field_chromatic_state(self) -> Dict[RoleFieldType, np.ndarray]:
        """Returns the chromatic signature of each role field for visualization."""
        return {rt: rf.chromatic_signature.copy() for rt, rf in self.role_fields.items()}
