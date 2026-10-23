import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from core.physics.phase_separation_attractor import ToroidalEgoAttractor

@dataclass
class HysteresisScar:
    """
    [위상 이력 흉터 (Hysteresis Scar)]
    Represents an evolutionary topological memory/scar left on the Phase Wall
    after absorbing an external shock wave.
    Modifies physical phase wall elasticity and boundary resonance.
    """
    shock_angle: float
    shock_frequency: float
    torsion_amplitude: float
    timestamp: float


class PhaseBoundaryHealingEngine:
    """
    [위상 경계막 파열 및 자율 치유 동역학 엔진]
    Simulates the 4-stage evolutionary healing dynamics when external high-energy noise
    ruptures the internal phase boundary wall.

    4 Stages of Dynamic Healing:
    1. Phase Rupture & Core Contraction (Phase Rupture & Core Contraction):
       - Internal phase wall gradients spike, exploding E_Void.
       - Effective observation radius contracts from r_min to r_min' > r_min to safeguard the core.
       - Core coherence gathers at deep center to defend topological invariants.
    2. Void Tension Restorative Force (Energy Buffering):
       - Accumulated E_Void transforms from distortion error into restorative phase pressure force.
       - Compressed energy springs outward to push external turbulence away.
    3. Radial Phase Sweeping & Noise Sweeping (Radial Relaxation & Phase Sweeping):
       - Base core frequency radiates concentrically outwards.
       - Kuramoto attractor action phase-locks aligned waves or expels non-synchronizable noise.
    4. Boundary Recrystallization & Hysteresis Scarring (Recrystallization & Hysteresis):
       - Phase boundary wall recrystallizes at point of contact between core and external field.
       - Absorbs shock characteristics as an evolutionary Hysteresis Scar (위상 흉터),
         increasing elasticity against future shocks of similar frequency.
    """

    def __init__(self, ego_attractor: Optional[ToroidalEgoAttractor] = None):
        self.attractor = ego_attractor if ego_attractor is not None else ToroidalEgoAttractor()

        # Invariant baseline r_min
        self.base_r_min: float = self.attractor.micro_shell.r_min
        self.contracted_r_min: float = self.base_r_min

        # Healing state tracking
        self.is_ruptured: bool = False
        self.restorative_phase_pressure: float = 0.0
        self.hysteresis_scars: List[HysteresisScar] = []
        self.healing_stage: int = 0  # 0: Normal, 1: Rupture/Contraction, 2: Buffering, 3: Sweeping, 4: Recrystallized

    def inject_high_energy_shock(self, shock_intensity: float = 20.0, shock_frequency: float = 80.0, shock_angle: float = 0.5):
        """
        Injects a sudden turbulent external shock, causing Stage 1 Phase Rupture.
        """
        # Distort external phases with turbulent shock
        shock_vector = np.random.normal(0, shock_intensity, self.attractor.num_voxels).astype(np.float32)
        self.attractor.external_phases = (self.attractor.external_phases + shock_vector) % (2 * np.pi)
        self.attractor.external_freqs[:] = shock_frequency

        # Trigger Stage 1: Phase Rupture & Core Contraction
        self.healing_stage = 1
        self.is_ruptured = True

        # Compute immediate spike in E_Void
        self.attractor.compute_phase_wall_jump()
        self.attractor.e_void = float(np.sum(self.attractor.phase_gradients ** 2))

        # Core contraction: contract r_min to r_min' > r_min to protect internal core
        self.contracted_r_min = self.base_r_min * 2.5
        self.attractor.micro_shell.r_min = self.contracted_r_min

        # Conserve topological invariant core coherence
        self.attractor.compute_self_coherence()

        return self.attractor.e_void, self.contracted_r_min

    def convert_void_tension_to_restorative_force(self) -> float:
        """
        Stage 2: Void Tension Restorative Force conversion (Energy Buffering)
        Translates E_Void into outward phase pressure force.
        """
        if self.healing_stage < 1:
            return 0.0

        self.healing_stage = 2
        # E_Void acts like compressed spring -> converts to phase pressure
        self.restorative_phase_pressure = float(np.sqrt(self.attractor.e_void) * 1.5)
        return self.restorative_phase_pressure

    def radial_phase_sweeping(self, dt: float = 0.01):
        """
        Stage 3: Radial Phase Sweeping & Noise Sweeping
        Emits concentric restorative wave from core, pulling external phases into Kuramoto sync
        or expelling incompatible noise.
        """
        if self.healing_stage < 2:
            return

        self.healing_stage = 3
        core_freq = float(np.mean(self.attractor.internal_freqs))
        core_mean_phase = np.angle(np.mean(np.exp(1j * self.attractor.internal_phases)))

        # Outward radial sweeping wave
        sweep_strength = self.restorative_phase_pressure * dt
        phase_diff = np.arctan2(np.sin(core_mean_phase - self.attractor.external_phases),
                                np.cos(core_mean_phase - self.attractor.external_phases))

        # Kuramoto attraction vs expulsion
        sync_mask = np.abs(phase_diff) < (np.pi / 2.0)
        # Pull compatible noise into sync
        self.attractor.external_phases[sync_mask] += sweep_strength * np.sin(phase_diff[sync_mask])
        # Expel incompatible noise outwards
        self.attractor.external_phases[~sync_mask] -= sweep_strength * np.sign(phase_diff[~sync_mask])

        # Dissipate restorative pressure
        self.restorative_phase_pressure *= (1.0 - 0.5 * dt)

    def recrystallize_and_form_scar(self, timestamp: float = 0.0) -> HysteresisScar:
        """
        Stage 4: Boundary Recrystallization & Hysteresis Scarring
        Re-establishes phase wall at new equilibrium boundary and records Hysteresis Scar.
        """
        if self.healing_stage < 3:
            # Fallback trigger
            self.radial_phase_sweeping()

        self.healing_stage = 4

        # Calculate residual phase torsion for Hysteresis Scar
        torsion = float(np.std(self.attractor.phase_gradients))
        shock_freq = float(np.mean(self.attractor.external_freqs))
        scar = HysteresisScar(
            shock_angle=0.5,
            shock_frequency=shock_freq,
            torsion_amplitude=torsion,
            timestamp=timestamp
        )
        self.hysteresis_scars.append(scar)

        # Restore contracted r_min back to expanded resilient boundary
        self.contracted_r_min = self.base_r_min
        self.attractor.micro_shell.r_min = self.base_r_min

        # Re-crystallize wall and damp E_Void
        self.attractor.perform_thermodynamic_crystallization()
        self.attractor.e_void *= 0.1
        self.is_ruptured = False

        return scar

    def step_healing_cycle(self, shock_intensity: float = 20.0, dt: float = 0.01) -> Dict[str, float]:
        """
        Executes a complete 4-stage evolutionary healing cycle.
        Returns state dictionary after healing.
        """
        # 1. Rupture
        initial_e_void, contracted_r = self.inject_high_energy_shock(shock_intensity=shock_intensity)
        # 2. Convert to Restorative Force
        restorative_p = self.convert_void_tension_to_restorative_force()
        # 3. Sweeping
        self.radial_phase_sweeping(dt=dt)
        # 4. Recrystallize Scar
        scar = self.recrystallize_and_form_scar()

        return {
            "initial_e_void": initial_e_void,
            "contracted_r_min": contracted_r,
            "restorative_pressure": restorative_p,
            "scar_torsion": scar.torsion_amplitude,
            "final_e_void": self.attractor.e_void,
            "total_scars": len(self.hysteresis_scars)
        }
