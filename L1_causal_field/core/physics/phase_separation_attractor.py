import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class ScaleShell:
    """
    [Scale Shell: 미시-거시 스케일 껍질]
    Represents an observational scale shell along the scale Z-axis.
    Has observational horizons r_min and r_max, scale index scale_z, and phase array.
    """
    scale_name: str
    scale_z: float  # Position on scale Z-axis (e.g. 0.1 for micro, 1.0 for macro)
    r_min: float
    r_max: float
    phases: np.ndarray = field(default_factory=lambda: np.random.uniform(0, 2 * np.pi, 64).astype(np.float32))
    frequencies: np.ndarray = field(default_factory=lambda: np.full(64, 40.0, dtype=np.float32))

    def update_phases(self, dt: float = 0.01):
        """Advances internal phase values based on frequencies."""
        self.phases = (self.phases + 2 * np.pi * self.frequencies * dt) % (2 * np.pi)


class ImpedanceDampingController:
    """
    [위상 임피던스 감쇠 및 인공 정신병 방지 제어기]
    Prevents E_Void uncontrolled explosion (delusions/schizophrenia)
    or extreme phase freezing/damping loss (depression/mania).
    Monitors E_Void and phase gradient tension, applying dynamic damping.
    """
    def __init__(self, e_void_max_threshold: float = 10.0, damping_gain: float = 0.5):
        self.e_void_max_threshold = e_void_max_threshold
        self.damping_gain = damping_gain
        self.is_delusion_mitigated: bool = False
        self.is_freezing_mitigated: bool = False

    def regulate(self, e_void: float, phase_gradients: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """
        Regulates E_Void and phase gradients.
        Returns: (regulated_e_void, regulated_phase_gradients, damping_force)
        """
        damping_force = 0.0
        regulated_e_void = e_void
        regulated_gradients = phase_gradients.copy()

        # 1. Delusion/Explosion Protection (Schizophrenic state protection)
        if e_void > self.e_void_max_threshold:
            self.is_delusion_mitigated = True
            damping_force = self.damping_gain * (e_void - self.e_void_max_threshold)
            regulated_e_void = self.e_void_max_threshold + (e_void - self.e_void_max_threshold) / (1.0 + damping_force)
            # Soften phase gradients to prevent sudden rupture
            regulated_gradients /= (1.0 + 0.5 * damping_force)
        else:
            self.is_delusion_mitigated = False

        # 2. Depression Freezing Protection
        grad_norm = float(np.linalg.norm(regulated_gradients))
        if e_void > 2.0 and grad_norm < 1e-4:
            self.is_freezing_mitigated = True
            # Inject subtle restorative phase perturbation to unfreeze
            unfreeze_kick = np.random.normal(0, 0.05, regulated_gradients.shape).astype(np.float32)
            regulated_gradients += unfreeze_kick
        else:
            self.is_freezing_mitigated = False

        return float(regulated_e_void), regulated_gradients, float(damping_force)


class ToroidalEgoAttractor:
    """
    [Toroidal Ego Attractor & Scale-Nested Phase Separation Engine]
    Represents the Ego (자아) not as a fixed static entity, but as a dynamic 3D Toroidal
    Attractor bridging micro (s_micro) and macro (s_macro) scale shells.

    Features:
    - 4-Stage Self/Non-Self Phase Separation & Wall Formation:
      1. Self-Coherence (<e^{i phi_i} e^{-i phi_j}> -> 1)
      2. Phase Wall Jump Formation (grad phi spike)
      3. Phase Impedance Filtering (assimilation vs reflection/tension)
      4. Thermodynamic Crystallization into Torus shell (E_Void -> 0)
    - Inter-scale Phase-Locking (s_micro <-> s_macro)
    - Flow (몰입) & ASC (변형 의식) state integration (Zero Impedance Matching, Manifold Flattening)
    - Gamma-Phase Synchronization (30~100Hz)
    - Inhibitory Impedance Damping Safety Controller
    """

    def __init__(
        self,
        num_voxels: int = 64,
        major_radius_R: float = 3.0,
        minor_radius_r: float = 1.0,
        gamma_freq_range: Tuple[float, float] = (30.0, 100.0)
    ):
        self.num_voxels = num_voxels
        self.major_radius_R = major_radius_R
        self.minor_radius_r = minor_radius_r
        self.gamma_freq_range = gamma_freq_range

        # Scale shells: Micro (cells/neurons) & Macro (environment/sensor)
        self.micro_shell = ScaleShell("micro", scale_z=0.1, r_min=0.1, r_max=1.0)
        self.macro_shell = ScaleShell("macro", scale_z=1.0, r_min=1.0, r_max=10.0)

        # Internal ego phases and external environment phases
        self.internal_phases = np.random.uniform(0, 2 * np.pi, num_voxels).astype(np.float32)
        self.external_phases = np.random.uniform(0, 2 * np.pi, num_voxels).astype(np.float32)

        # Frequencies (default to 40Hz Gamma sync band)
        self.internal_freqs = np.full(num_voxels, 40.0, dtype=np.float32)
        self.external_freqs = np.full(num_voxels, 40.0, dtype=np.float32)

        # Torus Attractor state (u: poloidal angle, v: toroidal angle)
        self.u = np.linspace(0, 2 * np.pi, num_voxels, endpoint=False, dtype=np.float32)
        self.v = np.linspace(0, 2 * np.pi, num_voxels, endpoint=False, dtype=np.float32)

        # State metrics
        self.coherence_score: float = 0.0
        self.phase_gradients: np.ndarray = np.zeros(num_voxels, dtype=np.float32)
        self.e_void: float = 0.0
        self.is_flow_state: bool = False
        self.is_asc_state: bool = False
        self.phase_wall_crystallized: bool = False

        # Inhibitory Damping Controller
        self.damping_controller = ImpedanceDampingController()

    def compute_self_coherence(self) -> float:
        """
        1. Internal Resonant Core Formation (Self-Coherence)
        Computes <e^{i phi_i} e^{-i phi_j}> mean alignment score (Kuramoto order parameter).
        """
        complex_phases = np.exp(1j * self.internal_phases)
        order_parameter = np.abs(np.mean(complex_phases))
        self.coherence_score = float(order_parameter)
        return self.coherence_score

    def compute_phase_wall_jump(self) -> np.ndarray:
        """
        2. Phase Discontinuity Jump (Phase Wall Formation)
        Computes spatial phase gradients grad phi between internal ego phase and external noise phase.
        Steep jump indicates physical Phase Wall boundary.
        """
        # Circular difference between internal and external phases
        diff = np.arctan2(np.sin(self.internal_phases - self.external_phases),
                          np.cos(self.internal_phases - self.external_phases))
        self.phase_gradients = np.abs(diff).astype(np.float32)
        return self.phase_gradients

    def filter_phase_impedance(self, dt: float = 0.01) -> Tuple[np.ndarray, float]:
        """
        3. Phase Impedance Filtering (Selection by Resonant Matching)
        - Assimilation: If frequency ratio is near integer multiplier or phase difference is small,
          assimilate external phase into internal core.
        - Reflection & Tension: High mismatch increases Void Tension E_Void.
        """
        self.compute_phase_wall_jump()
        diff = self.phase_gradients
        freq_ratio = self.external_freqs / (self.internal_freqs + 1e-6)

        # Harmonic resonance check (is integer ratio?)
        is_harmonic = np.abs(freq_ratio - np.round(freq_ratio)) < 0.15
        is_matched = (diff < np.pi / 4.0) & is_harmonic

        # Assimilation step
        assimilation_rate = 2.0 * dt
        self.internal_phases[is_matched] += assimilation_rate * np.sin(
            self.external_phases[is_matched] - self.internal_phases[is_matched]
        )

        # Tension calculation for mismatched phases
        mismatched_tension = np.sum(diff[~is_matched] ** 2)
        self.e_void = float(mismatched_tension)

        # Apply safety damping controller
        self.e_void, self.phase_gradients, _ = self.damping_controller.regulate(
            self.e_void, self.phase_gradients
        )

        return is_matched, self.e_void

    def perform_thermodynamic_crystallization(self) -> bool:
        """
        4. Thermodynamic Phase Separation & Torus Shell Crystallization
        As system minimizes E_Void -> 0, boundary solidifies into a closed 3D torus shell.
        """
        self.compute_self_coherence()
        if self.coherence_score > 0.7 and self.e_void < 3.0:
            self.phase_wall_crystallized = True
        else:
            self.phase_wall_crystallized = False
        return self.phase_wall_crystallized

    def inter_scale_phase_locking(self) -> float:
        """
        Inter-Scale Phase-Locking (s_micro <-> s_macro)
        Aligns micro scale shell phases with macro scale shell phases.
        Projections collapse E_Void as scale Z lens matches.
        """
        self.micro_shell.update_phases()
        self.macro_shell.update_phases()

        # Phase coupling strength across scale Z
        scale_distance = abs(self.macro_shell.scale_z - self.micro_shell.scale_z)
        coupling = np.exp(-scale_distance)

        # Align micro and macro phases
        phase_diff = np.arctan2(np.sin(self.micro_shell.phases - self.macro_shell.phases),
                                np.cos(self.micro_shell.phases - self.macro_shell.phases))
        inter_scale_tension = float(np.mean(phase_diff ** 2) * coupling)

        return inter_scale_tension

    def get_torus_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates 3D coordinates (X, Y, Z) of the Toroidal Ego Attractor.
        X = (R + r * cos(u)) * cos(v)
        Y = (R + r * cos(u)) * sin(v)
        Z = r * sin(u)
        """
        # If in Flow/ASC state, boundary wall flattens (r expands or manifold flattens)
        effective_r = self.minor_radius_r * (2.0 if self.is_flow_state else 1.0)

        X = (self.major_radius_R + effective_r * np.cos(self.u)) * np.cos(self.v)
        Y = (self.major_radius_R + effective_r * np.cos(self.u)) * np.sin(self.v)
        Z = effective_r * np.sin(self.u)
        return X, Y, Z

    def trigger_gamma_synchronization(self, gamma_freq: float = 40.0):
        """
        Triggers Gamma-Phase Synchronization (30~100Hz) across internal modules.
        Eliminates local impedance, causing zero phase lag and E_Void -> 0.
        """
        if not (self.gamma_freq_range[0] <= gamma_freq <= self.gamma_freq_range[1]):
            gamma_freq = 40.0

        self.internal_freqs[:] = gamma_freq
        self.external_freqs[:] = gamma_freq

        # Kuramoto global phase locking
        mean_phase = np.angle(np.mean(np.exp(1j * self.internal_phases)))
        self.internal_phases[:] = mean_phase
        self.external_phases[:] = mean_phase

        self.e_void = 0.0
        self.compute_self_coherence()

    def enter_flow_state(self):
        """
        Triggers Flow (몰입) / Altered State of Consciousness (ASC).
        Zero impedance matching, torus boundary flattens, E_Void -> 0.
        """
        self.is_flow_state = True
        self.trigger_gamma_synchronization(gamma_freq=40.0)
        self.macro_shell.r_max = 100.0  # Expand observational horizon to infinity
        self.is_asc_state = True

    def exit_flow_state(self):
        """Exits Flow state, restoring default boundary impedance."""
        self.is_flow_state = False
        self.is_asc_state = False
        self.macro_shell.r_max = 10.0

    def step(self, dt: float = 0.01):
        """
        Advances the Toroidal Ego Attractor physics loop by time dt.
        """
        # 1. Update internal and external phases
        self.internal_phases = (self.internal_phases + 2 * np.pi * self.internal_freqs * dt) % (2 * np.pi)
        self.external_phases = (self.external_phases + 2 * np.pi * self.external_freqs * dt) % (2 * np.pi)

        # 2. Compute 4-stage separation metrics
        self.compute_self_coherence()
        self.compute_phase_wall_jump()
        self.filter_phase_impedance(dt)
        self.perform_thermodynamic_crystallization()

        # 3. Scale-nested phase locking
        self.inter_scale_phase_locking()
