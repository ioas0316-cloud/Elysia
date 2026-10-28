"""
core/embodied/swarm_lift_field.py

Drone Swarm Virtual Lift Field & 5D Phase Crystal Dislocation Jump Engine
==========================================================================
Integrates Trinity Wave Cells (1sin, 1cos, 1tan) Onion-Skin Field with
5D Clifford Bivector Rotors to achieve instantaneous Swarm Phase Re-locking
under severe environmental wind gust disruptions.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from core.physics.quantum_rotor_phase import Clifford5DRotorEngine


@dataclass
class DroneAgentState:
    """개별 드론의 5D 위상 상태 [X, Y, Z, V_x, V_y]"""
    drone_id: int
    state_5d: np.ndarray
    trinity_cell: Dict[str, float]  # {sin, cos, tan}
    is_dislocated: bool
    phase_locked: bool


class SwarmLiftField:
    """드론 군집 삼위일체 양파 껍질 포텐셜 장 및 5D 위상 도약 재정렬 모듈"""

    def __init__(self, num_drones: int = 6, dim: int = 5):
        self.num_drones = num_drones
        self.dim = dim
        self.rotor_engine = Clifford5DRotorEngine(dim=dim)

        self.drones: List[DroneAgentState] = []
        self._initialize_swarm_lattice()

    def _initialize_swarm_lattice(self):
        """120도 대칭 삼위일체 위상 결정 원형으로 드론 군집 초기 배치"""
        radius = 2.0
        for i in range(self.num_drones):
            angle = 2.0 * np.pi * i / self.num_drones
            state = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                1.0,  # Z 고도
                -0.1 * np.sin(angle),
                0.1 * np.cos(angle)
            ], dtype=float)

            trinity = self._compute_trinity_cell(state)
            self.drones.append(
                DroneAgentState(
                    drone_id=i,
                    state_5d=state,
                    trinity_cell=trinity,
                    is_dislocated=False,
                    phase_locked=True
                )
            )

    def _compute_trinity_cell(self, state_5d: np.ndarray) -> Dict[str, float]:
        """5D 위치 벡터로부터 (1sin, 1cos, 1tan) 삼위일체 파동 수치 산출"""
        r = np.linalg.norm(state_5d[:3]) + 1e-8
        sin_v = np.sin(r)
        cos_v = np.cos(r)
        tan_v = np.tan(r) if abs(cos_v) > 1e-3 else np.sign(sin_v) * 1e3
        return {"sin": float(sin_v), "cos": float(cos_v), "tan": float(tan_v)}

    def apply_wind_gust_shock(self, gust_vector_3d: np.ndarray):
        """외부 돌풍(Wind Gust) 충격 인가: 양파 껍질 포텐셜 장 왜곡 및 위상 결함 유발"""
        gust = np.asarray(gust_vector_3d, dtype=float)

        for drone in self.drones:
            drone.state_5d[:3] += gust * (0.8 + 0.4 * np.random.rand())
            drone.state_5d[3:5] += gust[:2] * 1.5

            drone.trinity_cell = self._compute_trinity_cell(drone.state_5d)

            if abs(drone.trinity_cell["tan"]) > 2.5 or np.linalg.norm(drone.state_5d[3:5]) > 1.8:
                drone.is_dislocated = True
                drone.phase_locked = False

    def detect_and_resolve_dislocations(self) -> Dict[str, Any]:
        """5D 결함 감지 시 직교 바이벡터 평면(e35)으로의 클리퍼드 로터 도약 및 위상 재고정"""
        jumps_executed = 0
        total_dislocations = sum(1 for d in self.drones if d.is_dislocated)

        for drone in self.drones:
            if drone.is_dislocated:
                Omega = self.rotor_engine.build_bivector_omega(
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, np.pi / 2.0, 0])
                )
                R = self.rotor_engine.exponential_map(Omega)
                v_transformed = self.rotor_engine.sandwich_transform(drone.state_5d, R)
                drone.state_5d = v_transformed / (np.linalg.norm(v_transformed) + 1e-12)

                drone.trinity_cell = self._compute_trinity_cell(drone.state_5d)
                drone.is_dislocated = False
                drone.phase_locked = True
                jumps_executed += 1

        mean_sin = np.mean([d.trinity_cell["sin"] for d in self.drones])
        mean_cos = np.mean([d.trinity_cell["cos"] for d in self.drones])
        phase_coherence = float(np.sqrt(mean_sin**2 + mean_cos**2))

        return {
            "initial_dislocations": total_dislocations,
            "rotor_jumps_executed": jumps_executed,
            "phase_coherence": phase_coherence,
            "all_phase_locked": all(d.phase_locked for d in self.drones)
        }


class DroneSwarmLiftFieldSimulator:
    """
    Simulates N drones creating a distributed aerodynamic potential field,
    reacting to wind gust shocks, and executing phase re-locking relaxation.
    """

    def __init__(
        self,
        num_drones: int = 32,
        coupling_K: float = 2.5,
        natural_freq_std: float = 0.1,
        spatial_dim: int = 3
    ):
        self.num_drones = num_drones
        self.coupling_K = coupling_K
        self.spatial_dim = spatial_dim

        # Natural frequencies ω_i
        np.random.seed(100)
        self.omega = np.random.normal(1.0, natural_freq_std, num_drones)

        # Drone phase angles θ_i ∈ [-π, π]
        self.phases = np.random.uniform(-np.pi, np.pi, num_drones)

        # Drone 3D lattice positions
        self.positions = np.random.uniform(-10.0, 10.0, (num_drones, spatial_dim))

        # Wind gust shock vector
        self.wind_gust = np.zeros(spatial_dim, dtype=float)

    def compute_order_parameter(self) -> Tuple[float, float]:
        """
        Computes Kuramoto macro order parameter R ∈ [0, 1] and average phase Ψ.
        R = |1/N ∑ e^(i θ_j)|
        """
        complex_sum = np.mean(np.exp(1j * self.phases))
        R = float(np.abs(complex_sum))
        Psi = float(np.angle(complex_sum))
        return R, Psi

    def apply_wind_gust_shock(self, gust_vector: np.ndarray, intensity: float = 5.0) -> None:
        """
        Applies external wind gust shock vector, disrupting drone phase synchronization.
        """
        self.wind_gust = np.asarray(gust_vector, dtype=float) * intensity

        # Disruption perturbation directly added to phase angles
        phase_pert = np.random.uniform(-np.pi * 0.8, np.pi * 0.8, self.num_drones)
        self.phases += phase_pert
        self.phases = (self.phases + np.pi) % (2.0 * np.pi) - np.pi

    def step_phase_locking_dynamics(self, dt: float = 0.05) -> Tuple[float, float, float]:
        """
        Integrates Kuramoto non-linear coupled differential equations:
        dθ_i/dt = ω_i + K/N ∑ sin(θ_j - θ_i) - Wind_Influence
        """
        N = self.num_drones
        phases_tile = np.tile(self.phases, (N, 1))
        phase_diffs = phases_tile - phases_tile.T

        # Coupling force: K/N * sum_j sin(θ_j - θ_i)
        coupling = (self.coupling_K / N) * np.sum(np.sin(phase_diffs), axis=1)

        # Wind shock influence on phase derivative
        wind_magnitude = float(np.linalg.norm(self.wind_gust))
        wind_influence = wind_magnitude * 0.1 * np.sin(self.phases)

        # Differential dθ_i/dt
        dtheta_dt = self.omega + coupling - wind_influence

        # Decay wind gust over time (dissipation)
        self.wind_gust *= 0.85

        # Integration step
        self.phases += dt * dtheta_dt
        self.phases = (self.phases + np.pi) % (2.0 * np.pi) - np.pi

        # Order parameter R and Psi
        R, Psi = self.compute_order_parameter()

        # Phase Divergence Error ΔΦ
        delta_phi = 1.0 - R

        # Synthetic Lift Field Energy Saved Ratio (relative to brute-force hovering)
        energy_saved_ratio = R * 0.75  # Up to 75% energy saved under perfect lock R = 1.0

        return R, delta_phi, energy_saved_ratio

    def run_relaxation_until_phase_lock(
        self,
        target_R: float = 0.90,
        max_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Runs phase re-locking relaxation loop until macro order parameter R exceeds target threshold.
        """
        history = []
        for step in range(1, max_steps + 1):
            R, delta_phi, energy_saved = self.step_phase_locking_dynamics(dt=0.05)
            history.append({"step": step, "R": R, "delta_phi": delta_phi, "energy_saved": energy_saved})

            if R >= target_R:
                break

        return {
            "converged": history[-1]["R"] >= target_R,
            "total_steps": len(history),
            "final_R": history[-1]["R"],
            "final_delta_phi": history[-1]["delta_phi"],
            "final_energy_saved": history[-1]["energy_saved"],
            "history": history
        }
