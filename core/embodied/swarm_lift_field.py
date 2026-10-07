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
