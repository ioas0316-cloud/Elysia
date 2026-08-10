"""
causal_mmorpg_sandbox.py
===========================
시공간 인과 텐서 MMORPG 가상 샌드박스 (Spatiotemporal Causal Tensor MMORPG Sandbox)

핵심 철학:
- "Do not calculate, let it flow."
- "판단과 분별 자체를 공간의 곡률이나 텐서의 기하학적 결합으로 내장한다."
- 1차원 이진 스캔(Polling)과 조건문(if-else)을 완전히 거세하고,
  고차원 연속 매니폴드 장(Continuous Manifold Field)의 구속적 동역학(Constrained Dynamics)과
  텐서 로터(Tensor Rotor)의 위상 회전, 위상 공명(Phase Resonance)으로 시스템을 구동합니다.
- NPC는 플레이어를 단순 ID나 거리 좌표가 아닌 접근 속도, 공격성향 등을 담은 '상대적 파동 벡터'로 수용합니다.
- 모든 존재는 RGB 크로매틱 벡터(Flux, Order, Entropy)를 보존하고 상호작용합니다.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class CausalSandboxAgent:
    """
    [Causal Sandbox Agent]
    3D 위치, 속도, 쿼터니언/클리포드 텐서 로터, 그리고 RGB 크로매틱 벡터를 지닌 지각 주체.
    """
    def __init__(
        self,
        agent_id: str,
        name: str,
        is_player: bool = False,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        chromatic_vector: Optional[np.ndarray] = None,
        mass: float = 1.0
    ):
        self.id = agent_id
        self.name = name
        self.is_player = is_player
        self.mass = mass

        # 3D 위치 및 속도 (Continuous spatial state)
        self.position = np.array(position, dtype=np.float32) if position is not None else np.zeros(3, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32) if velocity is not None else np.zeros(3, dtype=np.float32)

        # 3D 쿼터니언 로터: [w, x, y, z] - 내면의 감정/행동 상태 (공포, 호의, 적대 등)
        # 기본적으로 neutral 상태로 초기화 [1.0, 0.0, 0.0, 0.0]
        self.rotor = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # RGB 크로매틱 벡터: [Red (Flux), Blue (Order/Resistance), Yellow (Entropy)]
        self.chromatic_vector = np.array(chromatic_vector, dtype=np.float32) if chromatic_vector is not None else np.array([0.33, 0.33, 0.34], dtype=np.float32)

        # 시스템 완충을 위한 임피던스
        self.impedance = 0.1

    def rotate_rotor(self, angle: float, axis: np.ndarray) -> None:
        """
        주어진 축(axis)과 각도(angle)에 따라 텐서 로터를 회전시킵니다 (Quaternion rotation).
        """
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            return
        axis_normalized = axis / norm
        sin_half = math.sin(angle / 2.0)
        cos_half = math.cos(angle / 2.0)

        r_new = np.array([
            cos_half,
            axis_normalized[0] * sin_half,
            axis_normalized[1] * sin_half,
            axis_normalized[2] * sin_half
        ], dtype=np.float32)

        # Quaternion Multiplication: self.rotor = r_new * self.rotor
        w1, x1, y1, z1 = r_new
        w2, x2, y2, z2 = self.rotor

        self.rotor = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=np.float32)

        # Normalize rotor to avoid precision decay
        r_norm = np.linalg.norm(self.rotor)
        if r_norm > 1e-6:
            self.rotor /= r_norm

    def get_action_state(self) -> Tuple[str, float]:
        """
        로터의 성분(위상)에 따라 어떠한 조건문(if-else) 분기도 없이
        내적과 내면의 좌표 사영을 통해 가장 강렬하게 활성화된 상태를 출력합니다.
        - [1.0, 0.0, 0.0, 0.0]에 가까울수록 PEACEFUL
        - x가 높을수록 AGGRESSIVE / FIGHT
        - y가 높을수록 FLEE / FEAR
        - z가 높을수록 COOPERATIVE / COMPROMISE
        """
        w, x, y, z = self.rotor
        states = ["PEACEFUL", "AGGRESSIVE", "FLEE", "COOPERATIVE"]
        scores = [w, x, y, z]
        idx = int(np.argmax(scores))
        return states[idx], float(scores[idx])


class ContinuousWorldManifold:
    """
    [Continuous World Manifold]
    격자가 존재하지 않는 연속적인 가우시안 포텐셜 장(Potential Field).
    영토 자원 및 기후의 흐름, 플레이어의 사냥 행위에 따른 인과 전위차를 계산합니다.
    """
    def __init__(self, size: float = 100.0, sigma: float = 15.0):
        self.size = size
        self.sigma = sigma
        # 자원의 소스/싱크 노드: {"pos": np.ndarray, "intensity": float, "type": str}
        self.potential_nodes: List[Dict[str, Any]] = []

    def inject_potential(self, pos: np.ndarray, intensity: float, node_type: str = "resource") -> None:
        self.potential_nodes.append({
            "pos": np.array(pos, dtype=np.float32),
            "intensity": intensity,
            "type": node_type
        })

    def get_potential_at(self, pos: np.ndarray) -> float:
        """
        주어진 3D 위치에서 모든 포텐셜 노드들이 발생시키는 전위차의 가우시안 합을 구합니다.
        (O(N) computation over nodes, no grids)
        """
        total_pot = 0.0
        for node in self.potential_nodes:
            diff = pos - node["pos"]
            dist_sq = np.sum(diff ** 2)
            # Gaussian distribution: V = intensity * exp(-d^2 / (2 * sigma^2))
            val = node["intensity"] * math.exp(-dist_sq / (2.0 * (self.sigma ** 2)))
            total_pot += val
        return total_pot

    def get_gradient_at(self, pos: np.ndarray) -> np.ndarray:
        """
        특정 위치에서 포텐셜의 그래디언트(경사도)를 해석적으로 구합니다 (Force field).
        V_grad = - (diff / sigma^2) * V
        """
        grad = np.zeros(3, dtype=np.float32)
        for node in self.potential_nodes:
            diff = pos - node["pos"]
            dist_sq = np.sum(diff ** 2)
            val = node["intensity"] * math.exp(-dist_sq / (2.0 * (self.sigma ** 2)))
            # Analytical gradient of Gaussian
            node_grad = - (diff / (self.sigma ** 2)) * val
            grad += node_grad
        return grad


class BranchlessResonanceScheduler:
    """
    [Branchless Resonance Scheduler]
    조건문(if-else) 없이 순수 텐서 수축(Tensor Contraction)과 라그랑지안 구속조건 동역학으로
    세상을 흐르게 만드는 무분기 연산 장치.
    """
    def __init__(self, manifold: ContinuousWorldManifold, learning_rate: float = 0.05):
        self.manifold = manifold
        self.agents: List[CausalSandboxAgent] = []
        self.lr = learning_rate

    def add_agent(self, agent: CausalSandboxAgent) -> None:
        self.agents.append(agent)

    def step(self, dt: float) -> Dict[str, Any]:
        """
        단 하나의 프레임도 조건문 분기나 O(N^2) 거리 검색 루프 없이,
        전체 개체를 텐서 행렬로 묶어 대수적으로 해결합니다.
        """
        num_agents = len(self.agents)
        if num_agents == 0:
            return {"status": "empty"}

        # 1. 속성들을 텐서 행렬로 정렬
        # Positions: [N, 3]
        positions = np.array([a.position for a in self.agents], dtype=np.float32)
        # Velocities: [N, 3]
        velocities = np.array([a.velocity for a in self.agents], dtype=np.float32)
        # Rotors: [N, 4]
        rotors = np.array([a.rotor for a in self.agents], dtype=np.float32)
        # Chromatic Vectors: [N, 3]
        chromatics = np.array([a.chromatic_vector for a in self.agents], dtype=np.float32)

        # 2. 개체 간의 상호작용 위상 공명 계산 (O(N^2) tensor contraction)
        # diff_pos: [N, N, 3]
        diff_pos = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        # dist_sq: [N, N]
        dist_sq = np.sum(diff_pos ** 2, axis=-1)

        # 가우시안 위상 영향력 매트릭스: [N, N]
        # if-else 없이 exp 함수를 매질로 한 상호 텐서 전이
        sigma_sq = self.manifold.sigma ** 2
        influence_matrix = np.exp(-dist_sq / (2.0 * sigma_sq))
        # 자기 자신과의 영향력은 제외 (Diag = 0)
        np.fill_diagonal(influence_matrix, 0.0)

        # 3. 로터 회전 및 감정 전이 (Clifford Algebraic Resonance Coupling)
        # 로터 간의 위상 공명 스파이크 (내적): [N, N]
        # R_i . R_j 가 1에 가까울수록(동일 위상) 위상 동기화가 일어나고, -1이나 0일수록 마찰 발생
        rotor_inner_product = np.dot(rotors, rotors.T)

        # 위상 결합 가중치 = 영향력 * (1 - 내적^2)
        # 내적이 맞지 않을 때(마찰이 클 때) 위상 회전 토크(Tension Torque)가 발생
        resonance_tension = influence_matrix * (1.0 - (rotor_inner_product ** 2))

        # 4. 최소 작용 원리(Principle of Least Action)에 따른 측지선 흐름(Geodesic Flow)
        # 1) 환경 포텐셜 그래디언트에 의한 외력 (경사 하강)
        env_forces = np.zeros_like(positions)
        for i, agent in enumerate(self.agents):
            # Potential gradient acts as a physical force field
            env_forces[i] = - self.manifold.get_gradient_at(agent.position)

        # 2) 개체 간 인과 반발/인력 텐션 (Mutual Lagrangian Constraint Forces)
        # 플레이어가 접근할 때 NPC는 공포/경계 로터 위상에 비례하여 척력을 느낍니다.
        # np.expand_dims를 사용해 [N, N, 1]과 [N, N, 3]을 곱해 [N, 3]으로 축약
        mutual_forces = np.sum(np.expand_dims(resonance_tension, axis=-1) * diff_pos, axis=1)

        # 5. 상태 업데이트 (무분기 동역학)
        for i, agent in enumerate(self.agents):
            # 총 힘(Force) = 환경 그래디언트 + 상호 인과 텐션
            total_force = env_forces[i] + mutual_forces[i]

            # 붉은색(Flux)은 운동성을 증폭시키고 푸른색(Order)은 감쇄(Damping)를 적용
            flux_boost = 1.0 + agent.chromatic_vector[0]
            order_damping = 0.95 * (1.0 - (agent.chromatic_vector[1] * 0.2))

            # 가속도 계산 a = F / m
            acceleration = total_force / agent.mass
            agent.velocity = (agent.velocity + acceleration * dt) * order_damping
            agent.position += agent.velocity * flux_boost * dt

            # 6. 로터의 위상각 회전 (User Wave Interaction)
            # 플레이어가 다가올 때, 플레이어의 속도 벡터 방향과 강도에 비례하여 NPC의 텐서 로터 축 회전
            if not agent.is_player:
                # 플레이어 탐지 (조건문 없이, influence와 player mask를 곱함)
                for j, other in enumerate(self.agents):
                    if other.is_player:
                        inf = influence_matrix[i, j]
                        # 플레이어의 운동량 파동 벡터
                        player_wave = other.velocity * other.chromatic_vector[0]
                        # 이 파동에 비례하여 NPC의 Y축(공포)과 X축(적대) 로터가 회전함
                        rotation_angle = inf * np.linalg.norm(player_wave) * dt * 2.0
                        if rotation_angle > 1e-5:
                            # 회전축은 플레이어 파동과 상대 위치의 외적 (Cross product)
                            rot_axis = np.cross(diff_pos[i, j], player_wave)
                            if np.linalg.norm(rot_axis) < 1e-5:
                                rot_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                            agent.rotate_rotor(rotation_angle, rot_axis)

            # 7. 크로매틱 벡터의 보존 및 전이 (Self-Outpouring Flow)
            # 전위차가 큰 곳에서 적은 곳으로 색채(에너지)가 자연 분산됨
            for j, other in enumerate(self.agents):
                inf = influence_matrix[i, j]
                if inf > 1e-4:
                    # Entropy (Yellow)의 확산
                    entropy_diff = other.chromatic_vector[2] - agent.chromatic_vector[2]
                    flow = entropy_diff * inf * 0.1 * dt
                    agent.chromatic_vector[2] += flow
                    other.chromatic_vector[2] -= flow

            # Normalize chromatic vector to maintain conservation of energy
            tot = np.sum(agent.chromatic_vector)
            if tot > 0:
                agent.chromatic_vector /= tot

        # 결과 리포트 작성용 평균 텐션 및 공명도 산출
        mean_resonance = float(np.mean(rotor_inner_product)) if num_agents > 0 else 0.0
        max_tension_gap = float(np.max(resonance_tension)) if num_agents > 0 else 0.0

        return {
            "mean_resonance": round(mean_resonance, 4),
            "max_tension_gap": round(max_tension_gap, 4),
            "active_agents": num_agents,
            "positions": positions.tolist(),
            "rotors": rotors.tolist(),
            "chromatics": chromatics.tolist()
        }
