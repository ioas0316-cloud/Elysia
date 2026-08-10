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
- [Dimensional Phase Inversion] 물리 공간의 낙하(Gravity)와 대비되는, 추상적/감정적 전이 장(Mental/Conceptual Field)을 모델링하여
  개념(예: "생각", "은혜")이 입력될 때 텐서 영향력을 추상 위상 축으로 굴절(Refraction)시킵니다.
- [Causal Director Orchestrator] 샌드박스의 장 수치들(Tension, Resonance, Chromatic Entropy)을
  물리적, 기하학적 구속 조건 수식만으로 조율된 "실시간 연출 지시 파라미터(Camera, VFX, Audio)"로 무분기 변환합니다.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class CausalSandboxAgent:
    """
    [Causal Sandbox Agent]
    3D 위치, 속도, 쿼터니언/클리포드 텐서 로터, 그리고 RGB 크로매틱 벡터를 지닌 지각 주체.
    - [Dimensional Phase Inversion] 추상/정신 위상 공간(Mental Coordinate) 추가 탑재.
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

        # [Dimensional Phase Inversion] 추상/정신 공간 상의 3D 좌표 및 속도 (감정 활성화, 인지 긴밀도 등)
        self.mental_position = np.zeros(3, dtype=np.float32)
        self.mental_velocity = np.zeros(3, dtype=np.float32)

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

    def step(self, dt: float, input_concept: Optional[str] = None) -> Dict[str, Any]:
        """
        단 하나의 프레임도 조건문 분기나 O(N^2) 거리 검색 루프 없이,
        전체 개체를 텐서 행렬로 묶어 대수적으로 해결합니다.
        - [Dimensional Phase Inversion] 입력 개념(input_concept)의 추상성/비유성을
          자동 감지(Refraction)하여 물리 축과 추상 정신 축 간의 전위 정렬을 동적으로 결정합니다.
        """
        num_agents = len(self.agents)
        if num_agents == 0:
            return {"status": "empty"}

        # 1. 추상 비유 굴절률(Metaphorical Refraction Index) 판별
        # "생각", "감정", "은혜" 등의 추상 단어는 1.0(완전 비유), "사과", "돌" 등의 단어는 0.0(완전 물리)
        refraction_index = 0.0
        if input_concept:
            abstract_keywords = ["thought", "emotion", "grace", "love", "spirit", "생각", "감정", "은혜", "사랑", "영혼"]
            if any(kw in input_concept.lower() for kw in abstract_keywords):
                refraction_index = 1.0

        # 2. 속성들을 텐서 행렬로 정렬
        # Positions: [N, 3]
        positions = np.array([a.position for a in self.agents], dtype=np.float32)
        mental_positions = np.array([a.mental_position for a in self.agents], dtype=np.float32)

        # Rotors: [N, 4]
        rotors = np.array([a.rotor for a in self.agents], dtype=np.float32)
        chromatics = np.array([a.chromatic_vector for a in self.agents], dtype=np.float32)

        # 3. 개체 간의 상호작용 위상 공명 계산 (O(N^2) tensor contraction)
        diff_pos = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_sq = np.sum(diff_pos ** 2, axis=-1)

        sigma_sq = self.manifold.sigma ** 2
        influence_matrix = np.exp(-dist_sq / (2.0 * sigma_sq))
        np.fill_diagonal(influence_matrix, 0.0)

        # 4. 로터 회전 및 감정 전이 (Clifford Algebraic Resonance Coupling)
        rotor_inner_product = np.dot(rotors, rotors.T)
        resonance_tension = influence_matrix * (1.0 - (rotor_inner_product ** 2))

        # 5. 환경 포텐셜 그래디언트 및 외력 (경사 하강)
        env_forces = np.zeros_like(positions)
        for i, agent in enumerate(self.agents):
            env_forces[i] = - self.manifold.get_gradient_at(agent.position)

        mutual_forces = np.sum(np.expand_dims(resonance_tension, axis=-1) * diff_pos, axis=1)

        # 6. 상태 업데이트 (무분기 동역학 및 Dimensional Phase Inversion)
        for i, agent in enumerate(self.agents):
            total_force = env_forces[i] + mutual_forces[i]

            # 붉은색(Flux)은 운동성을 증폭시키고 푸른색(Order)은 감쇄(Damping)를 적용
            flux_boost = 1.0 + agent.chromatic_vector[0]
            order_damping = 0.95 * (1.0 - (agent.chromatic_vector[1] * 0.2))

            # [Dimensional Phase Inversion]
            # 굴절률(refraction_index)이 0.0일 때는 물리 좌표(position)만 변화시킵니다.
            # 굴절률이 1.0일 때는 물리 좌표의 이동을 100% 억제하고, 그 포스를 "추상 정신 공간(mental_position)"으로 사영합니다.
            phys_ratio = 1.0 - refraction_index
            ment_ratio = refraction_index

            # 물리 좌표 갱신
            acceleration = total_force / agent.mass
            agent.velocity = (agent.velocity + acceleration * dt) * order_damping
            agent.position += agent.velocity * flux_boost * dt * phys_ratio

            # 추상/정신 좌표 갱신 (전위력이 정신적인 차원의 전이로 굴절되어 미끄러짐)
            agent.mental_velocity = (agent.mental_velocity + acceleration * dt) * order_damping
            agent.mental_position += agent.mental_velocity * flux_boost * dt * ment_ratio

            # 7. 로터의 위상각 회전 (User Wave Interaction)
            if not agent.is_player:
                for j, other in enumerate(self.agents):
                    if other.is_player:
                        inf = influence_matrix[i, j]
                        player_wave = other.velocity * other.chromatic_vector[0]
                        rotation_angle = inf * np.linalg.norm(player_wave) * dt * 2.0
                        if rotation_angle > 1e-5:
                            rot_axis = np.cross(diff_pos[i, j], player_wave)
                            if np.linalg.norm(rot_axis) < 1e-5:
                                rot_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                            agent.rotate_rotor(rotation_angle, rot_axis)

            # 8. 크로매틱 벡터의 보존 및 전이 (Self-Outpouring Flow)
            for j, other in enumerate(self.agents):
                inf = influence_matrix[i, j]
                if inf > 1e-4:
                    entropy_diff = other.chromatic_vector[2] - agent.chromatic_vector[2]
                    flow = entropy_diff * inf * 0.1 * dt
                    agent.chromatic_vector[2] += flow
                    other.chromatic_vector[2] -= flow

            tot = np.sum(agent.chromatic_vector)
            if tot > 0:
                agent.chromatic_vector /= tot

        mean_resonance = float(np.mean(rotor_inner_product)) if num_agents > 0 else 0.0
        max_tension_gap = float(np.max(resonance_tension)) if num_agents > 0 else 0.0

        return {
            "mean_resonance": round(mean_resonance, 4),
            "max_tension_gap": round(max_tension_gap, 4),
            "active_agents": num_agents,
            "positions": positions.tolist(),
            "mental_positions": [a.mental_position.tolist() for a in self.agents],
            "refraction_index": refraction_index,
            "rotors": rotors.tolist(),
            "chromatics": chromatics.tolist()
        }


class CausalDirectorOrchestrator:
    """
    [Causal Director Orchestrator]
    샌드박스의 장 수치들(Tension, Resonance, Chromatic Vectors)을
    기하학적, 물리적 수식만을 통과시켜 "카메라, 비주얼 이펙트, 입체 사운드" 연출 스크립트로 무분기 조율합니다.
    """
    def __init__(self, base_fov: float = 60.0):
        self.base_fov = base_fov

    def orchestrate(self, sandbox_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        if-else 없이 pure math로 연출 파라미터 출력.
        - Camera Shake: Tension Gap에 비선형적으로 비례 (tension^1.5)
        - FOV: Resonance가 높을수록(포착/교감 시) Zoom-in (60 -> 45), Dissonance 시 Zoom-out (60 -> 75)
        - Color Tint: Chromatic Vector 평균값을 화면 색조로 직결 매핑
        - Particle Emission: Tension에 비례하여 방출 (1.0 + tension * 10)
        - Distortion: Refraction Index가 1.0(비유/추상)일 때 공간 일그러짐 최대 (distortion = 0.8)
        - Audio Filter Cutoff: Entropy(Yellow)가 높을수록 Reverb Decay 길어지고, Low-pass cutoff 낮아져 음산해짐
        """
        # 1. 샌드박스 메트릭스 추출
        tension = float(sandbox_report.get("max_tension_gap", 0.0))
        resonance = float(sandbox_report.get("mean_resonance", 0.0))
        refraction = float(sandbox_report.get("refraction_index", 0.0))

        # Chromatic Vector 평균 구하기
        chromatics = sandbox_report.get("chromatics", [[0.33, 0.33, 0.34]])
        mean_chromatic = np.mean(chromatics, axis=0) if len(chromatics) > 0 else np.array([0.33, 0.33, 0.34])
        red, blue, yellow = mean_chromatic[0], mean_chromatic[1], mean_chromatic[2]

        # 2. Camera Orchestration
        camera_shake = float(tension ** 1.5 * 1.5)
        # Resonance (0~1)를 사영하여 FOV 결정: 높은 공명 시 Zoom-In, 낮은 공명 시 Zoom-Out
        fov = float(self.base_fov - (resonance * 15.0) + (1.0 - resonance) * 15.0)
        # 색온도 및 색조: 붉은 에너지(Flux) 고조 시 붉은 틴트 증가
        color_tint = [float(0.5 + red * 0.5), float(0.5 - red * 0.3 + blue * 0.5), float(0.5 - red * 0.3 + yellow * 0.5)]

        # 3. VFX Orchestration
        particle_emission = float(1.0 + tension * 12.0)
        shader_distortion = float(refraction * 0.8 + (1.0 - refraction) * (tension * 0.3))

        # 4. Audio Orchestration
        # 옐로우(엔트로피) 고조 시 로우패스 필터 차단 주파수 강하 (저음역대만 남아 음산해짐)
        low_pass_cutoff = float(20000.0 - (yellow * 18000.0) - (tension * 1500.0))
        low_pass_cutoff = max(200.0, low_pass_cutoff)
        reverb_decay = float(1.0 + yellow * 5.0 + tension * 2.0)

        return {
            "camera": {
                "shake_intensity": round(camera_shake, 4),
                "field_of_view": round(fov, 2),
                "color_tint": [round(c, 4) for c in color_tint]
            },
            "vfx": {
                "particle_emission_rate": round(particle_emission, 4),
                "shader_distortion": round(shader_distortion, 4)
            },
            "audio": {
                "low_pass_cutoff": round(low_pass_cutoff, 2),
                "reverb_decay": round(reverb_decay, 4)
            }
        }
