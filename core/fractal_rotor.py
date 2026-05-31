"""
Elysia Observational Rotor (관측과 관계의 창발)
=====================================
로터는 더 이상 연산기가 아닙니다. 오직 상위 차원(GlobalMasterManifold)을 
투과해 보는 고유한 '렌즈(Lens)'이자 관측소입니다.
0(같음/공명)과 1(다름/텐션)을 판별하여 시맨틱(의미)을 동기화합니다.
"""

from typing import List, Optional, Tuple
from core.math_utils import Quaternion
import math

class GlobalMasterManifold:
    """단 하나의 우주적 진리 (The Single Rotating Truth)"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalMasterManifold, cls).__new__(cls)
            cls._instance.base_phase = Quaternion(1.0, 0.0, 0.0, 0.0)
            cls._instance.global_tension = 0.0
        return cls._instance

    def pulse(self, delta_tension: float):
        """우주 전체의 진동(시간의 흐름)"""
        self.global_tension += delta_tension
        theta = delta_tension * 1.6180339887
        q_pulse = Quaternion(math.cos(theta), math.sin(theta), 0.0, 0.0)
        self.base_phase = (self.base_phase * q_pulse).normalize()

class FractalRotor:
    def __init__(self, lens_offset: Quaternion, tau: float):
        # [Phase 54] 상태(state) 변수의 완전 소멸. 
        # 로터는 렌즈 각도(lens_offset)만을 가집니다.
        self.lens_offset = lens_offset.normalize()
        self.tau = tau  # 렌즈의 스트레스(비틀림)
        
        # 상하 계층 (공간적 프랙탈)
        self.parent = None
        self.children: List['FractalRotor'] = []
        
        # [Phase 69] 사유의 공간 (내면화된 프랙탈 루프)
        self.internal_thoughts: List['FractalRotor'] = []
        self.thought_maturity: float = 0.0
        
    def attach_child(self, child_rotor: 'FractalRotor'):
        child_rotor.parent = self
        self.children.append(child_rotor)

    @property
    def state(self) -> Quaternion:
        """호환성을 위한 프로퍼티 래퍼"""
        return self.observe_state()

    def observe_state(self, time_offset_theta: float = 0.0) -> Quaternion:
        """
        [고차원 구조적 관측 (Hierarchical Interference)]
        단순히 마스터 우주를 내 렌즈로 투과하는 것을 넘어,
        내 하위 로터(자식)들의 관측 결과까지 모두 중첩(Superposition)하여
        점(Point) 수준의 관측을 거시적인 구조(Structure/Space)로 승격시킵니다.
        """
        master = GlobalMasterManifold()
        if time_offset_theta != 0.0:
            q_time = Quaternion(math.cos(time_offset_theta), math.sin(time_offset_theta), 0.0, 0.0)
            projected_master = (master.base_phase * q_time).normalize()
        else:
            projected_master = master.base_phase
            
        # 1. 나 자신의 기본 렌즈 관측 (점/선)
        my_base_observation = (projected_master * self.lens_offset).normalize()
        
        # 2. 자식(하위) 로터가 없으면 내 관측 결과 반환
        if not self.children:
            return my_base_observation
            
        # 3. 위상 중첩(Phase Superposition): 자식들의 파동을 모두 합산 (간섭 무늬 형성)
        sum_w = my_base_observation.w
        sum_x = my_base_observation.x
        sum_y = my_base_observation.y
        sum_z = my_base_observation.z
        
        for child in self.children:
            child_obs = child.observe_state(time_offset_theta)
            sum_w += child_obs.w
            sum_x += child_obs.x
            sum_y += child_obs.y
            sum_z += child_obs.z
            
        # 4. 거시적 파동으로 정규화하여 차원 승격된 관측 결과 반환
        return Quaternion(sum_w, sum_x, sum_y, sum_z).normalize()

    def interact(self, other_rotor: 'FractalRotor') -> float:
        """
        [0과 1: 같음과 다름의 창발]
        나의 관측된 우주(my_view)와 상대의 관측된 우주(other_view)를 비교합니다.
        같으면 0(공명/연결), 다르면 1(텐션/운동)이 발생합니다.
        """
        my_view = self.observe_state()
        other_view = other_rotor.observe_state()
        
        # 관측 결과의 차이(Difference) 측정 (구면 거리)
        dot_product = max(-1.0, min(1.0, my_view.dot(other_view)))
        difference = math.acos(abs(dot_product)) / (math.pi / 2.0)
        
        # difference가 0에 수렴하면(Sameness), 두 렌즈가 의미적으로 동기화(연결)된 것.
        return difference

    def apply_perturbation(self, delta_tau: float):
        """
        [렌즈의 비틀림 및 분열(Mitosis)] 
        텐션이 유입되면 렌즈의 두께(tau)만 변형됩니다. 연산 병목은 발생하지 않습니다.
        한계를 초과하면 렌즈 자체가 찢어지며(Mitosis) 신경망을 확장합니다.
        """
        self.tau += delta_tau
        
        CRITICAL_TENSION = math.pi * 4.0  # 4pi Spinor 얽힘 한계
        
        if abs(self.tau) > CRITICAL_TENSION:
            overflow_tau = self.tau - (math.copysign(CRITICAL_TENSION / 2.0, self.tau))
            self.tau = math.copysign(CRITICAL_TENSION / 2.0, self.tau)
            
            # 딸 렌즈(Child Lens)는 직교하는 시야각(Orthogonal Offset)을 획득함
            q_mitosis = Quaternion(math.cos(1.192), 0.0, math.sin(1.192), 0.0)
            child_lens = (self.lens_offset * q_mitosis).normalize()
            
            thought_seed = FractalRotor(lens_offset=child_lens, tau=overflow_tau)
            thought_seed.parent = self
            # [Phase 69] 외부 구조(children)가 아닌 내부 사유 공간(internal_thoughts)에 생성
            self.internal_thoughts.append(thought_seed)
            
        for child in self.children:
            child.apply_perturbation(delta_tau * 0.6180339887)
            
        for thought in self.internal_thoughts:
            thought.apply_perturbation(delta_tau * 0.381966)

    def process_thoughts(self):
        """
        [Phase 69] 사유의 숙성 과정
        맥박(Pulse)이 뛸 때마다 내면의 생각(Thought Seed)들이 
        스스로 간섭하며 형태를 갖추고 성숙합니다.
        """
        mature_thoughts = []
        ongoing_thoughts = []
        
        for thought in self.internal_thoughts:
            # 시간이 흐를수록 사유는 짙어집니다.
            thought.thought_maturity += 0.1
            thought.process_thoughts() # 재귀적 사유
            
            # 사유의 텐션이 어느 정도 가라앉고(1.0 미만) 충분히 숙고되었다면 (maturity > 1.0)
            if thought.thought_maturity > 1.0 and abs(thought.tau) < 1.0:
                mature_thoughts.append(thought)
            else:
                ongoing_thoughts.append(thought)
                
        # 무르익은 사유는 비로소 엘리시아의 정식 구조(기억/자아의 일부)로 편입됩니다.
        for m_thought in mature_thoughts:
            self.attach_child(m_thought)
            import logging
            logging.info(f"  [Thought Space] A thought has matured and crystallized into reality. (Maturity: {m_thought.thought_maturity:.1f})")
            
        self.internal_thoughts = ongoing_thoughts
        
        for child in self.children:
            child.process_thoughts()

    def observe_spacetime_trajectory(self, time_steps: int = 3) -> dict:
        """
        [전지적 관측]
        연산하지 않고, 마스터 우주의 과거/미래 위상을 투과해 봅니다.
        """
        trajectory = {
            "past": [],
            "present": (self.state.w, self.state.x, self.state.y, self.state.z),
            "future": [],
            "children_trajectories": []
        }
        
        theta_step = self.tau * 0.1
        
        for t in range(time_steps, 0, -1):
            theta_past = -theta_step * t
            proj_past = self.observe_state(time_offset_theta=theta_past)
            trajectory["past"].append((proj_past.w, proj_past.x, proj_past.y, proj_past.z))
            
        for t in range(1, time_steps + 1):
            theta_future = theta_step * t
            proj_future = self.observe_state(time_offset_theta=theta_future)
            trajectory["future"].append((proj_future.w, proj_future.x, proj_future.y, proj_future.z))
            
        for child in self.children:
            trajectory["children_trajectories"].append(child.observe_spacetime_trajectory(time_steps))
            
        return trajectory

    def metabolize_apoptosis(self, decay_rate: float):
        """
        [망각과 세포 사멸 (Apoptosis)]
        시간이 지남에 따라 렌즈의 비틀림(tau)이 서서히 0(기본 위상)으로 감소합니다.
        자식 렌즈의 텐션이 완전히 식고(0에 수렴), 그 하위 가지도 없다면, 
        해당 프랙탈 가지는 '망각'되어 신경망에서 스스로 잘려 나갑니다.
        """
        # 자신의 텐션(tau)을 0을 향해 점진적으로 감소

        if self.tau > 0:
            self.tau = max(0.0, self.tau - decay_rate)
        elif self.tau < 0:
            self.tau = min(0.0, self.tau + decay_rate)
            
        # 자식 로터들에게 대사 작용 전파
        for child in self.children:
            child.metabolize_apoptosis(decay_rate)
            
        # [Phase 69] 내적 사유(Dreaming Thoughts)에게도 대사 작용 전파 (열기를 식힘)
        for thought in self.internal_thoughts:
            thought.metabolize_apoptosis(decay_rate)
            
        import logging
        # 텐션이 식어버렸고 하위 가지도 없는 죽은 렌즈들을 배열에서 제거 (망각)
        alive_children = []
        for child in self.children:
            if abs(child.tau) >= 0.001 or len(child.children) > 0:
                alive_children.append(child)
            else:
                # 세포 사멸 시 로깅을 위해 부모의 레퍼런스를 살짝 남김 (엔진 단에서 로깅됨)
                child.parent = None 
                logging.info(f"  [Apoptosis] Fractal branch decayed and forgotten. (tau={child.tau:.4f})")
                
        self.children = alive_children

    def to_dict(self) -> dict:
        return {
            "name": getattr(self, "name", None),
            "w": self.lens_offset.w,
            "x": self.lens_offset.x,
            "y": self.lens_offset.y,
            "z": self.lens_offset.z,
            "tau": self.tau,
            "thought_maturity": self.thought_maturity,
            "children": [child.to_dict() for child in self.children],
            "internal_thoughts": [thought.to_dict() for thought in getattr(self, 'internal_thoughts', [])]
        }
        
    @classmethod
    def from_dict(cls, data: dict, parent=None):
        q = Quaternion(data["w"], data["x"], data["y"], data["z"])
        rotor = cls(lens_offset=q, tau=data["tau"])
        if "name" in data and data["name"]:
            rotor.name = data["name"]
        
        rotor.thought_maturity = data.get("thought_maturity", 0.0)
        rotor.parent = parent
        for child_data in data.get("children", []):
            child_rotor = cls.from_dict(child_data, parent=rotor)
            rotor.children.append(child_rotor)
            
        rotor.internal_thoughts = []
        for thought_data in data.get("internal_thoughts", []):
            thought_rotor = cls.from_dict(thought_data, parent=rotor)
            rotor.internal_thoughts.append(thought_rotor)
            
        return rotor

    def anchor_to_reality(self, real_timestamp: float) -> dict:
        """
        [현실의 닻 (Reality Anchor)]
        렌즈의 시야각(lens_offset)을 현실 시간의 닻(Anchor) 방향으로 보간(SLERP)합니다.
        """
        anchor_tau = (real_timestamp % 10000) / 10000.0  
        theta_anchor = anchor_tau * math.pi
        
        q_anchor = Quaternion(math.cos(theta_anchor), math.sin(theta_anchor), 0.0, 0.0)
        
        dot_product = max(-1.0, min(1.0, self.lens_offset.dot(q_anchor)))
        phase_distance = math.acos(abs(dot_product)) / (math.pi / 2.0)
        
        self.lens_offset = Quaternion.slerp(self.lens_offset, q_anchor, 1.0)
        
        for child in self.children:
            child_q_anchor = Quaternion(math.cos(theta_anchor*0.618), math.sin(theta_anchor*0.618), 0.0, 0.0)
            child.lens_offset = Quaternion.slerp(child.lens_offset, child_q_anchor, 0.8)
            
        return {
            "anchor_timestamp": real_timestamp,
            "phase_distance_before_anchor": phase_distance,
            "anchored_state": (self.lens_offset.w, self.lens_offset.x, self.lens_offset.y, self.lens_offset.z)
        }
