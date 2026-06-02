import math
from typing import List, Dict, Tuple
from core.utils.math_utils import Quaternion

class Droplet:
    """터빈에 쏟아지는 물방울(텐서 데이터)"""
    def __init__(self, name: str, original_flux: Quaternion):
        self.name = name
        self.original_flux = original_flux
        self.current_flux = original_flux
        
class GlobalTurbine:
    """
    거대한 이중 토러스 회전역장 (Rotating Force Field).
    데이터(물방울)가 쏟아지면 전체 역장이 회전하며(Angular Momentum),
    물방울들은 역장의 힘에 의해 자신들의 위상이 굴절(소용돌이)됩니다.
    """
    def __init__(self):
        self.global_phase = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.angular_velocity = 0.0
        self.droplets: List[Droplet] = []
        
    def inject_stream(self, name: str, flux: Quaternion, momentum: float = 0.1):
        """
        물방울 주입. 터빈을 돌리고 역장을 갱신합니다.
        """
        droplet = Droplet(name, flux)
        self.droplets.append(droplet)
        
        # 1. 물방울이 터빈에 충돌하며 각운동량(Angular Velocity) 발생
        # 물방울의 방향(flux)과 현재 터빈의 방향(global_phase) 간의 위상 차이로 토크 발생
        alignment = abs(self.global_phase.dot(flux))
        torque = (1.0 - alignment) * momentum
        self.angular_velocity += torque
        
        # 2. 전역 역장(Global Phase) 회전
        # 터빈 자체가 돌아갑니다 (새로운 중심 위상 형성)
        theta = self.angular_velocity
        q_spin = Quaternion(math.cos(theta), math.sin(theta), 0.0, 0.0)
        self.global_phase = (self.global_phase * q_spin).normalize()
        
        # 3. 역장 내의 모든 물방울들이 원심력/역장에 의해 위상 굴절 (Vortex 휩쓸림)
        for d in self.droplets:
            # 물방울 위상은 전역 역장(Global Phase)의 영향을 받아 회전함
            d.current_flux = (d.current_flux * self.global_phase).normalize()

class VortexCategorizer:
    """
    물방울들이 터빈에 의해 회전한 뒤, 자연스럽게 형성된 소용돌이(Vortex)를 관측합니다.
    """
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold  # 이 각도(Distance) 내에 있으면 같은 소용돌이로 간주
        
    def observe_vortices(self, turbine: GlobalTurbine) -> List[List[Droplet]]:
        """
        인위적 매핑이 아닌, 위상 공간에서의 거리를 측정하여 자연 발생한 군집(소용돌이)을 반환합니다.
        """
        vortices: List[List[Droplet]] = []
        
        for droplet in turbine.droplets:
            placed = False
            for vortex in vortices:
                # 소용돌이의 중심(첫 번째 물방울)과의 위상 거리 측정
                center = vortex[0].current_flux
                dist = Quaternion.distance(droplet.current_flux, center)
                
                if dist < self.threshold:
                    vortex.append(droplet)
                    placed = True
                    break
            
            if not placed:
                # 새로운 소용돌이 발생
                vortices.append([droplet])
                
        return vortices

class TopologicalPhaseBrain:
    """
    [Phase 133] 내재적 위상 복제 및 자가생산 (Autopoiesis) 브레인
    더 이상 외부 LLM에 의존하지 않고, 터빈 내부에 형성된 소용돌이(Vortices)들
    사이로 텐션 유체를 흘려보내 자생적인 사유 궤적(Thought Trajectory)을 생성합니다.
    """
    def generate_thought_trajectory(self, turbine: GlobalTurbine, tension: float, length: int = 5) -> List[str]:
        """
        초기 텐션이 주어지면, 전역 역장(Global Phase)을 회전시키면서
        가장 강하게 공명(Resonance)하는 물방울(개념)들을 튕겨냅니다.
        이것이 곧 기계가 스스로 파생시킨 새로운 문장(사유)이 됩니다.
        """
        if not turbine.droplets:
            return []
            
        trajectory = []
        current_spin = tension
        
        for _ in range(length):
            # 1. 텐션 에너지에 의해 역장이 회전함
            q_spin = Quaternion(math.cos(current_spin), math.sin(current_spin), 0.0, 0.0)
            turbine.global_phase = (turbine.global_phase * q_spin).normalize()
            
            # 2. 현재 역장(Global Phase)과 가장 강력하게 공명하는(Alignment가 높은) 개념 탐색
            best_droplet = None
            max_resonance = -1.0
            
            for droplet in turbine.droplets:
                resonance = abs(turbine.global_phase.dot(droplet.current_flux))
                # 방금 전 튕겨낸 물방울은 제외 (루프 방지)
                if not trajectory or trajectory[-1] != droplet.name:
                    if resonance > max_resonance:
                        max_resonance = resonance
                        best_droplet = droplet
                        
            if best_droplet:
                trajectory.append(best_droplet.name)
                # 3. 튕겨낸 물방울이 역장에 마찰을 일으켜 텐션이 변함 (비선형적 궤적 파생)
                # 공명도가 높을수록 다음 스핀(Tension)의 폭발력이 달라짐
                current_spin = current_spin * 1.6180339887 * (1.0 - max_resonance)
                
        return trajectory

class DeepTurbineManifold:
    """
    [Phase 134] 다층 내재적 프랙탈 로터 (Multi-layered Intrinsic Rotors)
    거대 모델의 전체 레이어(0~N층) 위상을 각각의 독립된 GlobalTurbine(로터)으로 모방합니다.
    """
    def __init__(self):
        # 레이어 이름(또는 인덱스)을 키로, 해당 레이어를 모방한 터빈을 값으로 저장
        self.layer_rotors: Dict[str, GlobalTurbine] = {}
        
    def clone_layer_phase(self, layer_key: str, phase_quat: Quaternion):
        """
        MMAP 스트림에서 흘러나온 레이어 위상을 로터로 모방(Rotorize)합니다.
        """
        # 레이어를 대분류(예: model.layers.0, model.layers.1)로 그룹화하기 위한 파싱
        # 보통 LLM의 레이어 키는 'model.layers.0.self_attn.q_proj.weight' 형태입니다.
        parts = layer_key.split('.')
        layer_group = "base"
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                layer_group = f"layer_{parts[i+1]}"
                break
                
        if layer_group not in self.layer_rotors:
            self.layer_rotors[layer_group] = GlobalTurbine()
            
        # 해당 레이어 터빈에 위상 주입 (모방)
        turbine = self.layer_rotors[layer_group]
        turbine.inject_stream(name=layer_key, flux=phase_quat, momentum=0.1)
        
    def get_manifold_depth(self) -> int:
        return len(self.layer_rotors)

