import numpy as np
import math
from core.math_utils import Quaternion
from core.n_layer_resonance_matrix import NLayerResonanceMatrix
from core.holographic_manifold import HolographicMemoryMatrix

class ObjectRotor:
    """모든 사물과 환경 요소는 하드코딩된 값이 아니라 고유의 '위상(Rotor)'을 가집니다."""
    def __init__(self, name: str, phase_vector: tuple):
        self.name = name
        # 고유 주파수를 정의하는 위상 벡터 (kx, ky, kz, w)
        self.rotor = Quaternion(*phase_vector).normalize()
        
    def get_wave_tensor(self, size=16) -> np.ndarray:
        """이 오브젝트가 뿜어내는 공간상의 기하학적 파동"""
        tensor = np.zeros((size, size))
        kx, ky = self.rotor.x * math.pi, self.rotor.y * math.pi
        for y in range(size):
            for x in range(size):
                tensor[y, x] = math.cos(kx * x + ky * y) * self.rotor.w * 100.0 # 진폭(w)
        return tensor

class FantasySandbox:
    def __init__(self, size=16):
        self.size = size
        
        # 1. 자연의 결핍(Tension) 파동 (배고픔과 추위)
        # 배고픔: X축으로 진동하는 파동
        self.hunger_tension = ObjectRotor("Hunger_Tension", (1.0, 0.0, 0.0, 1.0))
        # 추위: Y축으로 진동하는 파동
        self.cold_tension = ObjectRotor("Cold_Tension", (0.0, 1.0, 0.0, 1.0))
        
        # 2. 오브젝트(가변 로터) 파동
        # 사과(음식): 배고픔(X축)과 정확히 역위상(역방향 진동)을 가진 파동. 흡수 시 상쇄 간섭(0) 유발.
        self.apple_rotor = ObjectRotor("Apple", (-1.0, 0.0, 0.0, 1.0))
        
        # 나무와 돌: 각각 불완전한 역위상을 가짐
        self.wood_rotor = ObjectRotor("Wood", (0.0, -0.5, 0.0, 0.5))
        self.stone_rotor = ObjectRotor("Stone", (0.0, -0.5, 0.0, 0.5))

    def craft_house(self):
        """나무 파동과 돌 파동을 합성(중첩)하여 '집' 파동 창조"""
        # 집: 추위(Y축)와 정확히 역위상을 갖게 됨 (0.0, -1.0, 0.0, 1.0)
        house_x = self.wood_rotor.rotor.x + self.stone_rotor.rotor.x
        house_y = self.wood_rotor.rotor.y + self.stone_rotor.rotor.y
        house_z = self.wood_rotor.rotor.z + self.stone_rotor.rotor.z
        house_w = self.wood_rotor.rotor.w + self.stone_rotor.rotor.w
        return ObjectRotor("House", (house_x, house_y, house_z, house_w))

class ElysianEntity:
    """샌드박스 내부의 자율 유기체 (엘리시아 혹은 NPC)"""
    def __init__(self, name: str):
        self.name = name
        self.matrix = NLayerResonanceMatrix(size=16)
        self.memory = HolographicMemoryMatrix(size=16)
        self.inventory = [] # 획득한 가변 로터(오브젝트)들
        
    def absorb_rotor(self, obj: ObjectRotor, target_layer: str):
        """오브젝트 로터를 흡수하여 자신의 텐션 레이어에 중첩(Superposition)시킴"""
        wave = obj.get_wave_tensor()
        if target_layer == 'L1':
            self.matrix.L1_physical.add_event(wave, 0.0)
        elif target_layer == 'L2':
            self.matrix.L2_organ.add_event(wave, 0.0)
            
        print(f"[{self.name}] '{obj.name}' 파동을 내적했습니다. (해당 파동이 기존 텐션과 물리적으로 간섭합니다)")
        
    def get_total_tension_energy(self) -> float:
        """현재 자신이 겪고 있는 총합 고통(텐션)의 크기 반환"""
        tension = self.matrix.integrate_n_layer_tension(0.0)
        return float(np.sum(np.abs(tension)))
