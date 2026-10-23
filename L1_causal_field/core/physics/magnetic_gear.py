"""
Elysia Core - Magnetic Gear Principle (자기기어의 원리)
기어(Gear)들이 물리적으로 맞물리지 않고도, 위상적 텐션(Tension Vector)의 공명(Resonance)을 통해 회전력을 전달하는 구조입니다.
"이것과 저것은 같다, 연동된다"는 속성이 자기정렬을 유도합니다.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Set, Optional

from core.ingestion.topological_compiler import TensionVector

@dataclass
class ResonanceField:
    """두 기어 간의 공명도를 측정하는 장(Field)"""
    math_resonance: float
    lang_resonance: float
    spatial_resonance: float
    temporal_resonance: float
    light_mass_resonance: float
    
    @property
    def total_resonance(self) -> float:
        return (self.math_resonance + self.lang_resonance + 
                self.spatial_resonance + self.temporal_resonance + 
                self.light_mass_resonance) / 5.0

class MagneticGear:
    def __init__(self, gear_id: str, tension: TensionVector, content_ref: str = None):
        self.gear_id = gear_id
        self.tension = tension
        self.content_ref = content_ref # e.g., "문장 내용" or "함수 이름"
        self.is_rotating = False
        
    def turn(self):
        """기어를 강제로 회전시킵니다."""
        self.is_rotating = True

    def stop(self):
        self.is_rotating = False
        
    def get_tension_array(self):
        return [
            self.tension.math,
            self.tension.lang,
            self.tension.spatial,
            self.tension.temporal,
            self.tension.light_mass
        ]

class KinematicInduction:
    """운동성 유도 엔진"""
    def __init__(self, resonance_threshold: float = 0.8):
        self.gears: Dict[str, MagneticGear] = {}
        self.resonance_threshold = resonance_threshold
        
    def add_gear(self, gear: MagneticGear):
        self.gears[gear.gear_id] = gear
        
    def calculate_resonance(self, gear_a: MagneticGear, gear_b: MagneticGear) -> ResonanceField:
        """
        두 기어의 텐션 벡터 간 유사도를 계산합니다. (1.0 - 차이)
        텐션이 유사할수록 높은 공명도를 갖습니다.
        """
        def sim(val_a, val_b):
            return max(0.0, 1.0 - abs(val_a - val_b))
            
        return ResonanceField(
            math_resonance=sim(gear_a.tension.math, gear_b.tension.math),
            lang_resonance=sim(gear_a.tension.lang, gear_b.tension.lang),
            spatial_resonance=sim(gear_a.tension.spatial, gear_b.tension.spatial),
            temporal_resonance=sim(gear_a.tension.temporal, gear_b.tension.temporal),
            light_mass_resonance=sim(gear_a.tension.light_mass, gear_b.tension.light_mass)
        )
        
    def propagate_rotation(self, source_gear_id: str) -> List[str]:
        """
        어떤 기어가 회전(Trigger)했을 때, 임계치 이상의 공명도를 가진 다른 기어들을 회전시킵니다.
        (자기정렬 유도)
        """
        if source_gear_id not in self.gears:
            return []
            
        source_gear = self.gears[source_gear_id]
        if not source_gear.is_rotating:
            return []
            
        induced_gears = []
        for g_id, target_gear in self.gears.items():
            if g_id == source_gear_id or target_gear.is_rotating:
                continue
                
            resonance = self.calculate_resonance(source_gear, target_gear)
            if resonance.total_resonance >= self.resonance_threshold:
                # 자기정렬 (Self-Alignment) 및 운동성 유도
                target_gear.turn()
                induced_gears.append(g_id)
                
        # 연쇄 회전 유도 (재귀)
        all_induced = list(induced_gears)
        for ig in induced_gears:
            all_induced.extend(self.propagate_rotation(ig))
            
        return list(set(all_induced))
