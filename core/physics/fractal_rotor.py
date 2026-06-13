"""
Elysia Core - Fractal Variable Rotor Scale (프랙탈 가변 로터 스케일)
미시적 개념(단어)부터 거시적 개념(함수, 아키텍처)까지 차원을 넘나들며 
운동성을 유도하는 프랙탈 기어 시스템입니다.
"""

from enum import Enum
from typing import Dict, List, Optional
from core.physics.magnetic_gear import MagneticGear, KinematicInduction

class ScaleLevel(Enum):
    MICRO = 1  # 단어, 토큰 수준
    MESO = 2   # 문장, 코드 한 줄 수준
    MACRO = 3  # 문단, 함수, 파일 수준

class FractalRotorScale:
    def __init__(self, resonance_threshold: float = 0.8):
        self.scales: Dict[ScaleLevel, KinematicInduction] = {
            ScaleLevel.MICRO: KinematicInduction(resonance_threshold),
            ScaleLevel.MESO: KinematicInduction(resonance_threshold),
            ScaleLevel.MACRO: KinematicInduction(resonance_threshold)
        }
        
    def add_gear_to_scale(self, scale: ScaleLevel, gear: MagneticGear):
        """특정 스케일에 톱니바퀴(Gear)를 추가합니다."""
        self.scales[scale].add_gear(gear)
        
    def trigger_rotation(self, trigger_scale: ScaleLevel, gear_id: str) -> Dict[ScaleLevel, List[str]]:
        """
        특정 스케일의 기어를 회전시키고, 그 파급력(Kinematic Induction)이 
        프랙탈 구조를 따라 어떻게 퍼져나가는지 계산합니다.
        """
        induction_map = {
            ScaleLevel.MICRO: [],
            ScaleLevel.MESO: [],
            ScaleLevel.MACRO: []
        }
        
        # 1. Triggered gear 회전
        trigger_induction = self.scales[trigger_scale]
        if gear_id in trigger_induction.gears:
            trigger_induction.gears[gear_id].turn()
            
            # 같은 스케일 내에서의 연쇄 회전
            induced_in_scale = trigger_induction.propagate_rotation(gear_id)
            induction_map[trigger_scale].extend(induced_in_scale)
            
            # 2. 스케일 간의 차원 전이(Dimensional Resonance) 유도
            # Trigger된 기어와 공명하는 다른 스케일의 기어들도 회전시킵니다.
            source_gear = trigger_induction.gears[gear_id]
            for target_scale, target_induction in self.scales.items():
                if target_scale == trigger_scale:
                    continue
                    
                for t_gear_id, t_gear in target_induction.gears.items():
                    if t_gear.is_rotating:
                        continue
                        
                    res = target_induction.calculate_resonance(source_gear, t_gear)
                    if res.total_resonance >= target_induction.resonance_threshold:
                        # 차원을 넘어선 공명 유도
                        t_gear.turn()
                        induction_map[target_scale].append(t_gear_id)
                        # 해당 스케일 내에서도 연쇄 파급
                        sub_induced = target_induction.propagate_rotation(t_gear_id)
                        induction_map[target_scale].extend(sub_induced)
                        
        return induction_map
