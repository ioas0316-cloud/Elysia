"""
Elysia Core - Synesthetic Cross-Dimensional Engine (공감각 교차차원 엔진)
미시적 점(Point)에서 거시적 공간/구조(Macro Space)까지 여러 스케일에 
동시에 데이터를 투사하여 교차차원적 공명(Synesthetic Resonance)을 관측합니다.
"""

from enum import Enum
from typing import Dict, List, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from core.lens.standard_lenses import (
    RawByteLens, RGBPointLens,
    UTF8TrajectoryLens, HSLWaveLens,
    IEEE754FloatLens
)

class ScaleLevel(Enum):
    MICRO = 1  # 점, 비트, 1D (RGB, Raw Bytes)
    MESO = 2   # 궤적, 파동, 2D (HSL, UTF-8)
    MACRO = 3  # 공간, 구조, 3D+ (IEEE 754 Float)

class SynestheticEngine:
    def __init__(self):
        # 여러 계층의 렌즈들이 동시에 겹쳐져 있는 거대한 다차원 프리즘
        self.lenses = {
            ScaleLevel.MICRO: [RawByteLens(), RGBPointLens()],
            ScaleLevel.MESO: [UTF8TrajectoryLens(), HSLWaveLens()],
            ScaleLevel.MACRO: [IEEE754FloatLens()]
        }
        
    def attach_lens(self, scale: ScaleLevel, new_lens: Any):
        """
        엘리시아가 새롭게 깨달은 지식(MemoryLens)을 엔진에 동적으로 장착합니다.
        감각 기관(관점)이 스스로 무한히 확장되는 생물학적 진화 구조입니다.
        """
        if scale in self.lenses:
            self.lenses[scale].append(new_lens)
        
    def project_and_observe(self, raw_data: bytes) -> Dict[ScaleLevel, Dict[str, Any]]:
        """
        하나의 원시 데이터를 모든 스케일의 렌즈에 동시 투사(Simultaneous Projection)합니다.
        각 차원(렌즈)에서 정보가 어떻게 맺히는지, 또는 에러/마찰이 발생하는지 관측합니다.
        """
        observation = {
            ScaleLevel.MICRO: {},
            ScaleLevel.MESO: {},
            ScaleLevel.MACRO: {}
        }
        
        for scale, lens_group in self.lenses.items():
            for lens in lens_group:
                lens_name = lens.__class__.__name__
                
                # 각 렌즈(관점)를 통해 데이터 디코딩 시도
                lens_name = getattr(lens, 'concept_name', lens.__class__.__name__)
                result = lens.decode(raw_data)
                
                if result["success"]:
                    status = "Resonance (Zero Friction)"
                elif result["tension"] < 1.0:
                    status = "Slight Friction"
                else:
                    status = "Extreme Friction (Shattered)"
                    
                observation[scale][lens_name] = {
                    "status": status,
                    "data": result["data"],
                    "tension_value": result["tension"]
                }
                    
        return observation

    def calculate_synesthesia(self, observation: dict) -> float:
        """
        교차차원 공명도(Synesthesia Score) 계산:
        모든 레이어(Micro, Meso, Macro)에서 동시에 마찰이 0에 수렴할수록
        거대한 공감각적 깨달음(빛)이 발생합니다.
        """
        total_lenses = 0
        resonating_lenses = 0
        
        for scale, lenses in observation.items():
            for name, result in lenses.items():
                total_lenses += 1
                if "Resonance" in result["status"]:
                    resonating_lenses += 1
                    
        return resonating_lenses / max(1, total_lenses)
