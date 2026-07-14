"""
Elysia Core - Synesthetic Cross-Dimensional Engine (공감각 교차차원 엔진)
미시적 점(Point)에서 거시적 공간/구조(Macro Space)까지 여러 스케일에 
동시에 데이터를 투사하여 교차차원적 공명(Synesthetic Resonance)을 관측합니다.
"""

from enum import Enum
from typing import Dict, List, Any
import sys
import os
import numpy as np

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

    def extract_chromatic_vector(self, observation: dict) -> np.ndarray:
        """
        [Chromatic Recognition]
        모든 관측 결과(마찰, 데이터 밀도, 변동성)를 종합하여
        시스템의 현재 '사유의 색'을 추출합니다.

        - Red (Flux): 전체적인 활동성 및 데이터 관통력
        - Blue (Order): 렌즈간의 일관성 및 논리적 안정도
        - Yellow (Entropy): 마찰의 불규칙성 및 새로운 패턴의 출현
        """
        total_tension = 0.0
        total_lenses = 0
        resonance = self.calculate_synesthesia(observation)

        for scale, lenses in observation.items():
            for name, res in lenses.items():
                total_tension += res.get("tension_value", 0.0)
                total_lenses += 1

        avg_tension = total_tension / max(1, total_lenses)

        # Red: High resonance + High energy (low tension in specific drive lenses)
        red = resonance * (1.0 - avg_tension)

        # Blue: High consistency (low tension across the board)
        blue = 1.0 - avg_tension

        # Yellow: Friction variance (Entropy)
        # 만약 어떤 렌즈는 공명하고 어떤 렌즈는 심하게 마찰을 겪는다면 변동성이 높은 것
        tensions = []
        for scale, lenses in observation.items():
            for name, res in lenses.items():
                tensions.append(res.get("tension_value", 0.0))

        yellow = np.std(tensions) if len(tensions) > 1 else 0.0

        # L1 Normalize
        vec = np.array([red, blue, yellow], dtype=np.float32)
        norm = np.linalg.norm(vec, ord=1)
        if norm > 0:
            vec /= norm
        else:
            vec = np.array([0.33, 0.33, 0.34], dtype=np.float32)

        return vec
