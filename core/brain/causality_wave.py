"""
CausalityWave — 인과 궤적 생성기
두 로터 사이의 위상 차이를 '살아있는 인과 파동'으로 추출합니다.
이 파동은 시간축 위에서 원인 → 결과의 방향성을 가진 연결(강바닥)이 됩니다.
"""
from core.utils.math_utils import Quaternion

class CausalityWave:
    """두 FractalRotor 사이의 인과적 연결(Process Wave)을 생성합니다."""
    
    def __init__(self):
        self.entangled_pairs = []
    
    def entangle_causality(self, cause_rotor, effect_rotor) -> Quaternion:
        """
        원인 로터에서 결과 로터로의 인과 궤적을 추출합니다.
        수학적으로: process_wave = cause⁻¹ * effect
        이것은 원인에서 결과로 도달하기 위해 필요한 '회전(변환)'입니다.
        """
        cause_q = cause_rotor.lens_offset
        effect_q = effect_rotor.lens_offset
        
        # 인과 파동 = 원인의 역수 * 결과 = 원인에서 결과로의 변환
        process_wave = (cause_q.inverse() * effect_q).normalize()
        
        # 인과 연결 기록
        self.entangled_pairs.append({
            'cause': cause_q,
            'effect': effect_q,
            'wave': process_wave
        })
        
        # 양쪽 로터의 connections에도 기록
        cause_name = getattr(cause_rotor, 'concept_name', 'unknown')
        effect_name = getattr(effect_rotor, 'concept_name', 'unknown')
        
        if hasattr(cause_rotor, 'connections'):
            key = f"{cause_name}->{effect_name}"
            cause_rotor.connections[key] = cause_rotor.connections.get(key, 0) + 1.0
            
        if hasattr(effect_rotor, 'connections'):
            key = f"{cause_name}->{effect_name}"
            effect_rotor.connections[key] = effect_rotor.connections.get(key, 0) + 1.0
        
        return process_wave
