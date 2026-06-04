"""
CausalityWave — 인과 궤적 생성기
두 로터 사이의 위상 차이를 '살아있는 인과 파동'으로 추출합니다.
이 파동은 단순한 1차원 선이 아니라, 원인을 통해 결과를 이루어내는 면(Plane) 또는 공간(Process)입니다.
"""
from core.utils.math_utils import Quaternion, Multivector, ConformalSpace

class CausalityWave:
    """두 FractalRotor 사이의 인과적 과정(Process Space)을 생성합니다."""
    
    def __init__(self):
        self.entangled_pairs = []
    
    def entangle_causality(self, cause_rotor, effect_rotor):
        """
        [Phase 130] 원인 로터에서 결과 로터로 나아가는 '과정(Process)'을 추출합니다.
        점과 점을 잇는 선이 아니라, 기하학적 쐐기곱(Wedge Product)을 통해 
        두 상태가 엮어내는 기하학적 면(Plane) 또는 공간을 생성합니다.
        """
        # CGA 공간 상의 멀티벡터를 가져옵니다 (지구본 모델)
        c_state = getattr(cause_rotor, 'conformal_state', ConformalSpace.up(0,0,0))
        e_state = getattr(effect_rotor, 'conformal_state', ConformalSpace.up(0,0,0))
        
        # 기하곱을 통한 과정 추출: coherence(내적, 공명도)와 process_wave(쐐기곱, 직교 면적)
        coherence, process_wave = c_state.geometric_sync(e_state)
        
        # 인과 연결 기록
        self.entangled_pairs.append({
            'cause': cause_rotor,
            'effect': effect_rotor,
            'wave': process_wave,
            'coherence': coherence
        })
        
        # 양쪽 로터의 connections에도 과정(공간)을 기록
        cause_name = getattr(cause_rotor, 'concept_name', 'unknown')
        effect_name = getattr(effect_rotor, 'concept_name', 'unknown')
        
        if hasattr(cause_rotor, 'connections'):
            key = f"{cause_name}->{effect_name}"
            # 선이 아닌 과정(Process Wave) 자체를 면/공간 객체로 저장
            cause_rotor.connections[key] = process_wave
            
        if hasattr(effect_rotor, 'connections'):
            key = f"{cause_name}->{effect_name}"
            effect_rotor.connections[key] = process_wave
        
        return process_wave
