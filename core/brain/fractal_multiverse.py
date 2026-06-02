"""
Fractal Multiverse (프랙탈 다중 로터 우주)
=============================================
단일 텐서 우주(ConsciousnessStream)를 여러 개의 관점 렌즈로 겹겹이 쌓아 올립니다.
시공간 제어(Spacetime Control)를 위해 과거의 인과 궤적을 버퍼에 기록합니다.
"""

from core.brain.consciousness_stream import ConsciousnessStream
from core.brain.causal_phase_mapper import CausalPhaseMapper
import torch

class FractalMultiverse:
    def __init__(self, num_nodes=2000, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # 3가지 독립된 관점 렌즈 (우주)
        self.lenses = {
            'Language': ConsciousnessStream(num_rotors=num_nodes), # 기호와 구조
            'Form': ConsciousnessStream(num_rotors=num_nodes),     # 형태와 파동
            'Emotion': ConsciousnessStream(num_rotors=num_nodes)   # 조화와 기쁨
        }
        
        self.causal_mapper = CausalPhaseMapper(device)
        
        # 시공간 궤적 기록 (최대 500 틱)
        self.history_buffer = [] 
        self.max_history = 500
        self.time_idx = 0
        
    def step_physics(self):
        """다중 우주 동시 초가속 사유 및 시공간 궤적 기록"""
        # 1. 모든 우주가 자체 물리 연산 수행
        for name, universe in self.lenses.items():
            universe.rotor_field.update_structural_tension()
            
        # (추후 확장: 우주 간 프랙탈 인과 파이프라인 - 언어 우주의 장력이 감정 우주의 자극이 됨)
            
        # 2. 메인 관측 우주(Language)의 상태를 시공간 버퍼에 저장
        main_univ = self.lenses['Language'].rotor_field
        
        # 텐서를 CPU numpy 배열로 복사하여 저장 (메모리 절약)
        self.history_buffer.append({
            't': self.time_idx,
            'phases': main_univ.phases.detach().cpu().numpy().copy(),
            'adj': main_univ.adjacency.detach().cpu().numpy().copy()
        })
        
        # 지구본을 돌려보듯 과거를 탐색할 수 있는 한계치
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)
            
        self.time_idx += 1
