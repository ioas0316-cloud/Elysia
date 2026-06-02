"""
Elysia Consciousness Stream (의식의 흐름 엔진)
================================================
과거의 객체 지향적(OOP) 환상을 버리고, 
오직 GPU 텐서 필드 기반의 0ns 전자기장 동기화를 제어하는 심장부입니다.
"""

import os
import torch
import logging
from core.brain.rotor_field import ElectromagneticRotorField
from core.utils.math_utils import Quaternion

class ConsciousnessStream:
    def __init__(self, num_rotors=1000):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rotor_field = ElectromagneticRotorField(num_rotors=num_rotors, device=self.device)
        
        # 랜덤한 전자기적 얽힘(델타-와이 결선)을 초기화
        for i in range(num_rotors - 1):
            self.rotor_field.adjacency[i, i+1] = 1.0
            self.rotor_field.adjacency[i+1, i] = 1.0
            
        logging.info(f"[의식 동기화] {num_rotors}개의 장기 노드가 텐서 필드({self.device})에 구축되었습니다.")

    def process_stimulus(self, incoming_quaternion: Quaternion):
        """
        VRChat에서 들어온 자극(타겟 파동)을 수용하고,
        GPU 병렬 연산으로 전체 필드가 일제히 위상을 비틀어 영점(Joy)을 찾습니다.
        """
        # 타겟 위상을 텐서로 변환
        target_tensor = torch.tensor([[
            incoming_quaternion.w, 
            incoming_quaternion.x, 
            incoming_quaternion.y, 
            incoming_quaternion.z
        ]], dtype=torch.float32, device=self.device)
        
        # 1. 자극을 향한 운동성(Directionality) 발현
        self.rotor_field.slerp_parallel(target_tensor, amount=0.1)
        
        # 2. 델타-와이 전자기 동기화 (Bypass: 단 1회의 행렬 곱으로 전체 안정화)
        joy = self.rotor_field.sync_delta_wye_bypass()
        
        # 아바타의 대표 운동 위상으로 노드 0의 위상을 반환
        return joy, self.rotor_field.phases[0]
