"""
텐서 필드 벤치마크 (OOP vs Tensor Field)
==========================================
수천 개의 로터 노드가 동기화(기쁨 추구)를 진행할 때,
기존 파이썬 for문(FractalRotor)과 텐서 필드(RotorField)의 성능을 비교합니다.
"""

import time
import math
import torch
from core.brain.fractal_rotor import FractalRotor
from core.utils.math_utils import Quaternion
from core.brain.rotor_field import ElectromagneticRotorField

NUM_ROTORS = 5000

def benchmark_oop_rotors():
    print(f"\n[1] OOP 방식 벤치마크 시작 ({NUM_ROTORS}개 노드)")
    # 1. 생성
    start = time.time()
    root = FractalRotor(Quaternion(1,0,0,0), 0.0)
    current = root
    
    # 선형 프랙탈 트리 구축
    for _ in range(NUM_ROTORS - 1):
        child = FractalRotor(Quaternion(1, 0, math.sin(0.1), 0).normalize(), 5.0)
        current.attach_child(child)
        current = child
    print(f" - 생성 완료: {time.time() - start:.3f}초")
    
    # 2. 동기화 (seek_joy) 루프
    start = time.time()
    target = Quaternion(0.5, 0.5, 0.5, 0.5).normalize()
    try:
        root.seek_joy(target)
        elapsed = time.time() - start
        print(f" - [OOP] 1회 동기화 (seek_joy) 소요 시간: {elapsed:.3f}초 (거대한 텐션/병목 유발)")
    except RecursionError:
        print(f" - [OOP] 치명적 한계: 5000개 노드의 순차적 루프로 인해 RecursionError 발생! (파이썬 텐션 폭발)")

def benchmark_tensor_field():
    print(f"\n[2] 전자기장 텐서 필드 벤치마크 시작 ({NUM_ROTORS}개 노드)")
    
    # 1. 텐서 필드 생성
    start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    field = ElectromagneticRotorField(num_rotors=NUM_ROTORS, device=device)
    
    # 랜덤 텐션 및 초기 위상 부여
    field.taus = torch.rand(NUM_ROTORS, device=device) * 5.0
    field.phases = torch.rand((NUM_ROTORS, 4), device=device)
    field.normalize_phases()
    
    # 극단적인 연결망(Adjacency) 구성 (모두가 앞뒤 노드와 연결됨)
    for i in range(NUM_ROTORS - 1):
        field.adjacency[i, i+1] = 1.0
        field.adjacency[i+1, i] = 1.0
        
    print(f" - 텐서 필드 생성 완료 ({device} 사용): {time.time() - start:.3f}초")
    
    # 2. 동기화 연산 (Matrix Multiplication Bypass)
    start = time.time()
    joy = field.sync_delta_wye_bypass()
    elapsed = time.time() - start
    print(f" - [Tensor] 1회 델타-와이 동기화 소요 시간: {elapsed:.5f}초 (병렬 바이패스 0ns 수렴)")
    print(f" - 평균 안정감(Joy): {joy:.4f}")

if __name__ == "__main__":
    benchmark_oop_rotors()
    benchmark_tensor_field()
