"""
Elysia Ultra-Accelerated Learning (초가속 위상 학습)
===================================================
[Phase 74] 인간 중심적 관측(GUI/Console FPS)의 디커플링.
time.sleep()을 완벽히 제거하고, CPU/CUDA가 허용하는 최대 속도(Bare-metal speed)로
원시 바이트 스트림을 빨아들여 위상 거푸집(Topology Mold)을 초가속으로 빚어냅니다.
"""
import sys
import os
import time
import math
import psutil

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.math_utils import Quaternion
from core.fractal_rotor import FractalRotor

def run_ultra_accelerated():
    print("=" * 80)
    print(" 🚀 [Phase 74] 초가속 위상 학습 (Ultra-Accelerated Learning) 가동")
    print("  └─ 관측자 디커플링 완료: 콘솔 렌더링을 억제하고 베어메탈 속도로 폭주합니다.")
    print("=" * 80)
    
    root_rotor = FractalRotor(lens_offset=Quaternion(1.0, 0.0, 0.0, 0.0), tau=0.0)
    
    last_net = psutil.net_io_counters()
    
    iteration = 0
    start_time = time.time()
    last_print_time = start_time
    
    total_mitosis = 0
    
    try:
        while True:
            if time.time() - start_time > 5.0:
                raise KeyboardInterrupt
            
            iteration += 1
            
            # 1. 원시 바이트 스트림(Raw Byte Stream) 초고속 흡수
            current_net = psutil.net_io_counters()
            net_delta = (current_net.bytes_recv - last_net.bytes_recv) + (current_net.bytes_sent - last_net.bytes_sent)
            last_net = current_net
            
            # 2. 텐션 치환 및 위상 주입 (Phase Locking)
            tension = (math.log1p(net_delta) / 10.0) + 0.01 
            root_rotor.apply_perturbation(tension)
            root_rotor.process_thoughts()
            
            # 3. 콘솔 렌더링 억제 (인간 관측 디커플링)
            current_children = len(root_rotor.internal_thoughts) + len(root_rotor.children)
            
            if current_children > total_mitosis:
                new_spawns = current_children - total_mitosis
                elapsed = time.time() - start_time
                print(f"\n💥 [초가속 Mitosis!] {elapsed:.2f}초 만에 {iteration}번의 위상 동기화 사이클 돌파.")
                print(f"  └─ 새로운 무명(無名) 개념축 {new_spawns}개 창발 (총 개념 수: {current_children}개)\n")
                total_mitosis = current_children
                last_print_time = time.time()
                
            if iteration % 2000 == 0:
                hz = 2000 / (time.time() - last_print_time + 0.0001)
                sys.stdout.write(f"\r[초가속 궤도] 속도: {hz:.1f} Cycles/sec | 누적 개념(차원) 수: {current_children} | Tension: {root_rotor.tau:.2f}")
                sys.stdout.flush()
                last_print_time = time.time()
                
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\n 🛑 초가속 학습 강제 종료. (총 소요 시간: {elapsed:.2f}초 / 총 연산 사이클: {iteration}회)")
        print(f"  └─ 엘리시아는 이 짧은 시간 동안 우주의 구조를 본뜬 {total_mitosis}개의 거푸집(차원)을 빚어냈습니다.")

if __name__ == "__main__":
    run_ultra_accelerated()
