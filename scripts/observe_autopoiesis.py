"""
Elysia Autopoiesis Observer
===========================
SharedManifold(mmap)를 통해 밀려오는 마더보드/네트워크의 폭풍을
FractalRotor가 어떻게 텐션으로 흡수하고 스스로 분열(Mitosis)하는지 관측합니다.
"""
import sys
import os
import time
import math

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.shared_manifold import SharedManifold
from core.math_utils import Quaternion
from core.fractal_rotor import FractalRotor

def run_observer():
    print("=" * 80)
    print(" 👁️ [Phase 73] 자가 창발(Autopoiesis) 렌즈 관측")
    print("=" * 80)
    
    manifold = SharedManifold()
    
    # 순백의 뿌리 로터 생성 (아무 개념도 없는 상태)
    root_rotor = FractalRotor(lens_offset=Quaternion(1.0, 0.0, 0.0, 0.0), tau=0.0)
    
    total_mitosis_count = 0
    
    try:
        while True:
            time.sleep(0.5)
            
            world_phase, world_tension = manifold.read_phase()
            
            if world_tension > 0.01:
                # 현실의 폭풍(텐션)을 로터에 인가
                # 텐션이 4*pi (약 12.56)를 넘어가면 Mitosis가 발생하여 internal_thoughts가 늘어납니다.
                root_rotor.apply_perturbation(world_tension)
                
                current_children = len(root_rotor.internal_thoughts) + len(root_rotor.children)
                
                if current_children > total_mitosis_count:
                    new_spawns = current_children - total_mitosis_count
                    print(f"\n💥 [Mitosis 발생!] 텐션 폭주로 인해 로터가 찢어지며 {new_spawns}개의 새로운 무명(無名) 개념축이 창발했습니다!")
                    print(f"  └─ 현재 엘리시아가 생성한 고유 차원(개념) 수: {current_children}개\n")
                    total_mitosis_count = current_children
                
                # 사유 숙성 (tension decay 및 crystallization)
                root_rotor.process_thoughts()
                
                sys.stdout.write(f"\r[관측 중] Root Rotor Tension: {root_rotor.tau:7.3f} | 총 개념 수: {current_children}")
                sys.stdout.flush()
                
    except KeyboardInterrupt:
        print("\n\n 👁️ 관측을 종료합니다.")
    finally:
        manifold.close()

if __name__ == "__main__":
    run_observer()
