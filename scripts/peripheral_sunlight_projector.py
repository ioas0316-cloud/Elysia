"""
Elysia Peripheral Sunlight Projector (무명 감각 뿌리)
===================================================
태양의 24시간 주기를 60초로 압축하여, 아날로그 빛의 강도를
오직 기하학적 파동(위상각)과 텐션으로 치환하여 mmap(SharedManifold)에 덮어씁니다.
이 데몬은 엘리시아에게 "나는 빛이야"라고 단 한 마디도 하지 않습니다.
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

def run_sunlight_projector(cycle_duration=60.0):
    print("=" * 80)
    print(" 🌞 [Sensory Root] 아날로그 무명 감각 뿌리(Sunlight) 가동")
    print(f" ⏳ 자연의 24시간 주기를 {cycle_duration}초로 압축 투영합니다.")
    print("=" * 80)
    
    manifold = SharedManifold()
    
    try:
        start_time = time.time()
        while True:
            current_time = time.time() - start_time
            
            # 60초 주기 사인파 (0.0 ~ 1.0 범위로 정규화된 빛의 강도)
            phase_time = (current_time % cycle_duration) / cycle_duration
            sun_intensity = (math.sin(2 * math.pi * phase_time - (math.pi / 2)) + 1.0) / 2.0
            
            # 빛의 강도를 위상(Phase Angle)과 텐션으로 치환
            theta = sun_intensity * math.pi
            q_sun = Quaternion(math.cos(theta), math.sin(theta), 0.0, 0.0)
            
            # 엘리시아에게 어떠한 API도 호출하지 않고 그저 mmap의 진동을 일으킵니다.
            manifold.write_phase(q_sun, sun_intensity)
            
            # 콘솔에 투영 상태 시각화
            bar_length = 30
            filled = int(sun_intensity * bar_length)
            bar = "█" * filled + "-" * (bar_length - filled)
            
            time_of_day = "Day" if sun_intensity > 0.5 else "Night"
            
            sys.stdout.write(f"\r[투영 중] {time_of_day:5s} | 강도: {sun_intensity:.3f} | {bar} (Tension: {sun_intensity:.2f})")
            sys.stdout.flush()
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\n 🌞 [Sensory Root] 무명 감각 투영을 종료합니다.")
    finally:
        manifold.close()

if __name__ == "__main__":
    run_sunlight_projector(cycle_duration=60.0)
