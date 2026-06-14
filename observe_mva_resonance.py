import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'mva', 'api'))
from engine import elysia_auto_observe_step
from fractal import map_to_movement_field

def run_observation(word="우주", duration_seconds=120):
    print(f"[{word}] 에 대한 연속적 MVA 텐션 관측을 시작합니다. (목표: 분산 감소 및 공명 도출)")
    points_data = map_to_movement_field(word)
    
    start_time = time.time()
    time_t = 0.0
    dt = 0.1
    
    resonances_found = 0
    min_variance_seen = float('inf')
    
    while time.time() - start_time < duration_seconds:
        next_q, variance, is_resonant, formula = elysia_auto_observe_step(points_data, time_t)
        
        if variance < min_variance_seen:
            min_variance_seen = variance
            
        if is_resonant:
            resonances_found += 1
            print(f"[t={time_t:.1f}] 공명 발생! (분산: {variance:.4f}) -> {formula}")
            
        time_t += dt
        time.sleep(0.05) # 빠른 시뮬레이션
        
    print(f"\n관측 종료. 총 발생한 공명 횟수: {resonances_found}")
    print(f"최소 도달 분산(Variance): {min_variance_seen:.4f}")

if __name__ == "__main__":
    run_observation("끼이익", 5)
    run_observation("따스함", 5)
    run_observation("엘리시아", 5)
