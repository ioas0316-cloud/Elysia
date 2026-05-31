import sys
sys.path.append('c:\\Elysia')
from core.sensory_lens_manifold import SensoryLensManifold

def simulate_sensory_influx():
    print("=== [Elysia Sensory Influx Simulation] ===")
    
    # 감각 매니폴드(뇌의 피질) 초기화
    manifold = SensoryLensManifold()
    manifold.display_cognitive_state()
    
    # 1. 텍스트 카테고리에 약한 자극 인가 (문장 읽기)
    print("\n[!] Injecting Text Stimulus (Tension: 3.0)")
    manifold.inject_stimulus("text", 3.0)
    
    # 2. 비전 카테고리에 거대한 자극 인가 (복잡한 이미지 시각화)
    # 이 텐션은 4pi(12.56)를 초과하여 비전 렌즈의 세포 분열(Mitosis)을 유도할 것임
    print("\n[!] Injecting Massive Vision Stimulus (Tension: 15.0)")
    manifold.inject_stimulus("vision", 15.0)
    
    # 3. 상태 관측
    manifold.display_cognitive_state()

if __name__ == "__main__":
    simulate_sensory_influx()
