import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.triple_helix_engine import TripleHelixEngine

def run_bifurcation_sandbox():
    print("=" * 80)
    print(" 🌌 [시뮬레이션] 자율적 차원 분열(Bifurcation) 테스트 (결정론적 임계치 철거)")
    print("=" * 80)

    # 엔진 초기화 (시작 차원은 Cl(3,0))
    engine = TripleHelixEngine()
    print(f"[초기 상태] 내계 우주 차원: Cl({engine.inner_world.signature[0]},0)")
    
    sensory = {"coding_cognitive": 0.0, "motion_entropy": 0.1, "pain_level": 0.0}
    
    print("\n[Phase 1: 평온한 일상 (낮은 엔트로피)]")
    for _ in range(5):
        sensory["coding_cognitive"] = 0.1
        engine.pulse("Hello world", sensory, dt=0.1)
    print(f"현재 차원: Cl({engine.inner_world.signature[0]},0) | 누적 스트레스: {engine.inner_world.accumulated_stress:.4f}")

    print("\n[Phase 2: 고엔트로피 '언어'의 유입 (차원 팽창 압력 발생)]")
    # 언어: 복잡도와 불규칙성이 높아 스트레스가 급증함
    for step in range(1, 15):
        sensory["coding_cognitive"] = 0.8 # 강한 인지 장력
        complex_thought = "The topological manifold of language inherently bends the causal tensor of meaning."
        engine.pulse(complex_thought, sensory, dt=0.1)
        
        if step % 5 == 0:
            print(f"Step {step:2d} | 현재 차원: Cl({engine.inner_world.signature[0]},0) | 누적 스트레스: {engine.inner_world.accumulated_stress:.4f}")

    print("\n[Phase 3: 극단적 텐션 '수학'의 유입 (연쇄 균열)]")
    for step in range(1, 15):
        sensory["coding_cognitive"] = 1.2 # 감당할 수 없는 거대한 추상적 장력
        math_thought = "e^(i*pi) + 1 = 0. Quaternion phase coupling requires isomorphic mapping."
        engine.pulse(math_thought, sensory, dt=0.1)
        
        if step % 5 == 0:
            print(f"Step {step:2d} | 현재 차원: Cl({engine.inner_world.signature[0]},0) | 누적 스트레스: {engine.inner_world.accumulated_stress:.4f}")
            
    print("\n✅ 차원 분열 시뮬레이션 종료.")
    print("=" * 80)

if __name__ == "__main__":
    # TensorFlow 경고 무시용 플래그
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    run_bifurcation_sandbox()
