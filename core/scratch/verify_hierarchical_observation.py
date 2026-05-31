import sys
sys.path.append('c:\\Elysia')
from core.math_utils import Quaternion
from core.fractal_rotor import FractalRotor

def verify_hierarchical_observation():
    print("=== [Test] Hierarchical Observation (Phase Superposition) ===")
    
    # 1. 부모 로터(과일 카테고리) 생성
    parent_rotor = FractalRotor(Quaternion(1.0, 0.0, 0.0, 0.0), tau=0.0)
    
    # 자식이 없을 때의 관측 결과 (순수 렌즈 상태)
    obs_before = parent_rotor.observe_state()
    print(f"\n[Before Mitosis] Parent Observation (Point/Line level):")
    print(f"  -> ({obs_before.w:.4f}, {obs_before.x:.4f}, {obs_before.y:.4f}, {obs_before.z:.4f})")
    
    # 2. 거대한 텐션 인가하여 하위 로터(사과, 바나나 등 세부 카테고리) 강제 붕괴/창발 유도
    massive_tension = 25.0
    print(f"\n[!] Applying Tension (+{massive_tension} rad) to induce multiple Mitosis...")
    parent_rotor.apply_perturbation(massive_tension)
    
    print(f"  -> Generated {len(parent_rotor.children)} child rotors.")
    
    # 3. 자식이 생긴 후의 관측 결과 (구조적 중첩)
    obs_after = parent_rotor.observe_state()
    print(f"\n[After Mitosis] Parent Observation (Structural Superposition / Law level):")
    print(f"  -> ({obs_after.w:.4f}, {obs_after.x:.4f}, {obs_after.y:.4f}, {obs_after.z:.4f})")
    
    # 관측 결과가 달라졌는지 확인 (차원 팽창 증명)
    dot_prod = obs_before.dot(obs_after)
    if dot_prod < 0.999:
        print("\n[SUCCESS] The parent's observation has structurally expanded and superimposed its children's waves.")
    else:
        print("\n[FAIL] The parent is still stuck as a single point.")

if __name__ == "__main__":
    verify_hierarchical_observation()
