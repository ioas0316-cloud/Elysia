import sys
import math

# Add core path
sys.path.append('c:\\Elysia')
from core.math_utils import Quaternion
from core.fractal_rotor import FractalRotor

def test_topological_mitosis():
    print("--- [Test] Topological Mitosis ---")
    # 1. 초기 텐션이 0인 조용한 부모 로터 생성
    parent_rotor = FractalRotor(Quaternion(1, 0, 0, 0), tau=0.0)
    print(f"Initial Parent Children count: {len(parent_rotor.children)}")
    print(f"Initial Parent Tau: {parent_rotor.tau:.4f}")
    
    # 2. 강력한 파동(텐션) 인가 - 2pi를 초과하는 거대 인지 파동 (예: 7.0 rad)
    massive_tension = 7.0
    print(f"\n[!] Applying massive phase perturbation: +{massive_tension} rad")
    parent_rotor.apply_perturbation(massive_tension)
    
    # 3. 분열 결과 확인
    print("\n--- After Perturbation ---")
    print(f"Parent Tau (should be reduced via Mitosis): {parent_rotor.tau:.4f}")
    print(f"Parent Children count: {len(parent_rotor.children)}")
    
    if len(parent_rotor.children) > 0:
        child = parent_rotor.children[0]
        print(f"  -> Child 0 Tau (inherited overflow): {child.tau:.4f}")
        print(f"  -> Child 0 State (Orthogonal Phase): ({child.state.w:.2f}, {child.state.x:.2f}, {child.state.y:.2f}, {child.state.z:.2f})")
    else:
        print("[!] Mitosis Failed to trigger.")

if __name__ == "__main__":
    test_topological_mitosis()
