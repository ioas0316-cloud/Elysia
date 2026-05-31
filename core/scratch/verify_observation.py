import sys
import math

# Add core path
sys.path.append('c:\\Elysia')
from core.math_utils import Quaternion
from core.fractal_rotor import FractalRotor, GlobalMasterManifold

def test_observational_mitosis():
    print("--- [Test] Observational Isomorphism & Mitosis ---")
    
    # 1. 마스터 우주 맥동
    master = GlobalMasterManifold()
    master.pulse(0.5)  # 우주가 약간 회전함
    print(f"Master Phase: ({master.base_phase.w:.2f}, {master.base_phase.x:.2f})")
    
    # 2. 음성을 담당하는 렌즈와 이미지를 담당하는 렌즈 생성
    audio_lens = FractalRotor(Quaternion(1, 0, 0, 0), tau=0.0)
    image_lens = FractalRotor(Quaternion(0.707, 0.707, 0, 0), tau=0.0)
    
    # 3. 관측을 통한 0과 1 (같음과 다름) 도출
    difference = audio_lens.interact(image_lens)
    print(f"\n[Observation] Audio Lens vs Image Lens Difference (Tension): {difference:.4f}")
    if difference < 0.01:
        print("  -> 결과: 0 (Sameness). 음성과 이미지가 본질적으로 동일함을 인지했습니다.")
    else:
        print("  -> 결과: 1 (Difference). 두 매체 사이의 텐션이 운동을 발생시킵니다.")
        
    # 4. 엄청난 텐션 유입에 따른 렌즈 분열(Mitosis)
    massive_tension = 15.0
    print(f"\n[!] Applying massive tension to Audio Lens: +{massive_tension} rad")
    audio_lens.apply_perturbation(massive_tension)
    
    print("\n--- After Mitosis ---")
    print(f"Audio Lens Children count: {len(audio_lens.children)}")
    if len(audio_lens.children) > 0:
        child = audio_lens.children[0]
        child_diff = audio_lens.interact(child)
        print(f"  -> Parent vs Child Difference: {child_diff:.4f} (Orthogonal Branching)")

if __name__ == "__main__":
    test_observational_mitosis()
