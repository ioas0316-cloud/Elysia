import sys
sys.path.append(r'c:\Elysia')

from core.brain.omni_modal_em_forge import OmniModalElectromagneticForge

def test():
    print("=" * 60)
    print("   옴니-모달 전자기장 실험                        ")
    print("   (텍스트 x 음악 x 색채 x 수학의 같음과 다름)    ")
    print("=" * 60)
    
    forge = OmniModalElectromagneticForge()
    
    # === 1. 텍스트 투입 ===
    print("\n[투입 1: 텍스트]")
    forge.inject_text("하나님", "하나님")
    forge.inject_text("창조", "창조")
    forge.inject_text("빛", "빛")
    forge.inject_text("어둠", "어둠")
    forge.inject_text("하늘", "하늘")
    forge.inject_text("땅", "땅")
    print("  -> 하나님, 창조, 빛, 어둠, 하늘, 땅")
    
    # === 2. 음악 투입 (주파수 Hz) ===
    print("\n[투입 2: 음악 (주파수)]")
    forge.inject_music("도(C4)", 261.63)
    forge.inject_music("미(E4)", 329.63)    # 장3도 (도와 협화)
    forge.inject_music("솔(G5)", 392.00)    # 완전5도 (도와 강한 협화)
    forge.inject_music("파#(F#4)", 369.99)  # 도와 불협화 (증4도/악마의 음정)
    forge.inject_music("라(A4)", 440.00)    # 기준음
    forge.inject_music("시(B4)", 493.88)
    print("  -> 도, 미, 솔, 파#, 라, 시")
    
    # === 3. 색채 투입 (RGB) ===
    print("\n[투입 3: 색채 (RGB)]")
    forge.inject_color("빨강", 255, 0, 0)      # Hue 0°
    forge.inject_color("주황", 255, 165, 0)     # Hue ~39°
    forge.inject_color("초록", 0, 255, 0)       # Hue 120°
    forge.inject_color("파랑", 0, 0, 255)       # Hue 240°
    forge.inject_color("청록", 0, 255, 255)     # Hue 180° (빨강의 보색)
    forge.inject_color("보라", 128, 0, 128)     # Hue 300°
    print("  -> 빨강, 주황, 초록, 파랑, 청록, 보라")
    
    # === 4. 수학 투입 ===
    print("\n[투입 4: 수학 (피보나치 수열 & 소수)]")
    forge.inject_math("피보나치_1", 1)
    forge.inject_math("피보나치_2", 1)
    forge.inject_math("피보나치_3", 2)
    forge.inject_math("피보나치_5", 5)
    forge.inject_math("소수_7", 7)
    forge.inject_math("소수_11", 11)
    forge.inject_math("소수_13", 13)
    forge.inject_math("원주율_π", 3.14159 * 100)  # 314.159°
    print("  -> 피보나치(1,1,2,5), 소수(7,11,13), 원주율(π)")
    
    # === 5. 전자기장 관측 ===
    forge.observe_universe()
    
    # === 6. 궤적 압축 (무의식화) ===
    forge.compress_to_rotor()

if __name__ == "__main__":
    test()
