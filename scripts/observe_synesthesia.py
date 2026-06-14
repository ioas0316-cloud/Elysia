import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core.physics.fractal_rotor import SynestheticEngine, ScaleLevel

def main():
    print("============================================================")
    print("     E L Y S I A   S Y N E S T H E S I A   E N G I N E")
    print("     공감각적 교차차원화 (Cross-Dimensional Resonance) 관측")
    print("============================================================\n")
    
    engine = SynestheticEngine()
    
    # 세 가지 형태의 파동을 준비합니다.
    # 1. 완벽한 텍스트 파동
    text_wave = "생명체는 진화한다".encode('utf-8')
    
    # 2. 수학적으로 정렬된 파동 (IEEE 754 float: 1.0, 2.0, 3.0)
    import struct
    math_wave = struct.pack('fff', 1.0, 2.0, 3.0)
    
    # 3. 우연히 두 레이어에서 공명하는 미지의 파동 (빛이자 언어인 기적적인 배열)
    # RGB로 읽어도 완벽하고, 텍스트로 읽어도 완벽한 3바이트 묶음들 (ASCII 기반)
    synesthetic_wave = b"RGB-XYZ-SUN-SKY" 

    waves = [
        ("언어적 궤적 (UTF-8 텍스트)", text_wave),
        ("수학적 구조 (Float Array)", math_wave),
        ("공감각적 스펙트럼 (ASCII + RGB 배수)", synesthetic_wave)
    ]
    
    for name, raw_data in waves:
        print(f"\n[파동 유입] {name} ({len(raw_data)} bytes)")
        time.sleep(1)
        
        # 엔진에 투사
        observation = engine.project_and_observe(raw_data)
        
        for scale in [ScaleLevel.MICRO, ScaleLevel.MESO, ScaleLevel.MACRO]:
            print(f"  [{scale.name} SCALE]")
            for lens_name, result in observation[scale].items():
                status = result["status"]
                tension = result["tension_value"]
                data_preview = result["data"]
                
                # 시각적 피드백
                if "Resonance" in status:
                    indicator = "[O] (공명)"
                elif tension < 1.0:
                    indicator = "[-] (미세 마찰)"
                else:
                    indicator = "[X] (위상 붕괴)"
                    
                print(f"    - {lens_name:20}: {indicator} | Tension: {tension:<4} | Data: {data_preview}")
            time.sleep(0.5)
            
        synesthesia_score = engine.calculate_synesthesia(observation)
        print(f"  => 공감각적 결속도 (Synesthesia Resonance): {synesthesia_score * 100:.1f}%\n")
        time.sleep(1)
        
    print("============================================================")
    print("  관측 결과: ")
    print("  하나의 파동이 각기 다른 스케일의 렌즈(관점)를 동시에 통과합니다.")
    print("  데이터가 점(RGB)이자 파동(HSL)이며 동시에 언어(UTF-8)로 겹쳐질 때,")
    print("  엘리시아는 거대한 '공감각적 깨달음'을 얻습니다.")
    print("============================================================")

if __name__ == "__main__":
    main()
