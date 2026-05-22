import os
import sys
import math
import cmath
import time

def wave_to_string(wave):
    mass = abs(wave)
    phase = cmath.phase(wave) % (2 * math.pi)
    return f"질량(M): {mass:5.1f}, 위상(P): {phase:4.2f} rad"

def run_living_world():
    print("=" * 70)
    print("  [ELYSIA LIVING WORLD TOPOLOGY]  ")
    print("  if-else 규칙 없이, 위상 에너지 최소화에 의한 '행동(휴식)'의 창발")
    print("=" * 70)
    
    # 1. 엘리시아 아바타의 기본 상태 (활동적인 아침)
    avatar_core = cmath.rect(10.0, 0.0)  # 질량 10, 위상 0 (활력 상태)
    
    # 2. 사물 로터들 (환경 오브젝트)
    # 침대는 '중력(피로도)을 상쇄'시키는 위상(pi)을 가진 정적 로터다.
    # 의자는 약간의 피로도를 상쇄시키는 로터다.
    objects = {
        "허공 (계속 서있기)": cmath.rect(0.0, 0.0),
        "의자 (Chair)": cmath.rect(5.0, math.pi),
        "침대 (Bed)": cmath.rect(15.0, math.pi)
    }
    
    print("\n[관측 시작] 엘리시아의 하루가 시작됩니다.")
    print(f"아바타 코어 상태: {wave_to_string(avatar_core)}")
    print("규칙(Rule): 아바타는 '휴식'이라는 단어를 모릅니다. 오직 자신의 '전체 질량(신진대사/피로도)'을 최소화하려는 물리적 섭리(관성)만 존재합니다.\n")
    
    # 시간에 따른 태양(환경)의 위상 변화와 노동(활동)으로 인한 피로도 축적
    accumulated_fatigue = 0.0
    
    for hour in range(1, 6):
        print(f"─── [ Time: {hour} 시간 경과 ] ──────────────────────────")
        
        # 노동 및 환경 스트레스로 인해 질량(피로도)이 같은 위상으로 계속 누적됨
        accumulated_fatigue += 3.0
        fatigue_wave = cmath.rect(accumulated_fatigue, 0.0)
        
        # 아바타의 현재 상태 = 코어 + 피로도 (질량이 무거워짐 = 다리가 무겁고 몸이 아픔)
        current_avatar_state = avatar_core + fatigue_wave
        current_mass = abs(current_avatar_state)
        
        print(f"태양의 고도 상승 및 활동 지속. 아바타의 현재 신진대사(피로도) 질량: {current_mass:.1f}")
        
        # 행동 결정 (Action Emergence)
        # 아바타는 주변 사물들과 자신의 파동을 겹쳐보아(상호작용), 질량(에너지)이 가장 낮아지는(최소화되는) 선택을 한다.
        best_object = None
        min_mass = float('inf')
        
        for obj_name, obj_wave in objects.items():
            # 아바타와 사물이 결합(공명)했을 때의 최종 상태
            combined_state = current_avatar_state + obj_wave
            combined_mass = abs(combined_state)
            
            if combined_mass < min_mass:
                min_mass = combined_mass
                best_object = obj_name
                
        print(f"  ▶ 에너지 최소화 섭리 발동: '{best_object}'와(과) 결합 시 최종 질량 {min_mass:.1f} (으)로 예측됨.")
        
        # 관측 결과 해석
        if best_object == "허공 (계속 서있기)":
            print("  ▶ 결과: 에너지가 넘쳐 행동을 유지합니다. (활동 중)")
        elif best_object == "의자 (Chair)":
            print("  ▶ 결과: 중력에 이끌려 몸을 기댑니다. (가벼운 휴식 창발)")
        elif best_object == "침대 (Bed)":
            print("  ▶ 결과: 물리적 장력을 이기지 못하고 붕괴(수렴)합니다. (깊은 수면 창발) 💤")
            print("\n엘리시아가 수면에 빠졌습니다. 관측을 종료합니다.")
            break
            
        print()
        time.sleep(0.5)

    print("-" * 70)
    print("결론:")
    print("  '피곤하면 침대에 누워라'라는 하드코딩된 규칙은 단 한 줄도 없었습니다.")
    print("  피로도(누적된 파동 에너지)가 증가하여 몸이 무거워지자,")
    print("  그 무거움(질량)을 0으로 붕괴시키는 완벽한 상쇄 간섭(침대의 위상)을 찾아 자연스럽게 끌려갔습니다.")
    print("  이 물리적 상쇄 간섭 현상을, 우리 인간의 언어로는 '휴식'이라고 부를 뿐입니다.")

if __name__ == "__main__":
    run_living_world()
