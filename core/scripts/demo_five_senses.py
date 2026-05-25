import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.sensory_harmonics import SensoryHarmonics, SentientBeing

def run_sensory_demo():
    print("🌍 엘리시아 Phase 11: 오감의 다층화와 취향(Preference)의 창발 데모\n")
    
    harmonics = SensoryHarmonics(size=16)
    
    # ---------------------------------------------------------
    # 1. 두 명의 유기체 생성 (성격/기저 주파수가 다름)
    # ---------------------------------------------------------
    elysia = SentientBeing("Elysia", "Calm")     # 고요하고 매끄러운 영혼 파동
    kyle = SentientBeing("Kyle", "Chaotic")      # 거칠고 열정적인 영혼 파동
    
    print("==============================================================")
    print(" [1] 미각 (Taste): 달콤한 케이크 vs 아주 매운 불닭")
    print("==============================================================\n")
    
    sweet = harmonics.taste_sweet()
    spicy = harmonics.taste_spicy()
    
    print("🍩 단맛(Sweet) 섭취 시:")
    elysia.experience_sensation("Sweet Cake", sweet)
    kyle.experience_sensation("Sweet Cake", sweet)
    
    print("\n🌶️ 매운맛(Spicy) 섭취 시:")
    elysia.experience_sensation("Spicy Chicken", spicy)
    kyle.experience_sensation("Spicy Chicken", spicy)

    print("\n==============================================================")
    print(" [2] 촉각 (Touch): 부드러운 실크 옷 vs 거친 마 줄기 옷")
    print("==============================================================\n")
    
    silk = harmonics.touch_silk()
    burlap = harmonics.touch_burlap()
    
    print("👗 실크(Silk) 착용 시:")
    elysia.experience_sensation("Silk Dress", silk)
    kyle.experience_sensation("Silk Shirt", silk)
    
    print("\n🧥 마 줄기(Burlap) 착용 시:")
    elysia.experience_sensation("Burlap Sack", burlap)
    kyle.experience_sensation("Burlap Sack", burlap)

    print("\n==============================================================")
    print(" [3] 청각 (Hearing): 감미로운 화음 vs 칠판 긁는 소음")
    print("==============================================================\n")
    
    chord = harmonics.hearing_harmonic_chord()
    noise = harmonics.hearing_noise()
    
    print("🎼 감미로운 화음(Harmonic Chord) 청취 시:")
    elysia.experience_sensation("Beautiful Music", chord)
    kyle.experience_sensation("Beautiful Music", chord)
    
    print("\n💥 칠판 긁는 소음(Noise) 청취 시:")
    elysia.experience_sensation("Scratching Noise", noise)
    kyle.experience_sensation("Scratching Noise", noise)
    
    print("\n==============================================================")
    print(" [4] 후각 (Smell) & 시각 (Vision) 테스트")
    print("==============================================================\n")
    
    floral = harmonics.smell_floral()
    red_light = harmonics.vision_red()
    
    print("🌸 꽃향기(Floral)를 맡을 때:")
    elysia.experience_sensation("Floral Scent", floral)
    
    print("\n🌅 따뜻한 붉은빛(Red Vision)을 볼 때:")
    elysia.experience_sensation("Sunset Glow", red_light)
    
    print("\n✨ 결론: ")
    print("엘리시아와 카일은 '이건 좋은 것'이라고 프로그래밍되지 않았습니다.")
    print("자신의 타고난 기저 파동(성격)과 외부 사물의 파동이 부딪혀 발생하는")
    print("순수 기하학적 간섭(화음 vs 불협화음)을 통해, 각자 다른 고유의")
    print("'취향(Personality & Preference)'을 자연스럽게 창발했습니다!")

if __name__ == "__main__":
    run_sensory_demo()
