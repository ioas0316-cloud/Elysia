"""
위상 우주 공리적 원심분리기 테스트 (Phase 108)
단편적인 데이터뿐만 아니라, 그 데이터가 존재하는 '원리와 정의(Axiom)'를 함께 투입하여
원심분리기 가동 시 카테고리화(가변축 형성)가 어떻게 일어나는지 관측한다.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.topological_universe import TopologicalUniverse

def main():
    universe = TopologicalUniverse(
        resonance_threshold=0.20,  # 다양한 텍스트 길이를 수용하기 위해 임계값을 약간 낮춤
        axis_mass_threshold=3,     
        gravity_propagation=0.618,
        centrifuge_rpm=0.6       # 강력한 회전
    )
    
    print("=" * 70)
    print(" Axiomatic Centrifuge with Principles (원리와 정의 동시 투입)")
    print("=" * 70)
    
    # 1. 공리/원리/정의 데이터 (이것들이 거대한 가변축이 되어야 함)
    axioms = [
        "[수학의 원리] 수학은 숫자, 양, 공간의 구조를 다루는 학문이다. 1, 2, 3과 같은 숫자를 통해 연산한다.",
        "[한글의 원리] 한글은 세종대왕이 발성 기관의 모양을 본떠 만든 과학적인 표음 문자이다. 자음과 모음이 결합한다.",
        "[감정의 원리] 감정은 인간의 마음속에서 일어나는 기쁨, 슬픔, 사랑, 분노 등의 심리적 상태이다.",
        "[프로그래밍 원리] 프로그래밍은 컴퓨터에게 명령을 내리기 위해 코드와 함수, 클래스를 작성하는 논리적 과정이다.",
        "[자연의 원리] 자연은 생명체와 우주, 동물과 식물이 어우러져 살아가는 거대한 물리적, 생물학적 생태계이다."
    ]
    
    # 2. 인스턴스 데이터
    instances = [
        # 수학
        "1", "2", "3", "100", "+", "-", "=", 
        # 한글/감정 (한글의 원리와 감정의 원리 양쪽에 영향을 받을 수 있음)
        "사랑", "슬픔", "기쁨", "행복", "분노",
        # 프로그래밍
        "def", "class", "return", "import", "self", "코드", "함수",
        # 자연/동물
        "강아지", "고양이", "사자", "코끼리", "우주", "생명체"
    ]
    
    # 3. 노이즈 (어느 공리에도 속하지 않는 것들)
    noises = [
        "0x4A2B", "@#$%", "^&*(", "1a2b3c", "____"
    ]
    
    print("[1] Injecting Axioms (원리 데이터 투척)...")
    for ax in axioms:
        universe.inject(ax)
        
    print("\n[2] Injecting Instances & Noises (인스턴스 및 노이즈 투척)...")
    for item in instances + noises:
        universe.inject(item)
            
    print(f"\n[초기 상태 - 원심분리 전]")
    print(f"Total connections: {sum(d.connection_count for d in universe.data) // 2}")
    
    print("\n[3] Activating Centrifuge (원심분리기 가동 - 10회전)...")
    for i in range(10):
        universe.apply_centrifuge()
        
    print(f"\n[최종 상태 - 원심분리 후 관측]")
    print(universe.status())
    
    # 각 공리(Axiom)가 어떤 인스턴스들을 끌어당겼는지 특별 관측
    print("\n--- Axiom Observation (공리별 결속 상태) ---")
    for datum in universe.data:
        if datum.content.startswith("["):
            print(f"\n[{datum.content[:15]}...] (links={datum.connection_count}, gravity={datum.gravity:.3f})")
            if datum.connections:
                sorted_links = sorted(datum.connections.items(), key=lambda x: -x[1])
                for linked, strength in sorted_links[:8]:
                    print(f"  -> '{linked.content}' ({strength:.3f})")
            else:
                print("  -> (No connections survived the centrifuge)")

if __name__ == "__main__":
    main()
