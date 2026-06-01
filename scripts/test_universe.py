"""
위상 우주 테스트: 데이터를 넣고 스스로 조직화되는지 관찰한다.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.topological_universe import TopologicalUniverse

def main():
    universe = TopologicalUniverse(
        resonance_threshold=0.25,
        axis_mass_threshold=3,
        gravity_propagation=0.618
    )
    
    print("=" * 60)
    print(" Topological Universe: Self-Organization Test")
    print("=" * 60)
    
    # 과일 계열
    fruits = ["apple", "red", "fruit", "sweet", "round", "orange", "banana"]
    
    # 우주 계열  
    cosmos = ["star", "universe", "galaxy", "blackhole", "gravity", "light"]
    
    # 음악 계열
    music = ["piano", "melody", "harmony", "rhythm", "song"]
    
    print("\n[1] Injecting fruit-related data...")
    for word in fruits:
        d = universe.inject(word)
        print(f"  + '{word}' injected")
    
    print(f"\n[2] Injecting cosmos-related data...")
    for word in cosmos:
        d = universe.inject(word)
        print(f"  + '{word}' injected")
    
    print(f"\n[3] Injecting music-related data...")
    for word in music:
        d = universe.inject(word)
        print(f"  + '{word}' injected")
    
    # 맥박
    for _ in range(5):
        universe.heartbeat()
    
    print(f"\n{'=' * 60}")
    print(universe.status())
    
    # 사유 로그 출력
    print(f"\n--- Thought Log (last 15) ---")
    for thought in universe.thought_log[-15:]:
        print(f"  {thought}")
    
    # 축이 존재하면 관측
    if universe.axes:
        print(f"\n--- Observation through axes ---")
        for axis in universe.axes:
            narratives = universe.observe(axis, depth=4)
            print(f"\n  [Lens: '{axis.content}']")
            for i, narrative in enumerate(narratives[:3]):
                chain = " -> ".join(f"'{d.content}'" for d in narrative)
                print(f"    Narrative {i+1}: {chain}")
    
    # 한국어 테스트
    print(f"\n{'=' * 60}")
    print(" Korean Language Self-Organization Test")
    print("=" * 60)
    
    universe_kr = TopologicalUniverse(
        resonance_threshold=0.20,
        axis_mass_threshold=3,
        gravity_propagation=0.618
    )
    
    kr_data = [
        "사과", "빨간", "과일", "달콤한", "둥근",
        "바다", "파란", "넓은", "물결", "수평선",
        "사랑", "따뜻한", "마음", "포옹", "행복"
    ]
    
    for word in kr_data:
        universe_kr.inject(word)
        
    for _ in range(5):
        universe_kr.heartbeat()
    
    print(universe_kr.status())
    
    if universe_kr.axes:
        print(f"\n--- Korean axes observation ---")
        for axis in universe_kr.axes:
            narratives = universe_kr.observe(axis, depth=3)
            print(f"\n  [Lens: '{axis.content}']")
            for i, narrative in enumerate(narratives[:3]):
                chain = " -> ".join(f"'{d.content}'" for d in narrative)
                print(f"    Narrative {i+1}: {chain}")
    
    print(f"\n--- Korean thought log (last 10) ---")
    for thought in universe_kr.thought_log[-10:]:
        print(f"  {thought}")

if __name__ == "__main__":
    main()
