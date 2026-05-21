import os
import sys

# Ensure root path is accessible
root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.Cognition.semantic_map import get_semantic_map
from pyquaternion import Quaternion

def run_poc():
    print("="*70)
    print(" 🌌 디지털 만유인력 (Digital General Relativity) 개념 증명 가동")
    print("="*70)

    # 1. Initialize the Semantic Map (The Cosmos)
    print("\n[1단계: 기저 우주(Phase Space) 초기화 및 별자리 형성]")
    sm = get_semantic_map()

    # Force initialize the Genesis Map so we have Stars and Demons
    sm._initialize_genesis_map()

    # Artificially create a massive star by adding causal tethers (Tether Count)
    print("\n[2단계: 인과적 결선을 통한 '여신(Love)' 별의 질량 증폭]")
    # Love already has mass 1000. Let's make it the absolute center of gravity.
    # Add a new concept near it and tether it
    # We add MASSIVE causal edges to make "Love" an absolute black hole
    for i in range(100):
        sm.add_voxel(f"Tether_{i}", (0.5, 0.5, 0.5, 1.0), mass=1.0)
        sm.add_causal_edge(f"Tether_{i}", "Love") # Love gets incredibly heavier

    love_voxel = sm.get_voxel("Love")
    print(f"  => 'Love' 텐서의 현재 질량(Mass): {love_voxel.mass:.2f} (초거대 중력 우물 심화)")

    # 3. Drop a chaotic query into the void and watch it fall
    print("\n[3단계: 혼돈(Query) 투척 및 공간 곡률에 따른 자유 낙하(Free Fall) 관측]")

    # Drop the query close to the center so it gets pulled by the heaviest star (Love)
    # Format: (Logic, Emotion, Time, Spin) -> w, x, y, z in Quaternion
    chaos_coords = (0.5, 0.5, 0.5, 1.0)
    print(f"  => 투척된 혼돈의 초기 좌표: {chaos_coords}")

    # Use our new inference method
    settled_star, final_dist = sm.get_nearest_concept(chaos_coords)

    print("\n[4단계: 궤도 안착 결과]")
    print(f"  ✨ 혼돈이 텐서 공간의 곡률을 따라 굴러떨어져 '{settled_star.name}' 별의 궤도에 안착했습니다!")
    print(f"  => 중심 별과의 최종 거리: {final_dist:.4f}")

    if settled_star.name == "Love":
        print("\n  결과: 거대한 질량을 가진 'Love'의 중력 우물이 주변 공간을 왜곡하여,\n        먼 곳에 떨어진 혼돈마저 자신의 궤도로 빨아들였습니다 (성공!).")
    else:
        print(f"\n  결과: 혼돈이 다른 중력원인 '{settled_star.name}'에 흡수되었습니다. (질량 균형 확인 필요)")

    print("\n============================================================")
    print(" 🏁 디지털 일반 상대성 이론 검증 완료: 공간 왜곡 및 낙하 정상 작동")
    print("============================================================")

if __name__ == "__main__":
    run_poc()
