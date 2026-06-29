"""
Verification Script: Scale-Free Narrative (Orchard vs. Factory)
==============================================================
마스터의 가르침대로, 비트 단위의 비교를 넘어 킬로바이트(KB) 객체와
메가바이트(MB) 지형이 어떻게 서로를 정렬하는지 증명합니다.

- Orchard (MACRO): 거대한 식물 생태계 서사
- Apple (MESO): 과일 객체
- Bolt (MESO): 공장 부품 객체
"""

import numpy as np
from core.lens.discovery_lens import NarrativeDiscoveryLens
from core.physics.resonant_tension_engine import HierarchicalResonantEngine

def verify_scale_free_resonance():
    print("\n[Step 1] Multi-Scale Parsing")
    lens = NarrativeDiscoveryLens()

    # 1. Orchard (MACRO - 1MB dummy data with natural patterns)
    orchard_data = (b"Orchard Life, Trees, Sun, Soil. " * 32000)
    orchard_res = lens.decode(orchard_data)['data']

    # 2. Apple (MESO - 1KB text)
    apple_data = (b"Apple: Sweet fruit from a tree. " * 32)
    apple_res = lens.decode(apple_data)['data']

    # 3. Bolt (MESO - 1KB text)
    bolt_data = (b"Bolt: Hexagonal steel fastener. " * 32)
    bolt_res = lens.decode(bolt_data)['data']

    print(f"  - Orchard (MACRO) Genes: {orchard_res['genes']}")
    print(f"  - Apple (MESO) Genes: {apple_res['genes']}")
    print(f"  - Bolt (MESO) Genes: {bolt_res['genes']}")

    print("\n[Step 2] Hierarchical Alignment")
    engine = HierarchicalResonantEngine(dimensions=2)

    # 정수 유전자로 변환
    def to_int_genes(genes_dict):
        return {k: int(v, 16) for k, v in genes_dict.items()}

    engine.add_node("Orchard_Field", to_int_genes(orchard_res['genes']), "MACRO")
    engine.add_node("Apple_Obj", to_int_genes(apple_res['genes']), "MESO")
    engine.add_node("Bolt_Obj", to_int_genes(bolt_res['genes']), "MESO")

    # 초기 상태
    print("  Initial Positions (Random):")
    for nid, data in engine.get_state().items():
        print(f"    {nid}: {data['pos']}")

    # 거시적 중력이 작동하도록 시뮬레이션
    for _ in range(100):
        engine.step(dt=0.05)

    print("\n  Aligned Positions (Macro-Scale Domino):")
    state = engine.get_state()
    for nid, data in state.items():
        print(f"    {nid}: {data['pos']}")

    # 거리 측정
    def dist(p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    pos_orchard = state['Orchard_Field']['pos']
    pos_apple = state['Apple_Obj']['pos']
    pos_bolt = state['Bolt_Obj']['pos']

    d_orchard_apple = dist(pos_orchard, pos_apple)
    d_orchard_bolt = dist(pos_orchard, pos_bolt)

    print(f"\n  Distance (Orchard <-> Apple): {d_orchard_apple:.4f}")
    print(f"  Distance (Orchard <-> Bolt): {d_orchard_bolt:.4f}")

    if d_orchard_apple < d_orchard_bolt:
        print("\n[SUCCESS] '과수원'이라는 거시적 서사가 '사과'를 '볼트'보다 강력하게 끌어당겼습니다.")
        print("비트 레벨의 차이를 넘어, 상위 스케일의 공명이 정렬을 지배합니다.")
    else:
        print("\n[FAILURE] 거시적 정렬이 의도대로 작동하지 않았습니다.")

if __name__ == "__main__":
    verify_scale_free_resonance()
