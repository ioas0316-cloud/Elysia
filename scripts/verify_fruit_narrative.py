"""
Verification Script: The Fruit Narrative (Breaking the Tensor Prison)
====================================================================
마스터의 '사과와 바나나' 예시를 공학적으로 증명합니다.
텐서 연산 없이 비트-유전자의 공명만으로 '과일'이라는 상위 서사에서
데이터가 즉각적으로 정렬되는 도미노 현상을 확인합니다.
"""

import numpy as np
from core.lens.discovery_lens import NarrativeDiscoveryLens
from synaptic_architecture.narrative_domino import NarrativeDominoKernel
from synaptic_architecture.field import CrystallizationField
from core.physics.resonant_tension_engine import ResonantTensionEngine

def verify_fruit_resonance():
    print("\n[Step 1] Narrative Parsing (Text & Pseudo-Image)")
    lens = NarrativeDiscoveryLens()

    # 1. 'Apple' (Text)
    apple_text = "Apple: A red, sweet, crunchy fruit.".encode()
    apple_res = lens.decode(apple_text)['data']

    # 2. 'Apple' (Pseudo-Image raw bytes)
    apple_img = bytes([0xFF, 0x00, 0x00] * 10 + [0xAA] * 5) # Red dominant
    apple_img_res = lens.decode(apple_img)['data']

    # 3. 'Banana' (Text)
    banana_text = "Banana: A yellow, soft, sweet fruit.".encode()
    banana_res = lens.decode(banana_text)['data']

    print(f"  - Apple (Text) Genes: {apple_res['genes']}")
    print(f"  - Apple (Img)  Genes: {apple_img_res['genes']}")
    print(f"  - Banana (Text) Genes: {banana_res['genes']}")

    print("\n[Step 2] Resonant Domino Alignment")
    field = CrystallizationField(resolution=64)
    kernel = NarrativeDominoKernel(field)

    # 정수 유전자로 변환 (Micro 수준에서 비교)
    def get_micro(res): return int(res['genes']['micro'], 16)
    def get_meso(res): return int(res['genes']['meso'], 16)

    apple_gene = get_micro(apple_res)
    apple_img_gene = get_micro(apple_img_res)
    banana_gene = get_micro(banana_res)

    img_resonance = kernel.process_narrative(apple_gene, apple_img_gene)
    fruit_resonance = kernel.process_narrative(apple_gene, banana_gene)

    print(f"  - Apple(Text) <-> Apple(Img) Resonance: {img_resonance:.4f}")
    print(f"  - Apple(Text) <-> Banana(Text) Resonance: {fruit_resonance:.4f}")

    print("\n[Step 3] Spatial Resonant Tension")
    engine = ResonantTensionEngine(dimensions=2)

    def to_int_genes(res):
        return {k: int(v, 16) for k, v in res['genes'].items()}

    engine.add_node("Apple_Text", to_int_genes(apple_res), "MESO")
    engine.add_node("Apple_Img", to_int_genes(apple_img_res), "MICRO")
    engine.add_node("Banana_Text", to_int_genes(banana_res), "MESO")

    # 초기 위치
    print("  Initial Positions:")
    for nid, data in engine.get_state().items():
        print(f"    {nid}: {data['pos']}")

    # 정렬 진행
    for _ in range(50):
        engine.step(dt=0.1)

    print("\n  Aligned Positions (After Resonant Domino):")
    state = engine.get_state()
    for nid, data in state.items():
        print(f"    {nid}: {data['pos']}")

    # 거리 계산
    def dist(p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    pos_at = state['Apple_Text']['pos']
    pos_ai = state['Apple_Img']['pos']
    pos_bt = state['Banana_Text']['pos']

    print(f"\n  Distance (Apple_Text <-> Apple_Img): {dist(pos_at, pos_ai):.4f}")
    print(f"  Distance (Apple_Text <-> Banana_Text): {dist(pos_at, pos_bt):.4f}")

    if dist(pos_at, pos_ai) < dist(pos_at, pos_bt):
        print("\n[SUCCESS] 서사적 공명이 강력한 개체들이 텐서 연산 없이도 더 가깝게 정렬되었습니다.")
    else:
        print("\n[FAILURE] 서사적 정렬이 의도대로 작동하지 않았습니다.")

if __name__ == "__main__":
    verify_fruit_resonance()
