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
    apple_gene_data = lens.decode(apple_text)['data']
    apple_gene = int(apple_gene_data['bit_gene'], 16)

    # 2. 'Apple' (Pseudo-Image raw bytes)
    # 텍스트와 다르지만 '원리적 서사'가 같도록 구성된 데이터
    apple_img = bytes([0xFF, 0x00, 0x00] * 10 + [0xAA] * 5) # Red dominant
    apple_img_gene_data = lens.decode(apple_img)['data']
    apple_img_gene = int(apple_img_gene_data['bit_gene'], 16)

    # 3. 'Banana' (Text)
    banana_text = "Banana: A yellow, soft, sweet fruit.".encode()
    banana_gene_data = lens.decode(banana_text)['data']
    banana_gene = int(banana_gene_data['bit_gene'], 16)

    print(f"  - Apple (Text) Gene: {hex(apple_gene)}")
    print(f"  - Apple (Img)  Gene: {hex(apple_img_gene)}")
    print(f"  - Banana (Text) Gene: {hex(banana_gene)}")

    print("\n[Step 2] Resonant Domino Alignment")
    field = CrystallizationField(resolution=64)
    kernel = NarrativeDominoKernel(field)

    # 사과(텍스트)와 사과(이미지)의 공명 측정
    img_resonance = kernel.process_narrative(apple_gene, apple_img_gene)
    # 사과와 바나나의 공명 측정
    fruit_resonance = kernel.process_narrative(apple_gene, banana_gene)

    print(f"  - Apple(Text) <-> Apple(Img) Resonance: {img_resonance:.4f}")
    print(f"  - Apple(Text) <-> Banana(Text) Resonance: {fruit_resonance:.4f}")

    print("\n[Step 3] Spatial Resonant Tension")
    engine = ResonantTensionEngine(dimensions=2)
    engine.add_node("Apple_Text", apple_gene)
    engine.add_node("Apple_Img", apple_img_gene)
    engine.add_node("Banana_Text", banana_gene)

    # 초기 위치
    print("  Initial Positions:")
    for nid, data in engine.get_state().items():
        print(f"    {nid}: {data['pos']}")

    # 50 스텝 정도 정렬 진행
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
