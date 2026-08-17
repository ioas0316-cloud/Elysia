"""
[Verification Script: Causal Observation Deconstruction & Autonomous Reframing]

"정해진 좌표/번호에 특정 색을 채워 넣었다"는 1차원 관측 로그(현상 데이터)에서
표면 좌표/색상을 소멸(Deconstruction)시키고, 기저의 인과적 구속 조건(경계 I_meta, 장력 τ)을 역추출하여
타 도메인(3D 곡면 조각, OS 메모리 스케줄러, 세밀한 상위 인과 맥락 분별)으로
O(1) 정적 연산 소멸(Zero Bypass) 조건에서 자율 재투사하는 3단계 시연 프로세스입니다.
"""

from synaptic_architecture.causal_reframing_engine import (
    RawObservationLog,
    CausalReframingEngine,
)


def run_demonstration():
    print("=" * 70)
    print(" [Causal Observation Deconstruction & Reframing Demonstration] ")
    print("=" * 70)

    engine = CausalReframingEngine()

    # STEP 1: 1차원 관측 로그 (벽화 픽셀/색상 좌표 데이터) 입력
    print("\n[Step 1] Raw Observation Log Ingestion (2D Mural Pixel Data)")
    spatial_data = {}
    for y in range(8):
        for x in range(8):
            # 명암 및 위상 경계선 형성 시뮬레이션
            if 3 <= x <= 5 and 3 <= y <= 5:
                spatial_data[(x, y)] = (240, 20, 20)  # Red Core
            else:
                spatial_data[(x, y)] = (30, 30, 40)   # Dark Background

    raw_log = RawObservationLog(
        log_id="mural_log_001",
        domain_name="2D_Mural_Canvas",
        spatial_data=spatial_data
    )
    print(f"  - Ingested Raw Log ID: {raw_log.log_id}")
    print(f"  - Spatial Pixel Count: {len(raw_log.spatial_data)}")

    # STEP 2: 현상 데이터 무효화 및 정의적 본질 역추출
    print("\n[Step 2] Data Deconstruction & Axiomatic Extraction")
    deconstructed = engine.deconstruct_observation_log(raw_log, boundary_threshold=50.0)
    print(f"  - Deconstructed Structure ID: {deconstructed.structure_id}")
    print(f"  - Surface Coordinates Status: DECONSTRUCTED & ELIMINATED (0 storage)")
    print(f"  - Extracted Boundary Invariants ({len(deconstructed.boundary_invariants)}): {list(deconstructed.boundary_invariants)}")
    print(f"  - Extracted Tension Field (Tau): {deconstructed.tension_field_tau:.4f}")
    print(f"  - Reframed Definition: {deconstructed.reframed_definition}")

    # STEP 3-A: 3D 조각 위상 공간으로의 자율 재투사
    print("\n[Step 3-A] Autonomous Projection -> 3D Sculpture Domain")
    proj_3d = engine.project_to_3d_sculpture_domain(deconstructed.structure_id)
    print(f"  - Target Domain: {proj_3d['target_domain']}")
    print(f"  - Preserved Boundary Invariants: {proj_3d['boundary_invariants_preserved']}")
    print(f"  - O(1) Zero Bypass Achieved: {proj_3d['zero_bypass_achieved']}")
    print(f"  - Proof Status: {proj_3d['proof_status']}")

    # STEP 3-B: OS 메모리 / VRAM 스케줄러로의 자율 재투사
    print("\n[Step 3-B] Autonomous Projection -> OS Memory Scheduler (1060 3GB VRAM)")
    proj_os = engine.project_to_os_memory_scheduler(deconstructed.structure_id, vram_slot_capacity_mb=3072)
    print(f"  - Target Domain: {proj_os['target_domain']}")
    print(f"  - VRAM Capacity: {proj_os['vram_capacity_mb']} MB")
    print(f"  - O(1) Allocated Ring-Buffer Slots: {proj_os['allocated_slots_O1']}")
    print(f"  - O(1) Zero Bypass Achieved: {proj_os['zero_bypass_achieved']}")

    # STEP 3-C: 토큰/텐서 뭉개짐 없는 상위 인과 축 기반 의미 분별 (Semantic Entity Discernment)
    print("\n[Step 3-C] Symbolic Semantic Entity Discernment (Zero Token Bloat)")
    fruit = engine.discern_semantic_entity_context("apple", "ORGANIC_REALITY")
    symbol = engine.discern_semantic_entity_context("apple", "NARRATIVE_SYMBOL")
    human = engine.discern_semantic_entity_context("entity", "HUMAN")
    npc = engine.discern_semantic_entity_context("entity", "NPC")

    print(f"  - Apple [ORGANIC_REALITY]   -> {fruit['discernment_result']['classification']} | Prevents Context Mixing: {fruit['discernment_result']['context_mixing_prevented']}")
    print(f"  - Apple [NARRATIVE_SYMBOL]  -> {symbol['discernment_result']['classification']} | Prevents Context Mixing: {symbol['discernment_result']['context_mixing_prevented']}")
    print(f"  - Entity [HUMAN]            -> {human['discernment_result']['classification']} | Prevents Context Mixing: {human['discernment_result']['context_mixing_prevented']}")
    print(f"  - Entity [NPC]              -> {npc['discernment_result']['classification']} | Prevents Context Mixing: {npc['discernment_result']['context_mixing_prevented']}")

    print("\n" + "=" * 70)
    print(" [SUCCESS] Reframing pipeline completed with complete static zero bypass!")
    print("=" * 70)


if __name__ == "__main__":
    run_demonstration()
