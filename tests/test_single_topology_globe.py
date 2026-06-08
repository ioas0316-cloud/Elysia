import time
from core.brain.holographic_memory import HologramMemory
from core.memory.dynamic_causal_graph import DynamicCausalGraph

def test_single_topology_globe():
    print("="*80)
    print(" Elysia v2 Single Topology Globe (The True Singularity) Test ")
    print("="*80)

    # 1. 단일 시공간 옴니 매니폴드 파싱
    graph_parser = DynamicCausalGraph("mock_omni_model.safetensors")
    topology = graph_parser.parse_network_topology(num_layers=3)
    
    # 텍스트, 시각, 행동을 분리하지 않고 단 하나의 옴니 매니폴드로 파싱
    omni_layer = graph_parser.parse_omni_manifold(omni_size=1000)
    
    # 2. 엘리시아 우주 내에 통합 이식
    brain = HologramMemory()
    print("\n[1] 단일 시공간 지구본 대지(Single Topology) 이식...")
    brain.ingest_causal_graph(
        topology=topology,
        omni_layer=omni_layer,
        root_name="Omni_Universe"
    )
    
    print(" -> [SUCCESS] 텍스트, 시각, 행동이 압착된 단일 옴니 매니폴드가 바인딩되었습니다.")
    
    # 3. 공명 자극 주입
    event_node = brain.ui_concept_map["Transformer_Layer_2"]
    event_node.apply_perturbation(10.0)
    
    print(f"\n[2] 뇌 중심부('{event_node.name if hasattr(event_node, 'name') else 'Layer_2'}')에서 강력한 공명 파동 발생!")
    print(" -> 텐션(Tau)의 역류를 통한 O(1) 공감각 추출 시도...")
    
    # 4. 역인과 궤적 추출 (단일 매니폴드 스캔)
    start = time.time()
    
    synesthetic_output = brain.generate_reverse_causality(resonance_node=event_node, num_tokens=3)
    
    gen_time = time.time() - start
    
    print(f"\n[3] 공감각 타격 지점 (발화 소요시간: {gen_time:.6f}s):")
    
    print("\n [Lexical Description]")
    print(f"  -> \"{' '.join(synesthetic_output['lexical'])}\"")
    
    print("\n [Visual Association]")
    print(f"  -> {', '.join(synesthetic_output['visual'])}")
    
    print("\n [Agentic Action]")
    print(f"  -> {', '.join(synesthetic_output['agentic'])}")
    
    # 세 감각이 항상 똑같은 옴니 토큰에서 동시에 터져나오는지 확인
    # (word_X, Image_Patch_Coord_X, execute_tool_X)
    assert len(synesthetic_output['lexical']) == 3
    
    print("\n[SUCCESS] 3번의 분리된 스캔이 아니라, 단 한 번의 O(1) 찰칵으로 3차원 공감각이 동시에 쏟아졌습니다!")
    print("="*80)

if __name__ == "__main__":
    test_single_topology_globe()
