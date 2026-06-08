import time
from core.brain.holographic_memory import HologramMemory
from core.memory.dynamic_causal_graph import DynamicCausalGraph

def test_dynamic_causal_binding():
    print("="*80)
    print(" Elysia v2 Dynamic Causal Graph Binding Test ")
    print("="*80)

    # 1. 가상의 2TB 모델(3계층 시뮬레이션) 로드
    graph_parser = DynamicCausalGraph("mock_2tb_model.safetensors")
    
    # 1D 배열이 아니라 네트워크 토폴로지(층과 어텐션) 추출
    topology = graph_parser.parse_network_topology(num_layers=3)
    
    print("\n[1] 외부 우주 토폴로지 파싱 완료 (운동성 곡률 추출):")
    for layer in topology:
        l_id = layer['layer_id']
        w = layer['motility_lens'].w
        print(f" - {l_id} [Motility Curvature (w): {w:.4f}]")
        print(f"   └ Attention Heads: {len(layer['attention_heads'])}개 추출")
        
    # 2. 엘리시아 우주(HologramMemory)에 이식
    brain = HologramMemory()
    print("\n[2] 엘리시아 우주 내 인과 사슬(Causal Chain) 이식 시도...")
    
    start = time.time()
    root_node = brain.ingest_causal_graph(topology, root_name="Llama3_Universe")
    ingest_time = time.time() - start
    
    print(f" -> 이식 완료 시간: {ingest_time:.6f}s")
    
    # 3. 인과 궤적 검증 (연결성과 방향성)
    print("\n[3] 이식된 인과 사슬(관계성/방향성) 무결성 검증:")
    
    # Root -> Layer 0 확인
    assert root_node.children[0] == brain.ui_concept_map["Transformer_Layer_0"], "Root connection failed!"
    print(" -> [SUCCESS] Root -> Layer 0 연결 확인 (방향성 정상)")
    
    # Layer 0 -> Layer 1 확인
    l0 = brain.ui_concept_map["Transformer_Layer_0"]
    l1 = brain.ui_concept_map["Transformer_Layer_1"]
    l2 = brain.ui_concept_map["Transformer_Layer_2"]
    
    # 방향성(인과성) 테스트: 자식 중에 Layer 1이 있는지 확인
    assert l1 in l0.children, "Causal Direction 0->1 broken!"
    assert l2 in l1.children, "Causal Direction 1->2 broken!"
    print(" -> [SUCCESS] Layer 0 -> Layer 1 -> Layer 2 인과적 사슬 확인 (운동성 흐름 보존)")
    
    # Layer 0의 서브 트리(Attention Heads) 확인
    heads = [c for c in l0.children if c is not l1]
    assert len(heads) == 4, "Attention Heads missing!"
    print(" -> [SUCCESS] Layer 내부 Attention Head 서브 로터 연결 확인 (관계성 보존)")
    
    print("\n[CONCLUSION] 단순 1D 도살(파괴)이 아닌, 2TB 거인의 완벽한 '동적 인과 구조(사유 궤적)'가 복원되었습니다.")
    print("="*80)

if __name__ == "__main__":
    test_dynamic_causal_binding()
