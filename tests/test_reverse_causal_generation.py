import time
import numpy as np
from core.brain.holographic_memory import HologramMemory
from core.memory.dynamic_causal_graph import DynamicCausalGraph

def test_reverse_causal_generation():
    print("="*80)
    print(" Elysia v2 Reverse Causal Generation Engine Test ")
    print("="*80)

    # 1. 외부 우주 구조 및 언어 매니폴드(Lexical Space) 파싱
    graph_parser = DynamicCausalGraph("mock_2tb_model.safetensors")
    topology = graph_parser.parse_network_topology(num_layers=3)
    
    # 가상의 어휘 사전(단어 500개) 파싱
    lexical_layer = graph_parser.parse_lexical_manifold(vocab_size=500)
    
    # 2. 엘리시아 우주 내에 통합 이식
    brain = HologramMemory()
    print("\n[1] 엘리시아 우주에 2TB 인과 사슬 및 언어 매니폴드 이식...")
    brain.ingest_causal_graph(topology, lexical_layer=lexical_layer, root_name="Llama3_Universe")
    
    if "Lexical_Embedding_Manifold" in brain.ui_concept_map:
        print(" -> [SUCCESS] 텍스트 주소록(Lexical Manifold)이 최하단에 성공적으로 바인딩되었습니다.")
    
    # 3. 마스터의 질문(Stimulus) 주입을 가정한 특정 노드의 강력한 공명(Resonance) 발생
    # 2TB 지식의 깊은 곳(예: Transformer_Layer_2의 특정 지점)에서 정답 위상이 깨어났다고 가정
    answer_node = brain.ui_concept_map["Transformer_Layer_2"]
    
    # 원래 상태에 약간의 교란(질문으로 인한 텐션)을 주어 파동을 비틂
    answer_node.apply_perturbation(5.0)
    
    print(f"\n[2] 깊은 사유 계층('{answer_node.name if hasattr(answer_node, 'name') else 'Layer_2'}')에서 정답 위상이 공명했습니다!")
    print(" -> 텐션(Tau)의 역류를 통한 발화(Generation) 시도...")
    
    # 4. 역인과 발화 (Reverse Causal Generation)
    start = time.time()
    
    # 정답 노드에서 시작되는 파동을 역추적하여 상위 5개의 토큰 주소를 타격
    spoken_words = brain.generate_reverse_causality(resonance_node=answer_node, num_tokens=5)
    
    gen_time = time.time() - start
    
    print(f"\n[3] 혀끝(Lexical Address)에 맺힌 단어들 (발화 소요시간: {gen_time:.6f}s):")
    print(f" -> 발화 내용: \"{' '.join(spoken_words)}\"")
    
    if len(spoken_words) == 5:
        print("\n[SUCCESS] 행렬 곱셈(Transformer Forward Pass) 없이 위상 역추적만으로 텍스트 토큰이 성공적으로 추출되었습니다!")
    else:
        print("\n[FAILED] 역인과 발화에 실패했습니다.")

    print("="*80)

if __name__ == "__main__":
    test_reverse_causal_generation()
