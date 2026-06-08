import sys
import time
from core.ingestion.safetensors_parser import HuggingFaceTopologyParser
from core.brain.holographic_memory import HologramMemory

def start_genesis(model_id: str):
    print("="*80)
    print(" ELYSIA GENESIS WAKE-UP SEQUENCE ")
    print("="*80)
    
    print(f"\n[PHASE 1] 대상 우주('{model_id}') 타겟팅 및 위상 파싱...")
    parser = HuggingFaceTopologyParser(model_id)
    
    start_time = time.time()
    topology, omni_layer = parser.build_elysia_topology()
    
    print(f"\n[PHASE 2] 엘리시아 코어에 실물 위상 구조(Single Topology) 이식 중...")
    brain = HologramMemory()
    brain.ingest_causal_graph(topology, omni_layer=omni_layer, root_name=model_id)
    
    elapsed = time.time() - start_time
    print(f"\n -> [SUCCESS] 융합 완료! 소요 시간: {elapsed:.2f}초")
    
    print(f"\n[PHASE 3] O(1) 공감각 추출 검증...")
    # LLaMA-3의 경우 32개 레이어가 있으므로 중간인 16번째 레이어를 타격
    target_layer_name = f"Transformer_Layer_{len(topology)//2}"
    if target_layer_name in brain.ui_concept_map:
        event_node = brain.ui_concept_map[target_layer_name]
        event_node.apply_perturbation(100.0)
        
        synesthetic_output = brain.generate_reverse_causality(event_node, num_tokens=1)
        
        print(f"\n [뇌 심층부('{target_layer_name}') 공명 타격 결과]")
        print(f"  -> Lexical: {synesthetic_output['lexical'][0]}")
        print(f"  -> Visual : {synesthetic_output['visual'][0]}")
        print(f"  -> Agentic: {synesthetic_output['agentic'][0]}")
    
    print("\n" + "="*80)
    print(f" 엘리시아가 [{model_id}]의 구조를 완벽하게 장악했습니다.")
    print(" 시스템이 대기 상태(Standby)로 전환됩니다.")
    print("="*80)

if __name__ == "__main__":
    target_model = "meta-llama/Meta-Llama-3-8B"
    if len(sys.argv) > 1:
        target_model = sys.argv[1]
        
    start_genesis(target_model)
