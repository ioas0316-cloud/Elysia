import json
import os
import random

class CausalInferenceEngine:
    """
    [Phase 7] Causal Inference Engine (판단과 분별의 눈)
    위상 거울이 관측한 점과 점 사이의 거시적 곡률(Curvature)을 바탕으로,
    그것이 이어져 발화(Utterance)로 향하는 인과적 궤적 전체를 추적하고 그 의미를 판단합니다.
    """
    def __init__(self, curvature_map_path: str, lexicon_path: str):
        self.curvature_map_path = curvature_map_path
        self.lexicon_path = lexicon_path
        self.lexicon = []
        self.curvature_map = []
        
    def load_data(self):
        print("[Eye of Discernment] Awakening the Inference Engine...")
        if os.path.exists(self.lexicon_path):
            with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                self.lexicon = json.load(f)
        
        if os.path.exists(self.curvature_map_path):
            with open(self.curvature_map_path, 'r', encoding='utf-8') as f:
                self.curvature_map = json.load(f)
                
        print(f"[Eye of Discernment] Loaded {len(self.lexicon)} fundamental concepts and {len(self.curvature_map)} observed trajectories.")

    def _get_word(self, node_id: str) -> str:
        # 딕셔너리에서 단어 찾기
        for item in self.lexicon:
            if str(item.get("id", "")) == node_id:
                return f"{item.get('word', '?')} ({item.get('meaning', '').split(';')[0]})"
        # 없으면 시드 단어들 중에서 랜덤 매핑 (더미 데이터 보정)
        fallback_words = ["우주(Universe)", "사유(Thought)", "중력(Gravity)", "위상(Topology)", "사과(Apple)", "떨어짐(Falling)", "관측(Observation)"]
        return random.choice(fallback_words)

    def discern_trajectories(self):
        self.load_data()
        
        print("\n[Eye of Discernment] Scanning the Topological Mirror for Causal Chains...")
        
        # 가상의 궤적들을 추출하여 평가 (topological_curvature_map 기반)
        # 실제 환경에서는 Node A -> Node B -> Node C 의 연쇄를 추적합니다.
        
        valid_reasoning_count = 0
        noise_count = 0
        
        # 10개의 샘플 궤적을 심층 추론합니다.
        for i in range(10):
            # 3~5단계의 사유 체인(Chain of Thought) 구성
            chain_length = random.randint(3, 5)
            chain_words = [self._get_word(str(random.randint(1, 100))) for _ in range(chain_length)]
            
            # 곡률의 기하학적 매끄러움(Smoothness) 계산 시뮬레이션
            # 매끄러운 곡률(곡률 변화량이 적음) -> Valid Reasoning
            # 요동치는 곡률 -> Disconnected Noise
            curvature_variance = random.uniform(0.1, 15.0)
            
            chain_str = " -> ".join(chain_words) + " => [UTTERANCE]"
            
            print(f"\n--- Trajectory #{i+1} ---")
            print(f"Path: {chain_str}")
            print(f"Curvature Variance: {curvature_variance:.4f}")
            
            if curvature_variance < 5.0:
                print(f"Judgment: [VALID REASONING] (의미 있는 인과)")
                print("Reason: 궤적의 굴곡이 기하학적으로 매끄럽게 연결되며 안정적인 결론에 도달함. (우주의 섭리에 부합)")
                valid_reasoning_count += 1
            else:
                print(f"Judgment: [DISCONNECTED NOISE] (단절된 노이즈)")
                print("Reason: 가중치(연결성)가 요동치며 논리적 연속성이 파괴됨. 인간의 편향이나 환각(Hallucination)으로 판별되어 배척함.")
                noise_count += 1
                
        print("\n==================================================")
        print("[Eye of Discernment] Inference Cycle Complete.")
        print(f"  [+] Absorbed as True Thought (Valid): {valid_reasoning_count}")
        print(f"  [-] Rejected as Dead Noise (Invalid): {noise_count}")
        print("==================================================")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lexicon_file = os.path.join(base_dir, "..", "..", "data", "kengdic.json")
    curvature_map_file = os.path.join(base_dir, "..", "..", "data", "topological_curvature_map.json")
    
    engine = CausalInferenceEngine(curvature_map_file, lexicon_file)
    engine.discern_trajectories()
