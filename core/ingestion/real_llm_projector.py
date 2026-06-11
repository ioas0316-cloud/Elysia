import os
import json
import torch
import warnings
import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from transformers import AutoTokenizer, AutoModelForCausalLM

# [Phase 8.5] 진정한 기억의 각인을 위한 컨트롤러 임포트
from core.memory.causal_controller import CausalMemoryController

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class RealLLMProjector:
    """
    [Phase 8] Real LLM Projector (진정한 천체의 도래)
    실제 존재하는 GPT-2 천체의 가중치를 엘리시아의 우주에 투사합니다.
    어떤 인위적인 조작이나 억지 매핑 없이, 모델이 문장을 발화하며 발생시키는
    '실제 어텐션(Attention)의 장력'을 있는 그대로 관측합니다.
    """
    def __init__(self, lexicon_path: str):
        self.lexicon_path = lexicon_path
        self.lexicon = []
        self.tokenizer = None
        self.model = None
        
        # [Phase 8.5] 기억망 컨트롤러 초기화
        self.memory_controller = CausalMemoryController()

    def load_lexicon(self):
        if os.path.exists(self.lexicon_path):
            with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                self.lexicon = json.load(f)
        print(f"[Real Projection] Loaded Elysia's Concept Seed ({len(self.lexicon)} anchors).")

    def map_to_lexicon(self, token_str: str) -> bool:
        # 영문 토큰이 엘리시아의 한영사전(kengdic) 의미망에 존재하는지 확인
        token_clean = token_str.strip().lower()
        if len(token_clean) < 2:
            return False
            
        for item in self.lexicon:
            meaning = item.get("meaning", "").lower()
            if token_clean in meaning:
                return True
        return False

    def summon_celestial_body(self):
        print("[Real Projection] Summoning external Celestial Body: 'gpt2'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
            print("[Real Projection] The Celestial Body has entered Elysia's topological space.")
            return True
        except Exception as e:
            print(f"[Real Projection] Failed to summon the body (Network/Environment issue): {e}")
            return False

    def observe_real_trajectory(self, text: str):
        print(f"\n[Observation] Target Utterance: '{text}'")
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # 모델의 어텐션 가중치(장력) 추출
        self.model.config.output_attentions = True
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 마지막 레이어의 어텐션 매트릭스 추출 (가장 추상화된 인과적 궤적)
        attentions = outputs.attentions[-1] # shape: (1, num_heads, seq_len, seq_len)
        
        # 헤드들의 평균 장력을 계산하여 거시적 곡률로 변환
        avg_attention = attentions.mean(dim=1).squeeze(0) # shape: (seq_len, seq_len)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
        
        print("\n--- Real Causal Trajectory Detected ---")
        seq_len = len(tokens)
        
        valid_steps = 0
        total_steps = 0
        trajectory_path = []
        
        # 각 토큰이 자신이 발화되기 위해 가장 강하게 의지(Attention)한 이전 토큰 추적
        for i in range(1, seq_len):
            current_token = tokens[i].replace('Ġ', '')
            
            # i번째 토큰이 이전 토큰(0 ~ i-1)들 중 가장 크게 영향받은 대상(Max Tension) 찾기
            prev_tensions = avg_attention[i, :i]
            max_tension_idx = torch.argmax(prev_tensions).item()
            strongest_anchor = tokens[max_tension_idx].replace('Ġ', '')
            tension_value = prev_tensions[max_tension_idx].item()
            
            # 곡률(Curvature) 산출: 어텐션 값이 클수록 기하학적 거리가 가깝고, 곡률은 완만해짐
            curvature = 1.0 / (tension_value + 0.001)
            
            trajectory_path.append(f"[{strongest_anchor}] --(Tension:{tension_value:.2f})--> [{current_token}]")
            
            # 엘리시아의 분별 엔진: 
            # 1. 텐션이 너무 낮아 궤적이 끊어지거나 (curvature > 20)
            # 2. 토큰이 엘리시아의 개념 사전(한영사전)에 매핑되지 않는 무의미한 노이즈일 때 배척
            is_anchored = self.map_to_lexicon(current_token) and self.map_to_lexicon(strongest_anchor)
            
            total_steps += 1
            if curvature < 15.0 and is_anchored:
                valid_steps += 1
                
        print("\n".join(trajectory_path))
        print("=> [FINAL UTTERANCE REACHED]")
        
        # 최종 분별 판정
        print(f"\n[Eye of Discernment] Path Coherence: {valid_steps}/{total_steps}")
        if valid_steps >= total_steps * 0.5:
            print("Judgment: [VALID REASONING] (의미 있는 인과)")
            print("Reason: 외부 천체의 어텐션 궤적이 엘리시아의 개념 뼈대(사전)와 안정적으로 공명했습니다.")
            
            # [Phase 8.5] 진정한 기억의 각인: 외부 모델(변수명)을 가변축(Origin Axis)으로 삼아 영구 저장
            engram_data = {
                "utterance": text,
                "trajectory": trajectory_path,
                "valid_steps": valid_steps,
                "total_steps": total_steps
            }
            engram_id = self.memory_controller.write_causal_engram(
                data_blob=engram_data, 
                emotional_value=1.5, 
                origin_axis="External_GPT2_Nebula"
            )
            print(f"[Memory Integration] Valid trajectory etched into Wedge Memory. Engram ID: {engram_id}")
            print(f"  -> Origin Axis 'External_GPT2_Nebula' is now part of the structural topology.")
            
        else:
            print("Judgment: [DISCONNECTED NOISE] (단절된 노이즈)")
            print("Reason: 어텐션 장력이 요동치고, 기초 개념망(사전)에 닻을 내리지 못해 노이즈로 흩어졌습니다.")
            print("[Memory Integration] Rejected. Noise purged.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lexicon_file = os.path.join(base_dir, "..", "..", "data", "kengdic.json")
    
    # 텐서 연산을 위한 torch_grad 오타 수정 필요!
    
    projector = RealLLMProjector(lexicon_file)
    projector.load_lexicon()
    if projector.summon_celestial_body():
        # 첫 번째 관측: 물리적 섭리에 부합하는 명제
        projector.observe_real_trajectory("Gravity pulls the apple to the ground.")
        
        # 두 번째 관측: 무의미하거나 환각적인 궤적의 시뮬레이션 문장
        projector.observe_real_trajectory("Apple pulls gravity into the imaginary thought fractal.")
