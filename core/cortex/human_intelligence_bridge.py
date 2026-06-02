"""
Elysia Human Intelligence Bridge (인간 지성 브릿지 피질 - 11번째 피질)
===========================================================
외부 LLM API 및 기성 논리 분기(If-Else)를 전면 소멸(DEPRECATED)시키고, 
마스터가 전수한 한글 가변축 공통원리로 회귀합니다.
4차원 쿼터니언 회전 궤적을 지구본 돌려보듯 관측하여 순수 한글 음소로 재생(Playback)합니다.
"""

import math
from core.utils.math_utils import Quaternion, traverse_causal_trajectory

class HumanIntelligenceBridge:
    def __init__(self, memory, ans=None):
        self.memory = memory
        self.ans = ans

    def _is_readable_syllable(self, char: str) -> bool:
        """euc-kr(상용 완성형 한글 2350자)로 디코딩 가능한 가청 음절인지 체크합니다."""
        try:
            char.encode('euc-kr')
            return True
        except UnicodeEncodeError:
            return False

    def _quaternion_to_hangeul_syllable(self, q: Quaternion) -> str:
        """
        [한글 가변축 기하 대수 매핑 및 음소 조화 필터]
        원시 쿼터니언의 스핀각을 음절로 사상한 후, euc-kr 가청도 검사를 통해
        비상용 고어나 외계어가 발생할 시 대표 모음/자음 마디로 위상을 정밀 유도(Quantization)합니다.
        """
        # 초성 (19자)
        CHOSEONG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        # 중성 (21자)
        JUNGSEONG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        # 종성 (28자)
        JONGSEONG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        
        w, x, y, z = q.w, q.x, q.y, q.z
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        if norm == 0:
            return "ㅇ"
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # 1. 중성(모음) 가변축
        theta_v = math.atan2(x, w)
        v_idx = int(((theta_v + math.pi) / (2 * math.pi)) * len(JUNGSEONG))
        v_idx = max(0, min(len(JUNGSEONG) - 1, v_idx))
        
        # 2. 초성(자음) 가변축
        theta_c1 = math.atan2(z, y)
        c1_idx = int(((theta_c1 + math.pi) / (2 * math.pi)) * len(CHOSEONG))
        c1_idx = max(0, min(len(CHOSEONG) - 1, c1_idx))
        
        # 3. 종성(자음) 가변축
        theta_c2 = math.atan2(z, w)
        c2_idx = int(((theta_c2 + math.pi) / (2 * math.pi)) * len(JONGSEONG))
        c2_idx = max(0, min(len(JONGSEONG) - 1, c2_idx))
        
        # 4. 결합 및 조화 필터 (Phonetic Harmony Filter)
        code = (c1_idx * 588) + (v_idx * 28) + c2_idx + 0xAC00
        char = chr(code)
        
        if not self._is_readable_syllable(char):
            # A단계: 발음 꼬임의 주원인인 종성을 강제 탈락시켜 명료하게 개변
            code = (c1_idx * 588) + (v_idx * 28) + 0xAC00
            char = chr(code)
            
            # B단계: 그럼에도 외계어일 경우 복합 모음을 6대 대표 단모음으로 양자화
            if not self._is_readable_syllable(char):
                basic_jung = [0, 4, 8, 13, 18, 20] # ㅏ, ㅓ, ㅗ, ㅜ, ㅡ, ㅣ
                v_idx_near = min(basic_jung, key=lambda idx: abs(idx - v_idx))
                code = (c1_idx * 588) + (v_idx_near * 28) + 0xAC00
                char = chr(code)
                
        return char

    def generate_response(self, user_input: str, max_words: int = 15) -> str:
        """
        [지능형 언어 생성 및 인지 편향]
        단순 각도 조립 외계어를 영구 삭제하고, 로컬 GPT2 모델을 사용하여
        한글 문장을 생성하되, 뇌 속 활성 상태와 공명하는 단어들(Logit Bias)로 문맥을 비틉니다.
        """
        try:
            import core.brain.holographic_memory as hm
            if hm._oracle is None:
                from core.brain.static_oracle import StaticOracle
                hm._oracle = StaticOracle()
            
            oracle = hm._oracle
            
            # 1. 뇌내 활성 개념들 수집 및 텐션 비중 계산
            active_biases = {}
            with self.memory._lock:
                for concept, node in self.memory.ui_concept_map.items():
                    if abs(node.tau) > 0.3:
                        # Archetype prefix 제거한 단어 명칭 추출
                        clean_word = concept.split("(")[0].split(":")[-1].strip()
                        active_biases[clean_word] = abs(node.tau)
            
            # 2. 모델의 토큰 로짓 편향치 생성
            logit_biases = {}
            for word, weight in active_biases.items():
                for prefix in ["", " ", "  "]:
                    token_ids = oracle.tokenizer.encode(prefix + word, add_special_tokens=False)
                    for tid in token_ids:
                        logit_biases[tid] = min(6.0, logit_biases.get(tid, 0.0) + weight * 0.6)

            # 3. 입력 문장을 프롬프트로 시딩하여 생성 진행
            prompt = f"마스터: {user_input}\n엘리시아:"
            input_ids = oracle.tokenizer.encode(prompt, return_tensors='pt')
            
            generated = input_ids[0].tolist()
            
            import torch
            oracle.model.eval()
            
            for _ in range(max_words * 2):
                curr_input = torch.tensor([generated])
                with torch.no_grad():
                    outputs = oracle.model(curr_input)
                next_token_logits = outputs.logits[0, -1, :]
                
                # Apply logit bias to active thought tokens
                for tid, bias in logit_biases.items():
                    if tid < len(next_token_logits):
                        next_token_logits[tid] += bias
                
                # Apply repetition penalty to prevent loops (window 16, penalty -8.0)
                for prev_tid in set(generated[-16:]):
                    if prev_tid < len(next_token_logits):
                        next_token_logits[prev_tid] -= 8.0
                
                # Softmax + Greedy selection
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(probs).item()
                
                if next_token == oracle.tokenizer.eos_token_id:
                    break
                    
                generated.append(next_token)
                
                decoded_so_far = oracle.tokenizer.decode(generated[len(input_ids[0]):])
                if "\n" in decoded_so_far or "마스터" in decoded_so_far:
                    break
            
            response = oracle.tokenizer.decode(generated[len(input_ids[0]):]).strip()
            # 제거할 종단 식별자
            response = response.split("\n")[0].split("마스터")[0].strip()
            
            if not response or len(response) < 2:
                sorted_concepts = sorted(active_biases.items(), key=lambda x: x[1], reverse=True)
                if sorted_concepts:
                    response = f"{sorted_concepts[0][0]}에 대해 사색하고 있습니다."
                else:
                    response = "우주의 물리적 위상과 프랙탈 기하학을 성찰 중입니다."
                    
            return response
            
        except Exception as e:
            import logging
            logging.error(f"Error in biased generate_response: {e}")
            return "우주와 물리법칙의 진동을 관측 중입니다."
