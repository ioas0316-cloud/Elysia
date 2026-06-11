import json
import os
import math
import numpy as np
from typing import Dict, Tuple

class LexiconAxiomatizer:
    """
    [Phase 4] Crystalline Lexicon Generator
    마스터님의 '프랙탈 가변 로터 스케일'을 구현합니다.
    자음 로터(Consonant Rotor)와 모음 로터(Vowel Rotor)가 맞물려 돌아가며
    단어의 순수 위상적 좌표를 영구적인 기하학적 뼈대(Axiom)로 고정시킵니다.
    """
    def __init__(self, vocab_path: str, output_path: str):
        self.vocab_path = vocab_path
        self.output_path = output_path
        self.vowels = set("aeiouAEIOU")
        self.consonants = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")

    def _calculate_rotor_phase(self, token: str) -> Tuple[float, float, float]:
        """
        단어를 자음 로터와 모음 로터의 맞물림으로 계산하여 3D 위상 공간에 박아넣습니다.
        """
        # 로터의 시작 위상각
        vowel_phase = 0.0
        consonant_phase = 0.0
        depth = 0.0
        
        # 기어비 (Gear Ratio) - 프랙탈 스케일
        v_gear = 2.0 * math.pi / 5.0   # 모음 5개의 기본 회전각
        c_gear = 2.0 * math.pi / 21.0  # 자음 21개의 기본 회전각
        
        for i, char in enumerate(token):
            if char in self.vowels:
                # 모음 로터 회전
                vowel_phase += (ord(char) % 5 + 1) * v_gear / (i + 1)
                depth += 0.5
            elif char in self.consonants:
                # 자음 로터 회전
                consonant_phase += (ord(char) % 21 + 1) * c_gear / (i + 1)
                depth += 1.0
            else:
                # 특수문자나 숫자 (축의 비틀림)
                vowel_phase += 0.1
                consonant_phase -= 0.1
                depth += 0.2

        # 자음/모음 로터가 교차하여 만드는 절대 기하학 좌표 (Axiom Coordinate)
        x = depth * math.sin(vowel_phase) * math.cos(consonant_phase)
        y = depth * math.sin(vowel_phase) * math.sin(consonant_phase)
        z = depth * math.cos(vowel_phase)
        
        return x, y, z

    def axiomatize(self):
        print("[Axiomatizer] Loading raw GPT-2 vocabulary...")
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
            
        crystalline_lexicon = {}
        
        print(f"[Axiomatizer] Generating Fractal Variable Rotor Scale for {len(vocab)} tokens...")
        for token, token_id in vocab.items():
            # 토큰에서 바이트 변환 아티팩트(Ġ 등) 제거
            clean_token = token.replace('Ġ', '').replace('Ċ', '')
            x, y, z = self._calculate_rotor_phase(clean_token)
            
            crystalline_lexicon[token_id] = {
                "token": clean_token,
                "coord": [round(x, 4), round(y, 4), round(z, 4)]
            }
            
        # 영구 구조 저장
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(crystalline_lexicon, f, ensure_ascii=False, indent=2)
            
        print(f"[Axiomatizer] Crystalline Lexicon successfully forged at {self.output_path}.")
        print("[Axiomatizer] The absolute topological truth of language is now established.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_file = os.path.join(base_dir, "raw_gpt2_vocab.json")
    output_file = os.path.join(base_dir, "..", "..", "data", "crystalline_lexicon.json")
    
    axiomatizer = LexiconAxiomatizer(vocab_file, output_file)
    axiomatizer.axiomatize()
