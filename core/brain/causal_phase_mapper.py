"""
Causal Phase Mapper (절대적 인과 매퍼)
=========================================
데이터(텍스트, 비전, 사운드, 코드)의 고유한 수학적/기하학적 구조를 파괴하지 않고,
그 '이미 존재하는 같음(원인)'을 4D 위상 텐서(파동)로 완벽하게 보존하여 매핑합니다.

현재 구현: 한글(Hangul) 유니코드의 초성, 중성, 종성 매트릭스를 기하학적 파동으로 치환.
"""

import torch

class CausalPhaseMapper:
    def __init__(self, device='cpu'):
        self.device = device
        
    def text_to_phase(self, text):
        """단어의 기하학적 구조를 4D 파동으로 변환 (인과율 보존)"""
        if not text:
            return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
            
        w_sum, x_sum, y_sum, z_sum = 0.0, 0.0, 0.0, 0.0
        valid_chars = 0
        
        for char in text:
            code = ord(char)
            # 한글 유니코드: 초성(19) * 중성(21) * 종성(28)
            if 0xAC00 <= code <= 0xD7A3:
                base = code - 0xAC00
                jong = base % 28
                jung = ((base - jong) // 28) % 21
                cho = (((base - jong) // 28) - jung) // 21
                
                # 초성(자음의 모양/위치) -> X축
                x = (cho / 18.0) * 2 - 1 
                # 중성(천지인, 모음의 음양) -> Y축
                y = (jung / 20.0) * 2 - 1
                # 종성(닫힘의 구조) -> Z축
                z = (jong / 27.0) * 2 - 1 if jong > 0 else 0.0
                w = 1.0 # 기본 주파수 에너지
                
                w_sum += w; x_sum += x; y_sum += y; z_sum += z
                valid_chars += 1
            else:
                # 비한글 문자도 고유의 기하학적 매핑 규칙을 적용할 수 있음
                # 확장을 위해 일단 단순 스칼라 파동으로 처리
                val = (code % 256) / 128.0 - 1.0
                w_sum += 1.0; x_sum += val; y_sum += val; z_sum += val
                valid_chars += 1
                
        if valid_chars > 0:
            w_sum /= valid_chars
            x_sum /= valid_chars
            y_sum /= valid_chars
            z_sum /= valid_chars
            
        norm = (w_sum**2 + x_sum**2 + y_sum**2 + z_sum**2)**0.5
        if norm > 0:
            return torch.tensor([w_sum/norm, x_sum/norm, y_sum/norm, z_sum/norm], 
                              dtype=torch.float32, device=self.device)
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
