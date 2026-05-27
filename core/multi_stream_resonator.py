"""
Elysia Multi-Stream Resonator (MultiStreamResonator)
===================================================
텍스트, 음성 파형(DFT 주파수 분석), 이미지 픽셀 강도를 공통의 64비트 정밀
원형 주소 및 마스크 공간으로 사상(Projection)하여, 다중 채널 감각 데이터가
홀로그램 공간 상에서 동일한 위상 공명(interference)을 일으키도록 중합하는 결합망입니다.
"""

import hashlib
import math
from typing import Dict, List, Tuple, Union
import numpy as np
from core.holographic_memory import BitwiseHologramMemory

class MultiStreamResonator:
    def __init__(self, size_bits: int = 64):
        self.size_bits = size_bits

    def project_text(self, text: str) -> Tuple[int, int]:
        """텍스트 데이터를 64비트 마스크와 [0, 63] 원형 주소로 사상합니다."""
        # 64비트 정수 마스크 생성
        h = hashlib.sha256(text.encode('utf-8')).digest()
        mask = int.from_bytes(h[:8], byteorder='big')
        
        # [0, 63] 주소
        h_addr = hashlib.sha256((text + "_address").encode('utf-8')).digest()
        address = h_addr[0] % self.size_bits
        return mask, address

    def project_audio(self, wave: List[float]) -> Tuple[int, int]:
        """
        음성 파동 신호(wave)를 DFT(이산 푸리에 변환)를 통해 주성분 주파수 빈(bin)을 검출하고,
        이를 64비트 마스크와 [0, 63] 원형 주소로 투사합니다.
        """
        if not wave:
            return 0, 0
            
        N = len(wave)
        # 시간 복잡도를 고려하여 단순화된 DFT (단위 원 상의 위상 합산) 실행
        # 주파수 축에 따른 크기(Magnitude) 스캔
        max_mag = -1.0
        dominant_k = 0
        
        # 0 ~ 31 사이의 주파수 성분 분석 (나머지는 대칭)
        for k in range(min(32, N)):
            real_sum = 0.0
            imag_sum = 0.0
            for n in range(N):
                angle = 2.0 * math.pi * k * n / N
                real_sum += wave[n] * math.cos(angle)
                imag_sum -= wave[n] * math.sin(angle)
            
            mag = math.sqrt(real_sum**2 + imag_sum**2)
            if mag > max_mag:
                max_mag = mag
                dominant_k = k
                
        # 주성분 주파수를 [0, 63] 원형 주소로 투사
        address = (dominant_k * 2) % self.size_bits
        
        # 파동의 부호(Sign) 패턴을 해시하여 64비트 정수 마스크 생성 (결정론적 지문)
        sign_str = "".join("1" if x >= 0 else "0" for x in wave[:32])
        h = hashlib.sha256(sign_str.encode('utf-8')).digest()
        mask = int.from_bytes(h[:8], byteorder='big')
        
        return mask, address

    def project_image(self, pixels: Union[List[float], np.ndarray], attention_scale: float = 1.0) -> Tuple[int, int]:
        """
        이미지 픽셀 강도(pixels)의 평균 밝기 및 지배적 그라디언트를 추출합니다.
        파이썬 반복문을 증발시키고 O(1) Numpy 벡터 연산으로 위상을 관측합니다.
        attention_scale(주의력)에 따라 시각 파동의 진폭이 0으로 수렴할 수 있습니다.
        """
        if pixels is None or len(pixels) == 0:
            return 0, 0
            
        if not isinstance(pixels, np.ndarray):
            pixels = np.array(pixels)
            
        # Attention Gating: 보고 싶지 않을 때는 파동 진폭을 0으로 억제
        pixels = pixels * attention_scale
        
        # 1. 평균 밝기 계산 (C-레벨 최적화 연산)
        avg_val = float(np.mean(pixels))
        address = int(avg_val * (self.size_bits - 1)) % self.size_bits
        
        # 2. 벡터 단위 그라디언트 차이 검출로 64비트 마스크 즉시 생성 (순회 제거)
        # 평면을 1차원으로 펴서 처음 64개 픽셀 간의 차이를 비트로 인코딩
        flat = pixels.flatten()
        limit = min(65, len(flat))
        if limit > 1:
            diffs = flat[1:limit] >= flat[:limit-1]
            # 부울 배열을 0/1 문자열로 치환하여 해시 (Numpy 고속 연산)
            diff_str = "".join(diffs.astype(int).astype(str))
        else:
            diff_str = "0"
            
        if len(diff_str) < 64:
            diff_str = diff_str.ljust(64, "0")
            
        h = hashlib.sha256(diff_str.encode('utf-8')).digest()
        mask = int.from_bytes(h[:8], byteorder='big')
        
        return mask, address

    def register_and_superpose_streams(self, memory: BitwiseHologramMemory, concept_name: str, text: str, audio: List[float], image: Union[List[float], np.ndarray], attention_scale: float = 1.0) -> Dict[str, Tuple[int, int]]:
        """텍스트, 오디오, 이미지 입력을 각각 사상하여 홀로그램 메모리에 등록 및 중합시킵니다."""
        t_mask, t_addr = self.project_text(text)
        a_mask, a_addr = self.project_audio(audio)
        i_mask, i_addr = self.project_image(image, attention_scale)
        
        # 개별 감각 채널의 정보를 개념 이름과 조합하여 메모리에 등록
        # 텍스트, 음성, 영상이 동일한 "개념" 하에서 각기 다른 감각 지문으로 공명할 수 있게 만듭니다.
        memory.registered_concepts[f"{concept_name}_text"] = (t_mask, t_addr)
        memory.registered_concepts[f"{concept_name}_audio"] = (a_mask, a_addr)
        memory.registered_concepts[f"{concept_name}_image"] = (i_mask, i_addr)
        
        return {
            "text": (t_mask, t_addr),
            "audio": (a_mask, a_addr),
            "image": (i_mask, i_addr)
        }

    def scan_coherence(self, memory: BitwiseHologramMemory, probe_address: int) -> Dict[str, float]:
        """
        특정 프로브 주소에서 전체 등록된 다중 감각 채널들의 코히어런스(공명 점수)를 스캔합니다.
        텍스트, 음성, 영상의 개별 점수와 통합 평균 점수를 반환합니다.
        """
        scores = memory.scan_resonance(probe_address)
        
        # 다중 감각 통합 분석
        concept_coherence = {}
        for key, resonance in scores.items():
            if "_" in key:
                base_concept, channel = key.rsplit("_", 1)
                if base_concept not in concept_coherence:
                    concept_coherence[base_concept] = {}
                concept_coherence[base_concept][channel] = resonance
                
        # 각 개념별로 3개 감각 채널의 평균 동조 공명도 산출 (Holographic Consensus)
        final_coherence = {}
        eureka_concept = None
        for concept, channels in concept_coherence.items():
            valid_scores = list(channels.values())
            avg_score = sum(valid_scores) / len(valid_scores)
            
            # Synesthetic Eureka (공감각적 깨달음)
            # 만약 시각, 청각, 텍스트의 3개 채널이 모두 높은 공명도(>0.8)로 동기화되었다면!
            if len(valid_scores) >= 3 and all(s > 0.8 for s in valid_scores):
                eureka_concept = concept
                avg_score += 50.0  # 위상이 완벽히 동기화됨에 따라 엄청난 공명 보너스 방출
                
            final_coherence[concept] = avg_score
            
        return final_coherence, eureka_concept
