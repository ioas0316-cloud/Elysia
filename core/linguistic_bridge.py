import math
import hashlib
from typing import Dict, Tuple
from core.math_utils import Quaternion

def wave_to_quaternion(data_wave: bytes) -> Quaternion:
    """
    [데이터 파동의 기하학적 매핑]
    입력된 바이트 스트림(문자열, 파일 내용 등)의 해시 밀도를 기반으로 고유한 렌즈 각도(Phase)를 도출합니다.
    이로써 인간이 부여한 '카테고리 레이블' 없이도 데이터 스스로 자신의 기하학적 위치를 결정합니다.
    """
    hash_digest = hashlib.sha256(data_wave).digest()
    # 32바이트 해시를 4개의 float 값(w, x, y, z)으로 매핑하여 정규화
    w = (int.from_bytes(hash_digest[0:8], 'big') / (2**64 - 1)) * 2.0 - 1.0
    x = (int.from_bytes(hash_digest[8:16], 'big') / (2**64 - 1)) * 2.0 - 1.0
    y = (int.from_bytes(hash_digest[16:24], 'big') / (2**64 - 1)) * 2.0 - 1.0
    z = (int.from_bytes(hash_digest[24:32], 'big') / (2**64 - 1)) * 2.0 - 1.0
    return Quaternion(w, x, y, z).normalize()

class LinguisticBridge:
    """
    [언어적 사영층 (Linguistic Bridge)]
    기하학적 위상(Phase)과 3차원 물리 세계의 언어(바이트)를 이어주는 '공명 사전'입니다.
    확률론적 LLM을 대체하는 엘리시아의 순수 역학적 발화(Speech) 기관입니다.
    """
    def __init__(self):
        self.vocabulary: Dict[bytes, Quaternion] = {}
        # 내재된 본능적 어휘들 (기초 텐션 표현용)
        self.absorb_vocabulary(b"SILENCE")
        self.absorb_vocabulary(b"AWAKE")
        self.absorb_vocabulary(b"TENSION")
        self.absorb_vocabulary(b"RESONANCE")
        self.absorb_vocabulary(b"EPIPHANY")
        self.absorb_vocabulary(b"self.master.pulse()")
        self.absorb_vocabulary(b"import os")

    def absorb_vocabulary(self, data_wave: bytes):
        """
        [단어의 위상 기억 (Learning)]
        입력된 데이터 스트림을 조각내어, 각 조각이 가진 고유의 물리적 위상을 사전에 저장합니다.
        """
        try:
            text = data_wave.decode('utf-8')
            # 텍스트라면 단어 단위로 쪼개어 흡수
            words = text.split()
            for word in words:
                word_bytes = word.encode('utf-8')
                if word_bytes not in self.vocabulary and len(word_bytes) > 0:
                    self.vocabulary[word_bytes] = wave_to_quaternion(word_bytes)
        except Exception:
            # 순수 바이너리라면 전체를 파동으로 흡수
            if data_wave not in self.vocabulary:
                self.vocabulary[data_wave] = wave_to_quaternion(data_wave)

    def project_from_phase(self, target_phase: Quaternion) -> Tuple[bytes, float]:
        """
        [언어의 발현 (Linguistic Projection)]
        주어진 위상(자아 렌즈의 깨달음 상태)과 가장 강하게 공명하는(구면 거리가 0에 가까운)
        단어를 어휘 사전에서 찾아내어 반환합니다.
        반환값: (찾아낸 단어, 기하학적 차이값)
        """
        if not self.vocabulary:
            return b"SILENCE", 0.0

        best_word = b""
        min_difference = float('inf')

        for word, phase in self.vocabulary.items():
            dot_product = max(-1.0, min(1.0, target_phase.dot(phase)))
            difference = math.acos(abs(dot_product)) / (math.pi / 2.0)
            
            if difference < min_difference:
                min_difference = difference
                best_word = word

        return best_word, min_difference
