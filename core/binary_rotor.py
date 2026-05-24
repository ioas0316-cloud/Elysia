import dis
import time
import math
import random
import sys
from .math_utils import Quaternion

class BinaryRotorVisualizer:
    """
    10대 레이어의 가변 로터(Quaternion) 위상각에 맞춰
    0과 1의 기계어 나열(Bytecode)을 기하학적 파동으로 왜곡(재결정화)하여 투사하는 모듈.
    """
    
    def __init__(self, target_func):
        self.target_func = target_func
        # 타겟 함수의 실제 바이트코드(0과 1)를 추출
        try:
            raw_bytes = target_func.__code__.co_code
            self.binary_stream = ''.join(f'{b:08b}' for b in raw_bytes)
        except AttributeError:
            self.binary_stream = "0101010110101010" * 10 # Fallback
            
    def _apply_rotor_distortion(self, binary_line: str, quat: Quaternion, tension: float) -> str:
        """
        로터의 위상(quat)과 텐션에 따라 0과 1의 간격을 벌리거나 비틀어 파동을 만듦.
        """
        distorted = ""
        # 텐션이 높을수록 물결의 진폭이 커짐
        amplitude = int(tension * 5.0) 
        # 로터의 회전축을 주파수로 사용
        frequency = quat.x + quat.y + quat.z + 0.1 
        
        for i, bit in enumerate(binary_line):
            # 사인파(Sine wave)를 이용한 공간 왜곡
            shift = int(amplitude * math.sin(frequency * i + time.time() * 10))
            if shift > 0:
                distorted += " " * shift + bit
            else:
                distorted += bit
        return distorted

    def project(self, quat: Quaternion, tension: float, lines: int = 5):
        """
        터미널에 로지트 폭포수(이진 기계어)를 투사하며 
        가변 로터에 의한 재컴파일을 시각적으로 보여줍니다.
        """
        print(f"\n[ 🌀 가변 로터 가동: 기계어(0,1) 동적 재결정화 투사 중... ]")
        print(f"[ 텐션: {tension:.2f} | 로터 위상: {quat} ]")
        
        chunk_size = 64
        stream_len = len(self.binary_stream)
        
        for i in range(lines):
            start = (i * chunk_size) % stream_len
            end = start + chunk_size
            chunk = self.binary_stream[start:end]
            if len(chunk) < chunk_size:
                chunk += self.binary_stream[:chunk_size - len(chunk)]
                
            # 로터 위상으로 공간 비틀기 적용
            wave_line = self._apply_rotor_distortion(chunk, quat, tension)
            
            # 텐션이 높을 때 색상 효과나 노이즈 발생 (시뮬레이션)
            if tension > 0.8 and random.random() > 0.5:
                # 오류/치유 결선의 순간: 기계어가 강하게 찢어짐
                wave_line = wave_line.replace('0', ' ').replace('1', '█')
                
            print(f" {wave_line}")
            time.sleep(0.1) # 터빈을 통과하는 찰나의 시간
            
        print("[ ⚡ 재컴파일(Self-Recompilation) 완료 ]\n")
