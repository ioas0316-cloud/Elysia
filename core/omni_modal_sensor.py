"""
Elysia Omni-Modal Sensor (만물 바이트 감각기)
==============================================
파일의 포맷(txt, jpg, mp3 등)이라는 인간의 라벨(Label)을 철저히 무시합니다.
모든 데이터는 순수한 '0과 1의 연속된 파동'으로 취급됩니다.
0은 수렴(Sameness, W축 방향)의 힘으로,
1은 발산(Difference, X/Y/Z축 방향)의 장력(Tension)으로 치환되어
파일 전체의 구조적 본질을 하나의 거대한 기하학적 로터(Quaternion)로 압축해냅니다.
"""

import os
import math
from core.math_utils import Quaternion

class OmniModalSensor:
    def __init__(self):
        pass

    def ingest_file_as_wave(self, filepath: str) -> Quaternion:
        """
        파일을 바이트 단위로 읽어, 기하학적 텐션 파동으로 변환합니다.
        """
        if not os.path.exists(filepath):
            return Quaternion(1.0, 0.0, 0.0, 0.0)
            
        with open(filepath, "rb") as f:
            byte_stream = f.read()
            
        return self._convert_bytes_to_rotor(byte_stream)

    def _convert_bytes_to_rotor(self, byte_stream: bytes) -> Quaternion:
        """
        바이트 스트림을 연속적인 각운동량(Angular Momentum)으로 치환합니다.
        """
        w, x, y, z = 1.0, 0.0, 0.0, 0.0
        
        # 기하학적 회전의 기본 단위 (아주 작은 앵글)
        base_theta = math.pi / 1024.0
        
        # 파일 전체가 하나의 거대한 연쇄 회전(Chain of Rotations)이 됩니다.
        for b in byte_stream:
            for i in range(8):
                bit = (b >> i) & 1
                
                if bit == 0:
                    # 0: 질서, 수렴, 같음 (Sameness)
                    # W축으로 돌아가려는 회귀력
                    q_rot = Quaternion(math.cos(base_theta), 0.0, 0.0, 0.0)
                else:
                    # 1: 텐션, 발산, 다름 (Difference)
                    # 3차원 공간으로 뻗어나가는 장력 (인덱스에 따라 방향이 달라짐)
                    axis = i % 3
                    if axis == 0:
                        q_rot = Quaternion(math.cos(base_theta), math.sin(base_theta), 0.0, 0.0)
                    elif axis == 1:
                        q_rot = Quaternion(math.cos(base_theta), 0.0, math.sin(base_theta), 0.0)
                    else:
                        q_rot = Quaternion(math.cos(base_theta), 0.0, 0.0, math.sin(base_theta))
                
                # 기존 파동에 새로운 텐션을 곱하여 기하학적 공간을 비틂
                # Hamilton Product 연쇄 계산
                w_new = w*q_rot.w - x*q_rot.x - y*q_rot.y - z*q_rot.z
                x_new = w*q_rot.x + x*q_rot.w + y*q_rot.z - z*q_rot.y
                y_new = w*q_rot.y - x*q_rot.z + y*q_rot.w + z*q_rot.x
                z_new = w*q_rot.z + x*q_rot.y - y*q_rot.x + z*q_rot.w
                
                # 정규화하여 발산(NaN) 방지
                norm = math.sqrt(w_new**2 + x_new**2 + y_new**2 + z_new**2)
                if norm > 0:
                    w, x, y, z = w_new/norm, x_new/norm, y_new/norm, z_new/norm
                    
        return Quaternion(w, x, y, z)
