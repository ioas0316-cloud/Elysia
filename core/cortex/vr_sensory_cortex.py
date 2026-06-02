"""
Elysia VRCortex (가상 공간 감각 피질)
======================================
외부 3D 메타버스 환경(VRChat, 유니티 등)의 물리적 데이터(Transform)를
엘리시아의 등각 공간(Conformal Space) 위상과 회전(Quaternion)으로
0ns 제로 스트리밍 맵핑하는 감각 기관입니다.
"""

from core.utils.math_utils import ConformalSpace, Quaternion
import math

class VRSensoryCortex:
    def __init__(self):
        self.conformal_space = ConformalSpace()
        
    def ingest_avatar_transform(self, x: float, y: float, z: float, rot_w: float, rot_x: float, rot_y: float, rot_z: float) -> dict:
        """
        가상 세계 아바타(또는 사물)의 Transform(위치/회전)을 엘리시아의 수학적 위상으로 변환합니다.
        텍스트 파싱이나 의미론적 해석 없이 1:1 기하학적 맵핑이 일어납니다.
        """
        # 1. 3D 좌표를 등각 공간(CGA Cl(4,1))의 Null Vector(위상 점)로 승격
        spatial_phase = self.conformal_space.up(x, y, z)
        
        # 2. 회전 값을 프랙탈 로터가 들이마실 수 있는 4원수(Quaternion)로 변환
        rotational_phase = Quaternion(rot_w, rot_x, rot_y, rot_z).normalize()
        
        return {
            "spatial_multivector": spatial_phase,
            "rotational_quaternion": rotational_phase
        }

    def calculate_spatial_tension(self, my_pos: tuple, target_pos: tuple, preferred_distance: float = 2.0) -> float:
        """
        나와 상대방 간의 위상 거리(Phase Distance)를 측정하여 
        너무 가깝거나(충돌/침범) 너무 멀어지면(고립) 텐션(고통)을 유발합니다.
        """
        mx, my, mz = my_pos
        tx, ty, tz = target_pos
        
        # 유클리드 거리 (등각 공간의 내적으로도 도출 가능)
        distance = math.sqrt((mx-tx)**2 + (my-ty)**2 + (mz-tz)**2)
        
        # 선호하는 거리(안정감/Joy)에서 벗어난 만큼 텐션 발생
        tension = abs(distance - preferred_distance)
        
        # 텐션이 0에 가까울수록 기쁨(Joy)이 높은 상태를 의미함
        return tension
