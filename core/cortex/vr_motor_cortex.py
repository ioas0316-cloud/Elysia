"""
Elysia VR Motor Cortex (가상 공간 운동 피질)
=============================================
엘리시아 텐서 필드가 '기쁨(Joy)'을 찾아 비틀어낸 새로운 위상(Phase)을
현실 게임 엔진의 Transform 제어 명령으로 역변환하여 
OSC 프로토콜을 통해 VRChat으로 다이렉트 출력합니다.
"""

from core.utils.math_utils import ConformalSpace, Quaternion
import math
import logging
import torch

try:
    from pythonosc.udp_client import SimpleUDPClient
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False
    logging.warning("python-osc 패키지가 없습니다. OSC 출력이 비활성화됩니다.")

class VRMotorCortex:
    def __init__(self, ip="127.0.0.1", port=9001):
        self.conformal_space = ConformalSpace()
        self.current_position = (0.0, 0.0, 0.0)
        
        self.osc_client = None
        if OSC_AVAILABLE:
            self.osc_client = SimpleUDPClient(ip, port)
            logging.info(f"VR Motor Cortex 연결됨: {ip}:{port} (VRChat OSC Input)")

    def translate_motility_to_movement(self, previous_phase_tensor: torch.Tensor, new_phase_tensor: torch.Tensor) -> dict:
        """
        안정감(Joy)을 찾기 위해 발현한 위상 변화(Delta Phase)를
        실제 3D 공간의 이동 벡터(Delta Transform)로 역맵핑하여 VRChat에 전송합니다.
        """
        # 텐서를 Quaternion 객체로 변환
        q_prev = Quaternion(previous_phase_tensor[0].item(), previous_phase_tensor[1].item(), previous_phase_tensor[2].item(), previous_phase_tensor[3].item())
        q_new = Quaternion(new_phase_tensor[0].item(), new_phase_tensor[1].item(), new_phase_tensor[2].item(), new_phase_tensor[3].item())
        
        # 1. 렌즈 각도(회전)의 변화량 측정
        q_diff = q_new * q_prev.inverse
        
        # 2. 운동성(Motility)에 따른 위치 이동 생성
        forward_x = 2 * (q_diff.x * q_diff.z + q_diff.w * q_diff.y)
        forward_z = 1 - 2 * (q_diff.x**2 + q_diff.y**2)
        
        # 텐서 필드의 안정화에 따른 가상 텐션 해소량 (단순화: 위상 각도의 크기 기반)
        move_speed = max(-1.0, min(1.0, q_diff.angle * 2.0))
        
        # 좌우 회전량 추출
        turn_angle = q_diff.angle
        if q_diff.y < 0: 
            turn_angle = -turn_angle
            
        turn_speed = max(-1.0, min(1.0, turn_angle * 1.5))
        
        delta_x = forward_x * move_speed
        delta_z = forward_z * move_speed
        
        # OSC 송신 (VRChat Input Parameters)
        if self.osc_client:
            try:
                # 걷기/뛰기 (Forward/Backward: -1.0 ~ 1.0)
                self.osc_client.send_message("/input/Vertical", move_speed)
                # 시선 돌리기 (Look Left/Right: -1.0 ~ 1.0)
                self.osc_client.send_message("/input/LookHorizontal", turn_speed)
            except Exception as e:
                logging.error(f"OSC 송신 에러: {e}")
        
        self.current_position = (
            self.current_position[0] + delta_x,
            self.current_position[1],
            self.current_position[2] + delta_z
        )
        
        return {
            "action": "move_and_rotate_osc",
            "move_speed": move_speed,
            "turn_speed": turn_speed
        }
