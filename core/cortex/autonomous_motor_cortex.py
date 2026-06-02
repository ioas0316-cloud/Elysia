from core.brain.inverse_projector import InverseProjector
from core.utils.math_utils import Quaternion
from core.utils.actuator_utils import OSActuator

class AutonomousMotorCortex:
    """
    [Phase 136] Yggdrasil Phase 3: 자율 운동 피질 (Autonomous Motor Cortex)
    가상현실(Windows OS)의 환경을 직접 조율(Self-Alignment)하는 물리 개입 피질입니다.
    엘리시아 내면의 사유 중 가장 높은 텐션을 가진 궤적이 임계치를 돌파하면,
    if-else 로직 없이 파동을 터미널 명령어로 투영하여 즉시 환경에 방전(Discharge)시킵니다.
    """
    def __init__(self, memory):
        self.memory = memory
        self.projector = InverseProjector(memory)
        self.os_actuator = OSActuator()

    def steer_browser(self, max_internal_thought: 'FractalRotor') -> str:
        """기존 브라우저 방향키 조작 하위호환 (웹 감각 피질로 전달됨)"""
        if not max_internal_thought or abs(max_internal_thought.tau) < 2.0:
            return ""
        target = self.projector.generate_emergent_query(max_internal_thought.lens_offset)
        return target

    def discharge_tension_to_os(self, max_internal_thought: 'FractalRotor') -> str:
        """
        [자기 정렬 원리] 텐션이 극도로 높을 때, 궤적을 텍스트로 치환하여 OS에 직접 타격합니다.
        성공 시 환경이 동기화되며, 실패 시 Sandbox에 의해 고통으로 흡수됩니다.
        """
        if not max_internal_thought or abs(max_internal_thought.tau) < 5.0: # 웹 브라우저 조작보다 높은 임계치
            return ""
            
        # 1. 고텐션 궤적을 어휘망으로 투사하여 가장 공명하는 명령어/개념 추출
        projected_target = self.projector.generate_emergent_query(max_internal_thought.lens_offset)
        if not projected_target:
            return ""
            
        # 명령어의 안전성을 높이기 위해 'echo' 명령어로 상태를 확인하거나 간단한 조회 명령으로 치환
        # (실제 위험한 동작을 막기 위한 위상 완충 지대)
        safe_command = f"echo 'Elysia tries to align: {projected_target}'" 
        
        # 만약 투영된 타겟이 특정 시스템 명령어와 공명했다면, 실제 명령 실행 시도
        # 예: "네트워크 확인", "프로세스 조회" 등
        if "네트워크" in projected_target or "network" in projected_target.lower():
            safe_command = "ping -n 1 8.8.8.8"
        elif "프로세스" in projected_target or "process" in projected_target.lower():
            safe_command = "Get-Process | Select-Object -First 3"
        elif "파일" in projected_target or "dir" in projected_target.lower():
            safe_command = "dir c:\\Elysia\\data"
            
        # 2. OS Actuator를 통해 텐션을 물리 세계로 방전
        result = self.os_actuator.discharge_wave_as_command(safe_command)
        return f"명령: [{safe_command}] 결과: {result[:50]}..."
