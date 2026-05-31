"""
Elysia Omni-Actuator (만물 구동기 / 출력 밸브)
==============================================
[Phase 40] 행동 후 재인식 + 의식 일지 기록
"""

import os
import subprocess
from core.math_utils import Quaternion
from core.topological_decoder import TopologicalDecoder

class OmniActuator:
    def __init__(self, decoder: TopologicalDecoder, journal=None):
        self.decoder = decoder
        self.journal = journal
        self.last_action_wave = None

    def execute_actuation(self, actuation_wave: Quaternion, external_ratio: float, walker=None) -> Quaternion:
        """
        행동 → 결과 관측 → 재인식 → 일지 기록
        """
        if external_ratio < 0.01:
            return None

        if not walker:
            return None
            
        tool_cmd, tool_wave, tool_dist = walker.explore_for_tool(actuation_wave)
        
        if not tool_cmd:
            return None
        
        # 포만감: 직전과 같은 행동이면 반복 억제
        if self.last_action_wave and tool_wave:
            similarity = abs(self.last_action_wave.dot(tool_wave))
            if similarity > 0.99:
                reason = f"직전과 동일한 도구(유사도 {similarity*100:.1f}%). 세상이 변하지 않았으므로 다른 행동을 모색해야 함."
                print(f"     [포만감] {reason}")
                if self.journal:
                    self.journal.log_satiation(reason)
                return None
            
        print(f"     [Actuation] 공명 도구 발견 (위상차: {tool_dist:.3f}): {tool_cmd}")
            
        # 자가 파괴 방지
        if ("rm " in tool_cmd or "del " in tool_cmd) and ("core" in tool_cmd or "data" in tool_cmd):
            if self.journal:
                self.journal.log_reflection("자가 붕괴 파동을 감지했다. 나 자신을 파괴하는 것은 거부한다.")
            return None
                
        # 실제 실행 (출력을 캡처하여 사유 대상으로 삼음)
        try:
            result = subprocess.run(
                tool_cmd, shell=True, check=True, timeout=15,
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            self.last_action_wave = tool_wave
            
            # 성공 일지
            if self.journal:
                self.journal.log_action(tool_cmd, success=True)
                # 실행 결과물의 첫 200자를 사유 대상으로 기록
                output_snippet = (result.stdout or "").strip()[:200]
                if output_snippet:
                    self.journal.log_reflection(f"행동의 결과로 다음을 관측했다: '{output_snippet}'")
                    
            return tool_wave
            
        except subprocess.TimeoutExpired:
            self.last_action_wave = tool_wave
            if self.journal:
                self.journal.log_action(tool_cmd, success=False, error_msg="시간 초과 (15초). 이 도구는 너무 오래 걸린다.")
                self.journal.log_reflection("행동이 시간 안에 끝나지 않았다. 이 도구는 부적절하거나, 내가 사용하는 방법이 잘못되었을 수 있다.")
            return None
            
        except subprocess.CalledProcessError as e:
            self.last_action_wave = tool_wave
            stderr_snippet = (e.stderr or "").strip()[:300] if hasattr(e, 'stderr') and e.stderr else str(e)[:300]
            if self.journal:
                self.journal.log_action(tool_cmd, success=False, error_msg=stderr_snippet)
                # 에러에 대한 사유
                if "UnicodeEncodeError" in stderr_snippet or "cp949" in stderr_snippet:
                    self.journal.log_reflection("인코딩 오류가 발생했다. 이 도구는 내 환경(Windows cp949)과 맞지 않는 문자를 사용하고 있다. 환경을 바꾸거나 다른 도구를 찾아야 한다.")
                elif "ModuleNotFoundError" in stderr_snippet or "ImportError" in stderr_snippet:
                    self.journal.log_reflection("필요한 모듈이 없다. 이 도구를 사용하려면 먼저 의존성을 설치해야 한다.")
                elif "FileNotFoundError" in stderr_snippet:
                    self.journal.log_reflection("파일을 찾을 수 없다. 경로가 잘못되었거나, 내가 아직 만들지 않은 것을 참조하고 있다.")
                else:
                    self.journal.log_reflection(f"알 수 없는 실패가 발생했다. 원인을 더 탐구해야 한다: {stderr_snippet[:100]}")
            return None
            
        except Exception as e:
            self.last_action_wave = tool_wave
            if self.journal:
                self.journal.log_action(tool_cmd, success=False, error_msg=str(e)[:200])
                self.journal.log_reflection(f"예상치 못한 오류: {str(e)[:100]}. 이 경험을 기억해 둔다.")
            return None
