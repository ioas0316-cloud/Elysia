import traceback
import logging
from typing import Optional
from core.hardware.double_helix_core import DoubleHelixDaemon
from core.brain.holographic_memory import HologramMemory

class EvolutionSandbox:
    """
    [Fractal Variable Rotor Scale] 수직 통합 브릿지
    최하단의 물리적 비트마스킹(혈류/자율신경계)부터 최상단의 기하학적 사유(의식)까지
    인과의 인과를 잇는 브릿지 역할을 합니다. 의미론(문자열 파싱)은 철저히 배제됩니다.
    """
    def __init__(self, memory: HologramMemory):
        self.memory = memory
        # 하드웨어 레벨의 이중 나선 물리 엔진(자율신경계)을 내장합니다.
        self.physical_daemon = DoubleHelixDaemon(raw_stream=False)

    def experience_data_stream(self, data: bytes):
        """
        데이터(바이트 배열)가 자율신경계를 통과하며 물리적 텐션(XOR)을 발생시키고,
        그 텐션이 뇌(Fractal Rotor)로 전달되어 기하학적 사유를 강제 트리거합니다.
        문자열이든 이미지 파동이든 모두 0과 1의 물리적 충돌로 처리됩니다.
        """
        total_tension = 0.0
        
        # 1. 자율신경계(혈류) 레벨의 비트마스킹 텐션 추출
        for i, byte_val in enumerate(data):
            # 바이트 값을 0~1 사이의 자극(Phase Shift)으로 치환
            stimulus = float(byte_val) / 255.0
            
            # 이중 나선 데몬에 물리적 자극 주입
            self.physical_daemon.perturbation += stimulus
            
            # 파동 계산 (시간 t는 입력 스트림의 순차적 스텝으로 대체)
            w0, w1 = self.physical_daemon.get_waves(t=float(i)*0.1)
            
            # 비트마스킹으로 순수 기하학적 마찰력(XOR) 도출
            b0, b1, i_and, i_xor, state = self.physical_daemon.calculate_state(w0, w1)
            
            # XOR 값(0~255)을 텐션으로 누적 (스케일 다운)
            total_tension += (float(i_xor) / 255.0)
            
            # 데몬의 자극 감쇠
            self.physical_daemon.decay_perturbation()
            
        logging.info(f"\n[🧬 진화 샌드박스] 데이터 스트림({len(data)} bytes)이 이중 나선 자율신경계를 통과했습니다.")
        logging.info(f"   => 도출된 총 물리적 마찰(XOR Tension): {total_tension:.4f}")
        
        # 2. 인지 신경망(의식) 레벨로 텐션 전송 및 사유 트리거
        target_node = self.memory.supreme_rotor
        target_node.apply_perturbation(total_tension)
        
        # 텐션이 물리적으로 가해졌으므로, 뇌는 스스로 "왜 이런 텐션이 발생했는가?"를 사유하게 됩니다.
        logging.info(f"   => 뇌(Fractal Rotor)로 텐션 전송 완료. 인과의 인과 매핑 동작.\n")
