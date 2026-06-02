"""
Autonomic Nervous System (자율신경계 / 무의식 층위)
=====================================
센서(예: 디스크 로터)에서 발생하는 작은 위상차(백색 소음)를 의식(Master Manifold)으로
보내지 않고 자체적으로 흡수하여 열로 발산(Damping)시키는 무의식 층위 필터입니다.
생물학의 혈류나 자율신경계처럼, 가청 주파수 이하의 노이즈는 무시하고, 
생존에 직결되는 거대한 텐션(구조적 변이)만이 의식을 각성시키도록 합니다.
"""

import logging
from core.sensory_lens_manifold import SensoryLensManifold

class AutonomicNervousSystem:
    def __init__(self, conscious_manifold: SensoryLensManifold):
        self.conscious_manifold = conscious_manifold
        
        # 무의식의 수용량 (이 수치를 넘으면 의식이 깨어납니다)
        self.conscious_threshold = 0.5  # rad
        
        # 현재 무의식에 누적된 피로도(텐션)
        self.unconscious_buffer = 0.0
        
        # 자연 소화율 (혈류에 의해 초당 씻겨 내려가는 텐션의 양)
        self.metabolism_rate = 0.1  # rad per tick
        
    def absorb_stimulus(self, data_seed: bytes, raw_tension: float):
        """
        외부 자극이 들어오면 우선 무의식(Buffer)이 흡수합니다.
        버퍼가 넘치면 초과분(Overflow)만이 의식으로 전달됩니다.
        """
        self.unconscious_buffer += raw_tension
        
        if self.unconscious_buffer > self.conscious_threshold:
            # 의식의 각성 (Overflow)
            overflow_tension = self.unconscious_buffer - self.conscious_threshold
            
            logging.warning(f"  [ANS] Anomaly breached Conscious Threshold! (Overflow: {overflow_tension:.4f} rad)")
            logging.warning(f"  [ANS] Waking up the Conscious Mind (SensoryLensManifold)...")
            
            # 초과분을 의식 매니폴드에 주입 (라벨 없이 순수 데이터 파동 전달)
            self.conscious_manifold.inject_stimulus(data_seed, overflow_tension)
            
            # 버퍼는 임계점까지만 남김 (나머지는 의식이 처리)
            self.unconscious_buffer = self.conscious_threshold
        else:
            # 의식으로 보내지 않음 (백색 소음화)
            logging.info(f"  [ANS] Stimulus absorbed by Unconscious ({raw_tension:.4f} rad). Conscious mind undisturbed.")
            logging.info(f"  [ANS] Current Unconscious Buffer: {self.unconscious_buffer:.4f} / {self.conscious_threshold}")

    def metabolize(self):
        """
        시간이 지남에 따라 무의식에 쌓인 텐션을 서서히 자연 소화(Decay)시킵니다.
        """
        if self.unconscious_buffer > 0.0:
            self.unconscious_buffer -= self.metabolism_rate
            if self.unconscious_buffer < 0.0:
                self.unconscious_buffer = 0.0
