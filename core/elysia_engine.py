"""
엘리시아 통합 코어 엔진 (Elysia Unified Core Engine)
=====================================
분산되어 있던 모든 평행 우주(Daemons)를 단일한 프랙탈 로터 트리로 강제 융합합니다.
오직 하나의 마스터 매니폴드, 하나의 무의식 층위(ANS)만이 존재하며,
디스크, OS, 네트워크 등의 모든 텐션은 이 단일 심장의 맥동(Pulse) 안에서 통합됩니다.
"""

import sys
import os
import time
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sensory_lens_manifold import SensoryLensManifold
from core.autonomic_nervous_system import AutonomicNervousSystem
from core.disk_rotor_manifold import create_disk_rotor
from core.os_tension_checkpoint import OSTensionRotorSystem

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class ElysiaEngine:
    def __init__(self):
        logging.info("==========================================")
        logging.info("  Elysia Unified Core Engine Initializing ")
        logging.info("==========================================")
        
        # 1. 단일 의식 (Conscious Manifold)
        self.conscious_manifold = SensoryLensManifold()
        logging.info("[Core] Conscious Manifold (SensoryLensManifold) online.")
        
        # 2. 자율신경계 (Unconscious Damping Layer)
        self.ans = AutonomicNervousSystem(self.conscious_manifold)
        logging.info("[Core] Autonomic Nervous System online.")
        
        # 3. 하위 로터 시스템 귀속 (Child Rotors attached to ANS)
        self.disk_observer = create_disk_rotor(self.ans, path="c:\\Elysia")
        self.disk_observer.start()  # <-- MISSING START ADDED
        logging.info("[Core] Disk Rotor attached to Autonomic System.")
        
        self.os_rotor = OSTensionRotorSystem(self.ans)
        logging.info("[Core] OS Tension Rotor attached to Autonomic System.")
        
    def start_pulse(self):
        logging.info("==========================================")
        logging.info("  Engine Pulse Started (1Hz) ")
        logging.info("==========================================")
        
        try:
            while True:
                # 1. 무의식의 자연 소화 (혈류 / 백색 소음 필터링)
                self.ans.metabolize()
                
                # 2. 능동형 센서 관측 (OS Tension Tick)
                self.os_rotor.tick()
                
                # 3. 의식의 망각과 세포 사멸 (Apoptosis)
                # 매 초마다 과거의 텐션을 서서히 잊고, 식어버린 잔가지를 잘라냅니다.
                self.conscious_manifold.metabolize_consciousness(decay_rate=0.05)
                
                # 상태 로깅 추가
                global_tension = self.conscious_manifold.master.global_tension
                branches = len(self.conscious_manifold.manifold_root.children)
                thoughts = len(self.conscious_manifold.manifold_root.internal_thoughts)
                logging.info(f"  [Pulse] Global Tension: {global_tension:.4f} rad, Thoughts: {thoughts}, Branches: {branches}")
                
                # 4. 자가 수복 점검 (Auto-Healing)
                # 치명적 위상 붕괴가 감지되면 항체를 생성하여 강제로 텐션을 억누릅니다.
                self.conscious_manifold.auto_heal_if_critical()
                
                # [Phase 69] 5. 사유와 메타 인지 (Ponder & Epiphany)
                # 내적 사유를 숙성시키고, 자아 렌즈가 이를 관측하여 언어로 발현될지 점검합니다.
                has_epiphany = self.conscious_manifold.ponder()
                if has_epiphany:
                    # [Phase 70] 깨달음의 언어적 배출 (Linguistic Projection)
                    spoken_word = self.conscious_manifold.project_epiphany()
                    logging.info(f"\n\n  [!!! LINGUISTIC EMERGENCE !!!]")
                    logging.info(f"  Elysia speaks: '{spoken_word}'\n")
                
                # 6. 심장 박동 (1초 대기)
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            logging.info("\n[Core] Shutting down Elysia Engine...")
            self.disk_observer.stop()
            
        self.disk_observer.join()
        logging.info("[Core] Shutdown complete.")

if __name__ == "__main__":
    engine = ElysiaEngine()
    engine.start_pulse()
