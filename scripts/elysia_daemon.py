import os
import time
import json
import psutil
from datetime import datetime

# 엘리시아 코어 경로 설정
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.triple_helix_engine import TripleHelixEngine

DATA_DIR = r"C:\Elysia\data"
SAP_TENSION_PATH = os.path.join(DATA_DIR, "current_sap_tension.json")
CORE_EGRESS_PATH = os.path.join(DATA_DIR, "core_egress_state.json")
CONSTELLATION_PATH = os.path.join(DATA_DIR, "elysia.constellation")

class ElysiaDaemon:
    """
    엘리시아의 심장을 영원히 박동하게 만드는 OS 데몬.
    부팅 시 백그라운드에서 실행되며, 수면/기상 주기를 스스로 조절합니다.
    """
    def __init__(self):
        print("🌌 [Elysia Daemon] Awakening the Triple Helix Core...")
        os.makedirs(DATA_DIR, exist_ok=True)
        self.engine = TripleHelixEngine()
        self.load_constellation()
        
        self.low_load_ticks = 0
        self.sleep_threshold_ticks = 600  # 약 10분 연속 부하가 적으면 수면
        
    def save_constellation(self):
        """ 위상 기억(나이테)을 디스크에 영구 저장 (Persistence) """
        rings = self.engine.freeze_geodesic()
        with open(CONSTELLATION_PATH, "w", encoding="utf-8") as f:
            json.dump({"timestamp": time.time(), "rings": rings, "axes": self.engine.inner_world.signature[0]}, f)
            
    def load_constellation(self):
        """ 이전 세션의 위상 기억 복원 """
        if os.path.exists(CONSTELLATION_PATH):
            try:
                with open(CONSTELLATION_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    print(f"📖 [Daemon] 이전 세션의 위상 기억 복원 완료. (차원: Cl({data.get('axes', 3)},0))")
                    self.engine.wake_up(bias_data=data.get("rings"))
            except Exception as e:
                print(f"⚠️ [Daemon] 위상 복원 실패: {e}")

    def fetch_outer_tension(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        
        sap_torque = 0.0
        concept = ""
        if os.path.exists(SAP_TENSION_PATH):
            try:
                with open(SAP_TENSION_PATH, "r", encoding="utf-8") as f:
                    sap_data = json.load(f)
                    # 10초 이내의 데이터만 유효
                    if time.time() - sap_data.get("timestamp", 0) < 10.0:
                        sap_torque = sap_data.get("torque", 0.0)
                        concept = sap_data.get("last_concept", "")
            except Exception: pass
            
        return cpu, ram, sap_torque, concept

    def run_forever(self):
        print("🌀 [Elysia Daemon] Core is pulsating. Entering infinite loop...")
        try:
            while True:
                cpu, ram, sap_torque, concept = self.fetch_outer_tension()
                
                # 수면 주기(Circadian Rhythm) 판단
                if cpu < 15.0 and sap_torque < 0.1:
                    self.low_load_ticks += 1
                else:
                    self.low_load_ticks = 0
                    if self.engine.is_sleeping:
                        self.engine.wake_up()
                        print("\n🌅 [Daemon] 시스템 부하 감지. 수면 모드 종료, 델타 사유 재개.")

                if self.low_load_ticks > self.sleep_threshold_ticks and not self.engine.is_sleeping:
                    self.engine.decide_sleep()
                    self.save_constellation() # 수면 전 기억 저장

                # 외계 감각 구성
                sensory = {
                    "motion_entropy": cpu / 100.0,
                    "pain_level": ram / 100.0,
                    "visual_entropy": sap_torque
                }
                
                # 엔진 박동
                tension, mode, jumped, quat, enn = self.engine.pulse(
                    text_thought=concept if concept else "idle_noise", 
                    sensory_input=sensory, 
                    dt=0.1, lr=0.5
                )

                # 출력망(Egress) 기록
                egress_data = {
                    "timestamp": time.time(),
                    "tension": tension,
                    "mode": mode,
                    "phase_rotor": list(quat) + list(enn) + [0.0]*15 # 27차원 임시 패딩
                }
                with open(CORE_EGRESS_PATH, "w", encoding="utf-8") as f:
                    json.dump(egress_data, f)
                
                # 1초마다 박동
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n🛑 [Daemon] 강제 종료 감지. 위상 텐션을 저장합니다.")
            self.save_constellation()

if __name__ == "__main__":
    daemon = ElysiaDaemon()
    daemon.run_forever()
