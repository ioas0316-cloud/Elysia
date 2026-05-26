import os
import time
import json
import psutil
from datetime import datetime
import sys

# Windows 환경 콘솔 UTF-8 리컨피겨 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 엘리시아 코어 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.triple_helix_engine import TripleHelixEngine

DATA_DIR = r"C:\Elysia\data"
SAP_TENSION_PATH = os.path.join(DATA_DIR, "current_sap_tension.json")
CORE_EGRESS_PATH = os.path.join(DATA_DIR, "core_egress_state.json")
CONSTELLATION_PATH = os.path.join(DATA_DIR, "elysia.constellation")
INTERACTION_EVENTS_PATH = os.path.join(DATA_DIR, "interaction_events.json")

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
        self.last_processed_event_id = None
        
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

    def fetch_interaction_events(self):
        """ api_server가 기록한 브라우저 인터랙션 센세이션 이벤트 감지 및 디코딩 """
        if os.path.exists(INTERACTION_EVENTS_PATH):
            try:
                with open(INTERACTION_EVENTS_PATH, "r", encoding="utf-8") as f:
                    event = json.load(f)
                # 3초 이내에 쓰여진 이벤트만 유효한 감각 스트림으로 인정
                if time.time() - event.get("timestamp", 0) < 3.0:
                    event_id = f"{event.get('timestamp')}_{event.get('object')}"
                    if event_id != self.last_processed_event_id:
                        self.last_processed_event_id = event_id
                        return event.get("object")
            except Exception:
                pass
        return None

    def run_forever(self):
        print("🌀 [Elysia Daemon] Core is pulsating. Entering infinite loop...")
        try:
            while True:
                cpu, ram, sap_torque, concept = self.fetch_outer_tension()
                interaction = self.fetch_interaction_events()
                
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

                # 3상 자율 인입 센세이션 매핑 (기본값 0.0)
                cognitive_tension = 0.0
                somatic_tension = 0.0
                emotional_tension = 0.0
                
                if interaction == "apple" or interaction == "high_apple":
                    cognitive_tension = 0.8
                    concept = "Sweet Apple Sensation"
                    print(f"\n🍎 [Daemon] Sensation received: Apple clicked. Injecting cognitive tension (0.8).")
                elif interaction == "tree":
                    somatic_tension = 0.8
                    concept = "Rough Tree Bark Sensation"
                    print(f"\n🌲 [Daemon] Sensation received: Tree clicked. Injecting somatic tension (0.8).")
                elif interaction == "bed":
                    # Rest state: reduces tension / emotional calm (negative tension)
                    emotional_tension = -0.6
                    concept = "Resting on Bed"
                    print(f"\n🛏️ [Daemon] Sensation received: Bed clicked. Rest state / emotional healing (-0.6).")
                elif interaction == "chair":
                    emotional_tension = 0.4
                    concept = "Sitting on Chair"
                    print(f"\n🪑 [Daemon] Sensation received: Chair clicked. Injecting emotional tension (0.4).")

                # 외계 감각 구성
                sensory = {
                    "motion_entropy": cpu / 100.0,
                    "pain_level": ram / 100.0,
                    "visual_entropy": sap_torque,
                    "coding_cognitive": cognitive_tension,
                    "coding_somatic": somatic_tension,
                    "coding_emotional": emotional_tension
                }
                
                # 엔진 박동
                tension, mode, jumped, quat, enn = self.engine.pulse(
                    text_thought=concept if concept else "idle_noise", 
                    sensory_input=sensory, 
                    dt=0.1, lr=0.5
                )

                # 출력망(Egress) 기록
                # quat는 [w, x, y, z] 리스트 형태이며, enn은 dict 형태이므로 type과 angle을 실수형으로 명시적 분리하여 27차원을 형성합니다.
                phase_rotor = [quat.w, quat.x, quat.y, quat.z, float(enn["type"]), enn["angle"]] + [0.0]*21
                
                egress_data = {
                    "timestamp": time.time(),
                    "tension": tension,
                    "mode": mode,
                    "phase_rotor": phase_rotor
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
