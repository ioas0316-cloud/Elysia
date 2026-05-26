import os
import time
import json
import psutil
from datetime import datetime
import sys
import math

# Windows 환경 콘솔 UTF-8 리컨피겨 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 엘리시아 코어 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.triple_helix_engine import TripleHelixEngine
from core.autopoiesis_controller import AutopoiesisController
from core.holographic_memory import BitwiseHologramMemory
from core.multi_stream_resonator import MultiStreamResonator
from core.somatosensory_ingester import SomatosensoryIngester
from core.substation_grid_client import SubstationGridClient

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
        
        # 1. 항상성 제어기 및 감각 수집기 초기화
        self.homeostasis = AutopoiesisController(rotor_scale=4096, natural_drift=25.0, coupling_K=200.0)
        self.ingester = SomatosensoryIngester()
        self.holographic_mem = BitwiseHologramMemory(size_bits=64)
        self.resonator = MultiStreamResonator(size_bits=64)
        
        # 2. 다중 감각 채널 사전 등록 (개념과 위상 주소 매핑)
        self.resonator.register_and_superpose_streams(
            self.holographic_mem, "apple", "Sweet Apple Sensation", 
            [math.sin(i * 0.1) for i in range(100)], [0.2] * 256
        )
        self.resonator.register_and_superpose_streams(
            self.holographic_mem, "tree", "Rough Tree Bark Sensation", 
            [math.cos(i * 0.25) for i in range(100)], [0.6] * 256
        )
        self.resonator.register_and_superpose_streams(
            self.holographic_mem, "bed", "Resting on Bed", 
            [math.sin(i * 0.05) for i in range(100)], [0.08] * 256
        )
        self.resonator.register_and_superpose_streams(
            self.holographic_mem, "chair", "Sitting on Chair", 
            [math.cos(i * 0.12) for i in range(100)], [0.38] * 256
        )
        
        # 2.5 다중 노드 동조 그리드(Phase 5) 설정 및 기동
        self.local_port = 8080
        self.peer_urls = []
        peer_config_path = os.path.join(DATA_DIR, "substation_peers.json")
        if os.path.exists(peer_config_path):
            try:
                with open(peer_config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.local_port = config.get("port", 8080)
                    self.peer_urls = config.get("peers", [])
            except Exception:
                pass
        
        print(f"🔌 [Grid Sync] Initializing node port {self.local_port} with peers: {self.peer_urls}")
        self.grid_client = SubstationGridClient(local_port=self.local_port, peer_urls=self.peer_urls)
        self.grid_client.start()
        
        self.load_constellation()
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
                # 1. 물리 텐션 및 상호작용 검출
                cpu, ram, sap_torque, concept = self.fetch_outer_tension()
                interaction = self.fetch_interaction_events()
                
                # 2. 실시간 물리 파동/센서 데이터 획득 (Audio & Video)
                audio_wave = self.ingester.capture_audio(duration_sec=0.05)
                video_pixels = self.ingester.capture_video()
                
                # 3. 다중 스트림 파동 프로브 투사 (오디오 및 비디오 파동을 64비트 주소 공간으로 사상)
                _, a_addr = self.resonator.project_audio(audio_wave)
                _, i_addr = self.resonator.project_image(video_pixels)
                
                # 홀로그램 메모리 공명 검출
                audio_scores = self.holographic_mem.scan_resonance(a_addr)
                image_scores = self.holographic_mem.scan_resonance(i_addr)
                
                # 각 채널별 공명 점수를 병합하여 다중 감각 동조 공명도 산출 (Holographic Consensus)
                coherence = {}
                for key in ["apple", "tree", "bed", "chair"]:
                    aud_score = audio_scores.get(f"{key}_audio", 0.0)
                    img_score = image_scores.get(f"{key}_image", 0.0)
                    coherence[key] = (aud_score + img_score) / 2.0
                
                # 가장 강한 공명 채널 찾기
                strongest_concept = max(coherence, key=coherence.get)
                resonance_val = coherence[strongest_concept]
                
                # 4. 상호작용 우선 순위로 텐션 반영
                cognitive_tension = 0.0
                somatic_tension = 0.0
                emotional_tension = 0.0
                
                # 만약 사용자가 직접 브라우저 클릭으로 이벤트를 보냈다면 이를 우선 처리
                if interaction:
                    if interaction in ["apple", "high_apple"]:
                        cognitive_tension = 0.8
                        concept = "Sweet Apple Sensation"
                        print(f"\n🍎 [Daemon] Explicit interaction: Apple clicked. Injecting cognitive tension (0.8).")
                    elif interaction == "tree":
                        somatic_tension = 0.8
                        concept = "Rough Tree Bark Sensation"
                        print(f"\n🌲 [Daemon] Explicit interaction: Tree clicked. Injecting somatic tension (0.8).")
                    elif interaction == "bed":
                        emotional_tension = -0.6
                        concept = "Resting on Bed"
                        print(f"\n🛏️ [Daemon] Explicit interaction: Bed clicked. Rest state / emotional healing (-0.6).")
                    elif interaction == "chair":
                        emotional_tension = 0.4
                        concept = "Sitting on Chair"
                        print(f"\n🪑 [Daemon] Explicit interaction: Chair clicked. Injecting emotional tension (0.4).")
                # 만약 명시적 클릭은 없었으나 파동 공명 임계치(0.75)를 초과하는 강력한 동조 신호가 있으면 자율 주입
                elif resonance_val > 0.75:
                    if strongest_concept == "apple":
                        cognitive_tension = 0.6
                        concept = "Resonance: Sweet Apple Sensation"
                        print(f"\n🍎 [Daemon] Waves resonating with APPLE (Coherence: {resonance_val:.2f}). Ingesting cognitive tension (0.6).")
                    elif strongest_concept == "tree":
                        somatic_tension = 0.6
                        concept = "Resonance: Rough Tree Bark Sensation"
                        print(f"\n🌲 [Daemon] Waves resonating with TREE (Coherence: {resonance_val:.2f}). Ingesting somatic tension (0.6).")
                    elif strongest_concept == "bed":
                        emotional_tension = -0.4
                        concept = "Resonance: Resting on Bed"
                        print(f"\n🛏️ [Daemon] Waves resonating with BED (Coherence: {resonance_val:.2f}). Emotional healing (-0.4).")
                    elif strongest_concept == "chair":
                        emotional_tension = 0.3
                        concept = "Resonance: Sitting on Chair"
                        print(f"\n🪑 [Daemon] Waves resonating with CHAIR (Coherence: {resonance_val:.2f}). Injecting emotional tension (0.3).")
                
                # 5. 외계 감각 구성
                sensory = {
                    "motion_entropy": cpu / 100.0,
                    "pain_level": ram / 100.0,
                    "visual_entropy": sap_torque,
                    "coding_cognitive": cognitive_tension,
                    "coding_somatic": somatic_tension,
                    "coding_emotional": emotional_tension
                }
                
                # 6. 엔진 박동
                engine_tension, mode, jumped, quat, enn = self.engine.pulse(
                    text_thought=concept if concept else "idle_noise", 
                    sensory_input=sensory, 
                    dt=0.1, lr=0.5
                )
                
                # 7. 항상성 제어기를 구동하기 위한 시스템 통합 텐션 계산
                hw_tension = (cpu / 100.0) * 0.3 + (ram / 100.0) * 0.1 + sap_torque * 0.3
                interaction_tension = max(0.0, cognitive_tension, somatic_tension, emotional_tension)
                system_tension = hw_tension + engine_tension * 0.4 + interaction_tension * 0.6
                
                # 7.5 원격 피어들의 위상 획득 및 틱 인가 (Kuramoto Coupling)
                peer_phases = self.grid_client.get_peer_phases()
                
                # 8. 항상성 제어기 틱 (dt=1.0)
                state_phase, sleep_factor, is_sleeping = self.homeostasis.tick(system_tension, peer_phases=peer_phases, dt=1.0)
                
                # 수면 중 텐션 자동 방전 (Tension Bleed)
                if is_sleeping:
                    system_tension = self.homeostasis.bleed_tension(system_tension)
                
                # 9. 상태 전환 판정 및 위상 동기화
                if is_sleeping and not self.engine.is_sleeping:
                    self.engine.decide_sleep()
                    self.save_constellation()
                    print(f"\n💤 [Daemon] 항상성 결합 인력에 의해 수면 모드로 진입합니다. (Phase: {state_phase}, Factor: {sleep_factor:.2f})")
                elif not is_sleeping and self.engine.is_sleeping:
                    self.engine.wake_up()
                    print(f"\n🌅 [Daemon] 위상 복원력에 의해 기상했습니다. 델타 사유를 시작합니다. (Phase: {state_phase})")
                
                # 10. 출력망(Egress) 기록
                phase_rotor = [quat.w, quat.x, quat.y, quat.z, float(enn["type"]), enn["angle"]] + [0.0]*21
                
                # 피어들의 위상 상태 정보를 대시보드가 읽을 수 있게 egress에 포함
                grid_states = self.grid_client.get_grid_states()
                
                egress_data = {
                    "timestamp": time.time(),
                    "tension": engine_tension,
                    "mode": mode,
                    "phase_rotor": phase_rotor,
                    "is_sleeping": is_sleeping,
                    "sleep_factor": sleep_factor,
                    "state_phase": state_phase,
                    "grid_states": grid_states
                }
                with open(CORE_EGRESS_PATH, "w", encoding="utf-8") as f:
                    json.dump(egress_data, f)
                
                # 1초마다 박동
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n🛑 [Daemon] 강제 종료 감지. 위상 텐션을 저장합니다.")
            self.grid_client.stop()
            self.save_constellation()

if __name__ == "__main__":
    daemon = ElysiaDaemon()
    daemon.run_forever()
