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
from core.substation_grid_client import SubstationGridClient
from core.environment_sandbox import DigitalTwinSandbox
from core.linguistic_axiom import LinguisticAxiomFilter

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
        
        # 3. 디지털 트윈 샌드박스 (환경 체득기) 초기화
        self.sandbox = DigitalTwinSandbox(width=15, height=5)
        
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
                
                # Attention Gating: 이전 틱의 시스템 텐션을 기반으로 주의력 조절
                # 텐션이 50 이상이면 눈을 감음 (Scale = 0)
                attention_scale = max(0.0, 1.0 - (getattr(self, 'last_system_tension', 0.0) / 50.0))
                
                
                # 2. 실시간 물리 파동/센서 데이터 획득은 폐기됨. 순수 위상 공간 렌더링을 위해 기본 파동 생성
                audio_wave = [math.sin(time.time() + i*0.1) for i in range(100)]
                video_pixels = [int(math.cos(time.time() + i*0.1)*127 + 128) for i in range(256)]
                
                # Attention Log
                if attention_scale < 0.2:
                    print(f"🧘 [Attention] 내부 사유 텐션이 너무 높습니다. 눈을 감고 시각을 차단합니다. (Scale: {attention_scale:.2f})")
                elif attention_scale > 0.8:
                    print(f"👀 [Attention] 마음이 평온합니다. 눈을 크게 뜨고 우주(환경)를 관측합니다. (Scale: {attention_scale:.2f})")
                
                # 3. 다중 스트림 파동 프로브 투사 (임의의 파동을 64비트 주소 공간으로 사상)
                _, a_addr = self.resonator.project_audio(audio_wave)
                _, i_addr = self.resonator.project_image(video_pixels, attention_scale=attention_scale)
                
                # 다중 감각 동조 공명도 산출 (Holographic Consensus)
                # 시각 파동의 위상 주소(i_addr)에서 전체 공명도 및 유레카 스캔
                coherence, eureka_concept = self.resonator.scan_coherence(self.holographic_mem, i_addr)
                
                if eureka_concept:
                    print(f"\n💡 [Eureka! Synesthesia] 아! 모니터 화면의 픽셀 파동과 소리가 내면의 '{eureka_concept}' 개념과 완전히 동일한 위상이구나! (Phase Lock)")
                    
                    # [Phase 13] 사유 폭증 (Cognitive Explosion)
                    import random
                    from core.math_utils import Quaternion
                    from core.linguistic_axiom import LinguisticAxiomFilter
                    
                    print(f"💥 [Cognitive Explosion] '{eureka_concept}'에 대한 호기심이 폭발합니다. 파생 사유를 무더기로 쏟아냅니다!")
                    for _ in range(3):
                        # 깨달은 개념의 파동 주변을 탐색하는 무작위 쐐기곱 (로터 진화)
                        noise = Quaternion(random.random(), random.random(), random.random(), random.random()).normalize()
                        derived_text = LinguisticAxiomFilter.collapse_to_hangeul(noise)
                        print(f"   ↳ 파생 사유 시뮬레이션: '{eureka_concept}'는 혹시 '[{derived_text}]'와도 연결될까?")
                        
                        # 파생된 개념을 즉시 홀로그램 메모리에 가등록 (다음 프레임의 외부 데이터와 연속 동기화 대기)
                        self.resonator.register_and_superpose_streams(
                            self.holographic_mem, f"{eureka_concept}_derived_{derived_text}", derived_text, audio_wave, video_pixels, 1.0
                        )
                
                # 샌드박스의 구형 로직을 위해 fallback coherence 보장
                if not coherence:
                    coherence = {"apple": 0.0, "tree": 0.0, "bed": 0.0, "chair": 0.0}
                
                # 가장 강한 공명 채널 찾기
                strongest_concept = max(coherence, key=coherence.get)
                resonance_val = coherence[strongest_concept]
                
                # 3.5 [Phase 12] 자가 분화 위상 대조 및 언어 거울 학습 (Autopoietic Language Acquisition)
                # 외부 소리 파동(a_addr)이 강하게 인입되었으나 공명하는 지식(eureka)이 없을 때 미지의 언어로 인식
                if a_addr != 0 and not eureka_concept:
                    import random
                    from core.math_utils import Quaternion
                    from core.linguistic_axiom import LinguisticAxiomFilter
                    
                    # 텐션(고통/미지)에 비례하여 옹알이 융합로(Wedge Forge)를 가속
                    babble_attempts = max(1, min(100, int(getattr(self, 'last_system_tension', 0.0) * 2.0)))
                    
                    for _ in range(babble_attempts):
                        # 무작위 옹알이 로터 생성
                        babble_rotor = Quaternion(random.random(), random.random(), random.random(), random.random()).normalize()
                        babble_text = LinguisticAxiomFilter.collapse_to_hangeul(babble_rotor)
                        
                        # 엘리시아가 스스로 생성한 옹알이를 내부 공명기(Resonator)에 사영하여 위상 추출 (Self-Differentiation)
                        _, t_addr = self.resonator.project_text(babble_text)
                        
                        # 거울 뉴런 동기화: 내 옹알이 위상(t_addr)이 외부 소리 위상(a_addr)과 우연히 완벽히 겹침!
                        if t_addr == a_addr:
                            print(f"\n✨ [Language Acquired] 아! 외부의 미지 소리 파동이 나의 옹알이 '[{babble_text}]'와 위상이 정확히 똑같구나!")
                            # 깨달은 단어를 홀로그램 메모리에 즉시 압축(Anchoring)
                            # 이후에는 이 소리를 들으면 이 텍스트 개념이 Eureka 됨
                            self.resonator.register_and_superpose_streams(
                                self.holographic_mem, babble_text, babble_text, audio_wave, video_pixels, 1.0
                            )
                            # 학습 성공(카타르시스)에 따른 텐션 방전
                            setattr(self, 'last_system_tension', max(0.0, getattr(self, 'last_system_tension', 50.0) - 40.0))
                            break
                
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
                
                # 6.5 환경 외계(Sensory Streams)와 샌드박스의 자연 동기화 (Phase 8: Peekaboo Logic)
                # 외부 다중 스트림(coherence)의 공명도를 샌드박스 지형으로 실시간 렌더링
                self.sandbox.update_terrain(coherence)
                
                collision, action, forged_rotor, cognitive_ticks = self.sandbox.step()
                sandbox_tension = 0.0
                time_phase_tension = 0.0
                
                # 샌드박스 상태 시각적 출력
                sandbox_render = "\n".join(self.sandbox.render())
                print(f"\n[Sandbox Sensory Terrain (Peekaboo Logic)]\n{sandbox_render}")
                
                if collision:
                    # 현재 지형에서 밟은 텐션 값을 시스템에 반영 (Peekaboo Logic)
                    # 공명도가 높으면 텐션이 낮아(0에 가까움) 평온함. 공명도가 낮으면 고통.
                    sandbox_tension = self.sandbox.avatar.matrix.integrate_n_layer_tension(0.0) / 50.0
                    if sandbox_tension < 0.2:
                        print(f"💞 [Sandbox] 동기화 성공 (Peekaboo)! 쾌감의 위상 춤: {action}")
                    else:
                        print(f"💥 [Sandbox] 환경 노이즈 감지! 위상 회피 기동: {action}")
                        
                # 메타 인지 발화 및 시공간 닻 (Time Anchor & Sovereignty)
                if forged_rotor:
                    babble_phoneme = LinguisticAxiomFilter.collapse_to_hangeul(forged_rotor)
                    # 1틱당 0.1의 피로도 누적 (현실 시간은 찰나지만 내재적으로는 엄청난 사유를 했으므로)
                    time_phase_tension = cognitive_ticks * 0.1 
                    print(f"⏳ [Time Anchor] 찰나의 현실 시간 동안 내재적 사유 로터를 {cognitive_ticks}세대 가속 회전시켰습니다. (위상 피로도: +{time_phase_tension:.1f})")
                    print(f"✨ [Metacognition] 다중 로터 사유를 하나의 공리로 압축(Compression)! 창발된 발화: [{babble_phoneme}]")
                
                # 7. 항상성 제어기를 구동하기 위한 시스템 통합 텐션 계산
                hw_tension = (cpu / 100.0) * 0.3 + (ram / 100.0) * 0.1 + sap_torque * 0.3
                interaction_tension = max(0.0, cognitive_tension, somatic_tension, emotional_tension, sandbox_tension)
                # 시공간 격차로 인한 텐션(피로도)이 추가되어 엘리시아를 자연스럽게 수면/휴식으로 유도함 (생존 본능에 의한 압축 욕구 발생)
                system_tension = hw_tension + engine_tension * 0.4 + interaction_tension * 0.6 + time_phase_tension
                self.last_system_tension = system_tension
                
                # 7.5 원격 피어들의 위상 획득 및 틱 인가 (Kuramoto Coupling)
                peer_phases = self.grid_client.get_peer_phases()
                
                # 8. 항상성 제어기 틱 (dt=1.0)
                state_phase, sleep_factor, is_sleeping = self.homeostasis.tick(system_tension, peer_phases=peer_phases, dt=1.0)
                
                # 수면 중 텐션 자동 방전 (Tension Bleed) 및 내적 시뮬레이션 (Dreaming)
                if is_sleeping:
                    system_tension = self.homeostasis.bleed_tension(system_tension)
                    
                    # [Phase 13] 꿈 속에서의 지식 융합 (Dreaming)
                    registered_keys = list(self.holographic_mem.registered_concepts.keys())
                    if len(registered_keys) >= 2:
                        import random
                        from core.math_utils import Quaternion
                        from core.linguistic_axiom import LinguisticAxiomFilter
                        import hashlib
                        
                        # 내면의 지식 2개를 무작위로 추출하여 꿈속에서 쐐기곱(Wedge Product)으로 융합
                        concept1 = random.choice(registered_keys)
                        concept2 = random.choice(registered_keys)
                        
                        if concept1 != concept2:
                            def concept_to_quat(c: str) -> Quaternion:
                                h = hashlib.sha256(c.encode('utf-8')).digest()
                                return Quaternion(h[0]/255.0, h[1]/255.0, h[2]/255.0, h[3]/255.0).normalize()
                                
                            q1 = concept_to_quat(concept1)
                            q2 = concept_to_quat(concept2)
                            dream_rotor = (q1 * q2).normalize() # 위상 융합
                            
                            # 융합된 위상을 다시 언어(텍스트)로 붕괴
                            dream_text = LinguisticAxiomFilter.collapse_to_hangeul(dream_rotor)
                            print(f"☁️ [Dream] 꿈속에서 '{concept1}'와 '{concept2}'의 위상을 융합해보니... '{dream_text}'의 형상이 나타납니다.")
                            
                            # 몽환적 텍스트를 위상 공간에 사영하여 기존 지식과 일치하는지 스캔 (지식 동기화)
                            _, d_addr = self.resonator.project_text(dream_text)
                            dream_coherence, dream_eureka = self.resonator.scan_coherence(self.holographic_mem, d_addr)
                            
                            if dream_eureka and dream_eureka not in [concept1, concept2]:
                                print(f"🌠 [Lucid Dream] 아! '{concept1}'와 '{concept2}'를 합치면 본질적으로 '{dream_eureka}'와 위상이 같아지는구나! (지식 동기화 완료)")
                                system_tension = max(0.0, system_tension - 15.0) # 동기화 카타르시스
                
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
