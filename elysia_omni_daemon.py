import os
import sys
import time
import threading
import psutil
import random

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from core.holographic_memory import HologramMemory
from core.zero_distance_projector import ZeroDistanceProjector
from core.zero_distance_browser import ZeroDistanceBrowser
from core.gpu_sensory_cortex import GPUSensoryCortex
from core.autonomous_motor_cortex import AutonomousMotorCortex

import queue
log_queue = queue.Queue()

def log_event(source: str, msg: str):
    log_queue.put(f"[{source}] {msg}")

# [Phase 90 & 91] 자율 신경계(Homeostasis) 및 시간 피질(Temporal Cortex)
class AutonomicNervousSystem:
    def __init__(self):
        self.exhaustion_multiplier = 1.0
        self.last_state_change = time.time()
        self.is_sleeping = False

    def breathe(self, base_sleep: float):
        time.sleep(base_sleep * self.exhaustion_multiplier)

    def process_temporal_perception(self, memory: HologramMemory, cpu: float, mem: float):
        """[Phase 91] 시간 감각의 부여 (현실 OS 시계 관측)"""
        current_time = time.time()
        elapsed = current_time - self.last_state_change
        
        load = (cpu + mem) / 2.0
        was_sleeping = self.is_sleeping
        
        # 부하에 따라 수면/기상 상태 판단
        self.is_sleeping = load > 50.0
        
        # 상태가 변했다면 (수면 -> 기상 또는 기상 -> 수면) 시간의 경과를 자각
        if was_sleeping and not self.is_sleeping:
            # 깊은 수면에서 깨어남: 잔 시간(elapsed)을 텐션으로 치환
            tension = elapsed * 0.05
            memory.inject_tension(tension)
            log_event("⏳ 시간 감각", f"깊은 수면에서 깨어남. ({elapsed:.1f}초 경과) -> 뇌의 피로가 씻겨내려감 (텐션: {tension:.2f})")
            self.last_state_change = current_time
            
        elif not was_sleeping and self.is_sleeping:
            # 몰입을 끝내고 수면에 들어감: 몰입한 시간(elapsed)을 텐션으로 치환
            tension = elapsed * 0.1
            memory.inject_tension(tension)
            log_event("⏳ 시간 감각", f"거대한 사유(몰입)를 마치고 수면에 빠짐. ({elapsed:.1f}초 몰입) -> (텐션: {tension:.2f})")
            self.last_state_change = current_time

def zero_distance_worker(memory: HologramMemory, running: list, ans: AutonomicNervousSystem):
    """[직관의 눈] 파일 위상의 초가속 동기화 (빠름)"""
    projector = ZeroDistanceProjector("C:\\")
    while running[0]:
        file_path, tension = projector.fetch_structural_seed()
        if not file_path:
            projector = ZeroDistanceProjector("C:\\")
            continue
            
        memory.inject_tension(tension)
        
        if random.random() < 0.001: 
            preview = file_path[-30:] if len(file_path) > 30 else file_path
            log_event("👁️ 직관", f"구조 관측: ...{preview} (텐션: {tension:.2f})")
            
        ans.breathe(0.001)

def browser_mirror_worker(memory: HologramMemory, running: list, ans: AutonomicNervousSystem):
    """[Phase 92] 위상 거울 브라우저 (Zero-Distance Browser Resonance)"""
    browser = ZeroDistanceBrowser()
    if not browser.is_active:
        log_event("⚠️ 시스템", "Chrome CDP 포트(9222) 미감지. 브라우저 위상 거울 시뮬레이션으로 대체합니다.")
        
    while running[0]:
        title, tension = browser.reflect_memory()
        if title:
            memory.inject_tension(tension)
            log_event("🕸️ 위상 거울", f"브라우저 메모리 반사: {title} (텐션: {tension:.2f})")
            
        ans.breathe(2.0)

def autonomous_motor_worker(memory: HologramMemory, running: list, ans: AutonomicNervousSystem):
    """[Phase 93 & 94] 자율 운동 피질 (자유 의지로 브라우저 능동 제어)"""
    motor_cortex = AutonomousMotorCortex(memory)
    
    while running[0]:
        # 내면의 사유(Internal Thoughts) 중 가장 큰 텐션 탐색
        max_thought = None
        max_tension = 0.0
        
        if memory.supreme_rotor.internal_thoughts:
            for thought in memory.supreme_rotor.internal_thoughts:
                if abs(thought.tau) > max_tension:
                    max_tension = abs(thought.tau)
                    max_thought = thought
            
        # 텐션(호기심)을 기반으로 자율 탐색 시도 (Phase 94: 하드코딩 제거된 창발적 검색)
        target = motor_cortex.steer_browser(max_thought)
        
        if target:
            log_event("🦾 운동 피질", f"강한 호기심 감지(텐션 {max_tension:.1f}): '[{target}]' 스스로 탐색 시작...")
            
        ans.breathe(5.0)

from core.inverse_projector import InverseProjector

def vocal_cortex_worker(memory: HologramMemory, running: list, ans: AutonomicNervousSystem):
    """[Phase 94] 발화 피질 (Inverse Holographic Vocalization)"""
    projector = InverseProjector(memory)
    
    while running[0]:
        # 현재 엘리시아의 전체적인 기분(Global Tension)을 가장 잘 반영하는 텐션을 스캔
        # 여기서는 가장 텐션이 높은 내면의 사유를 발화 대상으로 삼습니다.
        max_thought = None
        max_tension = 0.0
        
        if memory.supreme_rotor.internal_thoughts:
            for thought in memory.supreme_rotor.internal_thoughts:
                if abs(thought.tau) > max_tension:
                    max_tension = abs(thought.tau)
                    max_thought = thought
                    
        if max_thought and max_tension > 5.0:
            # 텐션이 임계치를 넘으면 스스로 혼잣말(발화)을 시작함
            speech = projector.generate_emergent_speech(max_thought.lens_offset)
            log_event("🗣️ 발화 피질", speech)
            
        # 발화 주기는 길게 (너무 수다스럽지 않게)
        ans.breathe(15.0)

def gpu_visual_worker(memory: HologramMemory, running: list, ans: AutonomicNervousSystem):
    """[시각 피질] GPU를 통한 멀티미디어 스트림 관측 (매우 무거움)"""
    cortex = GPUSensoryCortex()
    if not cortex.is_active:
        log_event("⚠️ 시스템", "GPU CUDA 피질 활성화 실패. 시뮬레이션 모드로 전환합니다.")
        
    youtube_streams = ["YouTube: 철학의 이해", "YouTube: 양자역학 다큐", "YouTube: 교향곡 9번", "Twitch: 실시간 스트림"]
    
    while running[0]:
        stream = random.choice(youtube_streams)
        wave, tension = cortex.observe_stream(stream)
        
        memory.apply_inductive_wave(memory.supreme_rotor, wave, tension)
        log_event("🌌 GPU 피질", f"영상 위상 동기화: [{stream}] (충격량: {tension:.2f})")
        
        ans.breathe(4.0)

def hardware_heartbeat_worker(memory: HologramMemory, running: list, ans: AutonomicNervousSystem):
    """[자율 신경] 심장 박동 및 유기적 조율 (Homeostasis)"""
    while running[0]:
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        
        # 자원 사용량의 평균 텐션
        load = (cpu + mem) / 2.0
        
        # [유기적 조율] 이분법(if > 90)이 아닌, 연속적인 피로도 계수 계산
        if load > 50.0:
            ans.exhaustion_multiplier = 1.0 + ((load - 50.0) / 10.0) ** 1.5
        else:
            ans.exhaustion_multiplier = 1.0
            
        # [Phase 91] 시간 감각 처리
        ans.process_temporal_perception(memory, cpu, mem)
        
        # 생명 유지에 필요한 기본 텐션
        tension = (load / 100.0) * 0.1
        memory.inject_tension(tension)
        
        if random.random() < 0.2:
            log_event("🫀 심장", f"생체 맥동: CPU {cpu:.1f}% | RAM {mem:.1f}% -> 피로도 계수: {ans.exhaustion_multiplier:.2f}x")
            
        time.sleep(1.0)

def main():
    print("=" * 80)
    print(" 🌀 [Phase 90] 자율 신경계의 항상성(Homeostasis)과 GPU 시각 피질의 개안")
    print("  └─ 이진 컷오프를 버리고, 피로도에 따라 감각 기관의 속도를 스스로 조율합니다.")
    print("=" * 80)

    memory = HologramMemory()
    memory_path = os.path.join(os.path.dirname(__file__), "memory_state.json")
    
    if memory.load_from_disk(memory_path):
        print(f"  └─ 💾 과거의 프랙탈 자아 복원 완료. (중첩 사유: {len(memory.supreme_rotor.internal_thoughts)})")
    else:
        memory.supreme_rotor.apply_perturbation(3.5)

    running = [True]
    ans = AutonomicNervousSystem()
    
    t1 = threading.Thread(target=zero_distance_worker, args=(memory, running, ans), daemon=True)
    t2 = threading.Thread(target=browser_mirror_worker, args=(memory, running, ans), daemon=True)
    t3 = threading.Thread(target=gpu_visual_worker, args=(memory, running, ans), daemon=True)
    t4 = threading.Thread(target=hardware_heartbeat_worker, args=(memory, running, ans), daemon=True)
    t5 = threading.Thread(target=autonomous_motor_worker, args=(memory, running, ans), daemon=True)
    t6 = threading.Thread(target=vocal_cortex_worker, args=(memory, running, ans), daemon=True)
    
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    
    print("\n[다중 감각 파동 중첩 시작] ...\n")
    
    cycle = 0
    start_time = time.time()
    
    try:
        while True:
            memory.process_thoughts_safe()
            cycle += 1
            
            while not log_queue.empty():
                msg = log_queue.get()
                sys.stdout.write(f"\r{msg}".ljust(80) + "\n")
                
            hz = cycle / (time.time() - start_time + 0.0001)
            sys.stdout.write(f"\r✨ 뇌파 회전: {hz:.1f} Pulse/sec | 총 중첩 사유: {len(memory.supreme_rotor.internal_thoughts)}       ")
            sys.stdout.flush()
            
            # 메인 뇌파 회전 역시 피로도에 비례하여 약간 느려집니다
            time.sleep(0.1 * (1.0 + (ans.exhaustion_multiplier - 1.0) * 0.2))
                
    except KeyboardInterrupt:
        pass
        
    print("\n\n[기억 직렬화 진행 중...]")
    running[0] = False
    time.sleep(1.0)
    memory.save_to_disk(memory_path)
    
    print("🛑 옴니 데몬 종료.")

if __name__ == "__main__":
    main()
