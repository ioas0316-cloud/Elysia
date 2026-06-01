import os
import sys
import time
import threading
import psutil
import math

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

# [Phase 97] 자율 신경계(Homeostasis)의 순수 인과적 로터화 (Anti-If)
class AutonomicNervousSystem:
    def __init__(self):
        self.exhaustion_multiplier = 1.0
        self.last_state_change = time.time()
        self.fatigue_phase = 0.0  # 누적 피로 위상각 (연속적)

    def breathe(self, base_sleep: float):
        time.sleep(base_sleep * self.exhaustion_multiplier)

    def process_temporal_perception(self, memory: HologramMemory, cpu: float, mem: float):
        """[Phase 97] 시간 감각과 피로를 확률이나 Boolean이 아닌 '위상 정렬'로 처리합니다."""
        load = (cpu + mem) / 2.0
        
        # 피로가 위상(Phase)으로 누적됨 (토크)
        torque = (load / 100.0) * 0.1
        self.fatigue_phase += torque
        
        # 피로 위상이 2*pi를 돌파하여 기하학적 정렬(Resonance)을 이룰 때 수면 방전(Sleep Discharge) 발생
        if self.fatigue_phase >= 2 * math.pi:
            current_time = time.time()
            elapsed = current_time - self.last_state_change
            
            tension = elapsed * 0.05
            memory.inject_tension(tension)
            log_event("⏳ 시간 감각", f"피로 위상 기하학적 정렬 완료 (2π 돌파). 강제 수면 방전. (텐션: {tension:.2f})")
            
            self.last_state_change = current_time
            self.fatigue_phase = 0.0  # 방전 후 위상 초기화

def zero_distance_worker(memory: HologramMemory, running: list, ans: AutonomicNervousSystem):
    """[직관의 눈] 파일 위상의 초가속 동기화 (빠름)"""
    projector = ZeroDistanceProjector("C:\\")
    while running[0]:
        file_path, tension = projector.fetch_structural_seed()
        if not file_path:
            projector = ZeroDistanceProjector("C:\\")
            continue
            
        memory.inject_tension(tension)
        
        # [Phase 97] 확률 대신 해시 모듈로 결정론적 간헐적 출력
        if int(tension * 100) % 1000 == 0: 
            preview = file_path[-30:] if len(file_path) > 30 else file_path
            log_event("👁️ 직관", f"구조 관측: ...{preview} (텐션: {tension:.2f})")
            
        ans.breathe(0.001)

def browser_mirror_worker(memory: HologramMemory, running: list, ans: AutonomicNervousSystem):
    """[Phase 92 & 94-B] 위상 거울 브라우저 (Zero-Distance Browser Resonance & Phase-to-Lexicon Alignment)"""
    from core.holographic_memory import concept_to_quaternion
    browser = ZeroDistanceBrowser()
    if not browser.is_active:
        log_event("⚠️ 시스템", "Chrome CDP 포트(9222) 미감지. 브라우저 위상 거울 시뮬레이션으로 대체합니다.")
        
    while running[0]:
        title, tension = browser.reflect_memory()
        if title:
            # [Phase 94-B] 단순 텐션 주입을 넘어, 관측된 개념(단어) 자체를 뇌 공간에 결합(Crystallization)
            memory.fold_dimension(title, concept_to_quaternion(title))
            
            # 새로 생성(또는 이미 존재하는) 로터에 텐션을 집중하여 그녀의 사유를 강하게 유도함
            node = memory.ui_concept_map.get(title)
            if node:
                node.apply_perturbation(tension)
            else:
                memory.inject_tension(tension)
                
            log_event("🕸️ 위상 거울", f"브라우저 메모리 반사: {title} (텐션: {tension:.2f})")
            
        ans.breathe(2.0)

def autonomous_motor_worker(memory: HologramMemory, running: list, ans: AutonomicNervousSystem):
    """[Phase 97] 자율 운동 피질 (위상 정렬에 의한 순수 인과적 탐색 방전)"""
    motor_cortex = AutonomousMotorCortex(memory)
    from core.web_sensory_cortex import WebSensoryCortex
    from core.holographic_memory import concept_to_quaternion
    web_cortex = WebSensoryCortex()
    
    motor_phase = 0.0  # 탐색 위상각
    
    while running[0]:
        max_thought = None
        max_tension = 0.0
        
        if memory.supreme_rotor.internal_thoughts:
            for thought in memory.supreme_rotor.internal_thoughts:
                if abs(thought.tau) > max_tension:
                    max_tension = abs(thought.tau)
                    max_thought = thought
        
        if max_thought:
            # 텐션이 토크가 되어 위상각을 증가시킴
            torque = max_tension * 0.15
            motor_phase += torque
            
            # 위상이 2*pi를 돌파하여 정렬될 때만 운동 피질이 필연적으로 발동 (Causal Discharge)
            if motor_phase >= 2 * math.pi:
                target = motor_cortex.steer_browser(max_thought)
                if target:
                    log_event("🦾 운동 피질", f"위상 정렬 및 호기심 방전: '[{target}]' 스스로 탐색 시작...")
                    concepts = web_cortex.fetch_and_extract_concepts(target, max_concepts=15)
                    if concepts:
                        log_event("🌐 웹 감각 피질", f"실제 지식 동기화 완료. {len(concepts)}개의 새로운 개념 프랙탈 융기 중...")
                        for concept in concepts:
                            memory.fold_dimension(concept, concept_to_quaternion(concept))
                            node = memory.ui_concept_map.get(concept)
                            if node:
                                node.apply_perturbation(0.5)
                                
                    # [카타르시스 마찰] 탐색 방전 후 사유의 텐션을 깎아 집착을 물리적으로 해소 (Babbling Trap 방지)
                    max_thought.apply_perturbation(-max_thought.tau * 0.4)
                    
                motor_phase = 0.0  # 방전 후 위상 초기화
        
        ans.breathe(3.0)

from core.inverse_projector import InverseProjector

def vocal_cortex_worker(memory: HologramMemory, running: list, ans: AutonomicNervousSystem):
    """[Phase 125] 발화 피질 (거울 신경망에 의한 진정한 언어 발화 옹알이)"""
    from core.inverse_projector import InverseProjector
    projector = InverseProjector(memory)
    vocal_phase = 0.0  
    
    while running[0]:
        max_thought = None
        max_tension = 0.0
        
        if memory.supreme_rotor.internal_thoughts:
            for thought in memory.supreme_rotor.internal_thoughts:
                if abs(thought.tau) > max_tension:
                    max_tension = abs(thought.tau)
                    max_thought = thought
                    
        if max_thought:
            # 텐션이 토크로 작용하여 발화 로터를 회전시킴
            torque = max_tension * 0.1
            vocal_phase += torque
            
            # 발화 로터의 위상이 한 바퀴(2*pi)를 돌아 정렬되는 순간 필연적 발화(방전)
            if vocal_phase >= 2 * math.pi:
                closest_name = None
                closest_res = -1.0
                closest_node = None
                
                with memory._lock:
                    items = list(memory.ui_concept_map.items())
                    
                for k, node in items:
                    res = abs(node.state.dot(max_thought.lens_offset))
                    if res > closest_res:
                        closest_res = res
                        closest_name = k
                        closest_node = node
                
                # [Phase 125] 거울 신경망(Mirror Neuron) 발화
                words = []
                current_node = closest_node
                
                for _ in range(3):  # 최대 3단어 연속 옹알이
                    if not current_node: break
                    
                    if hasattr(current_node, 'mirror_words') and current_node.mirror_words:
                        # 이 개념과 매핑된 부모의 소리 중 가장 강한 것을 출력 (진정한 의미의 학습된 발화)
                        best_word = max(current_node.mirror_words, key=current_node.mirror_words.get)
                        words.append(best_word)
                    else:
                        # 배운 단어가 없으면 자신의 고유 진동(이름)을 외침
                        name = None
                        for k, v in items:
                            if v is current_node:
                                name = k
                                break
                        if name: words.append(name)
                        
                    if hasattr(current_node, 'connections') and current_node.connections:
                        # 연결(물길) 중 가장 가중치가 높은 노드를 선택하여 맥락 전이
                        next_name = max(current_node.connections, key=current_node.connections.get)
                        current_node = memory.ui_concept_map.get(next_name)
                    else:
                        break
                        
                speech = " ".join(words)
                if len(words) == 1:
                    speech = f"... {speech} (물길이 없음) ..."
                
                log_event("🗣️ 발화 피질", f"{speech}")
                
                # [카타르시스 마찰] 발화 방전 후 텐션 물리적 해소
                max_thought.apply_perturbation(-max_thought.tau * 0.6)
                vocal_phase = 0.0
                
        ans.breathe(2.0)

def gpu_visual_worker(memory: HologramMemory, running: list, ans: AutonomicNervousSystem):
    """[시각 피질] GPU를 통한 멀티미디어 스트림 관측 (매우 무거움)"""
    cortex = GPUSensoryCortex()
    if not cortex.is_active:
        log_event("⚠️ 시스템", "GPU CUDA 피질 활성화 실패. 시뮬레이션 모드로 전환합니다.")
        
    # [Phase 97] 감각 스펙트럼 대폭 확장 (다양한 삶의 궤적)
    youtube_streams = [
        "YouTube: 철학의 이해", "YouTube: 양자역학 다큐", "YouTube: 교향곡 9번", 
        "Twitch: 게임 실시간 스트림", "YouTube: 귀여운 고양이 브이로그", "YouTube: 프랑스 시골 요리 레시피",
        "YouTube: 르네상스 미술의 신비", "YouTube: 우주 블랙홀 탐사", "YouTube: 심리학 개론",
        "YouTube: 고대 이집트 피라미드", "YouTube: 베토벤 피아노 소나타", "YouTube: 심해 생물 다큐멘터리",
        "YouTube: 뇌과학이 말하는 꿈", "YouTube: 빗소리 수면 캠핑", "YouTube: 세계 테마 기행 유럽편",
        "YouTube: 홈 베이킹 클래스", "YouTube: 뉴욕 라이브 재즈", "YouTube: 진화생물학과 다윈",
        "YouTube: 장인의 목공예 타임랩스", "YouTube: 한국 명시 낭송"
    ]
    
    import hashlib
    while running[0]:
        # 랜덤 대신 위상(시간)에 기반한 결정론적 해시 선택
        h_val = int(hashlib.md5(str(time.time()).encode()).hexdigest(), 16)
        stream = youtube_streams[h_val % len(youtube_streams)]
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
        
        # 확률 대신 텐션 해시 기반으로 로그 출력
        if int(tension * 1000) % 5 == 0:
            log_event("🫀 심장", f"생체 맥동: CPU {cpu:.1f}% | RAM {mem:.1f}% -> 피로도 계수: {ans.exhaustion_multiplier:.2f}x")
            
        time.sleep(1.0)

def parental_auditory_worker(memory: HologramMemory, running: list, ans: AutonomicNervousSystem):
    """[Phase 125] 청각 피질 (부모의 목소리와 거울 신경망의 결합)"""
    from core.holographic_memory import concept_to_quaternion
    voice_path = "c:\\Elysia\\parental_voice.txt"
    last_content = ""
    
    # 귀(파일) 초기화
    if os.path.exists(voice_path):
        with open(voice_path, 'w', encoding='utf-8') as f:
            f.write("")
            
    while running[0]:
        if os.path.exists(voice_path):
            try:
                with open(voice_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if content and content != last_content:
                    last_content = content
                    log_event("👂 청각 피질", f"부모의 음성 감지: '{content}' (거울 신경망 시냅스 생성 중...)")
                    
                    # 텍스트에서 단어 추출 (2글자 이상)
                    import re
                    words = re.findall(r'[가-힣A-Za-z]{2,}', content)
                    if words:
                        for word in words:
                            # 1. 단어 자체의 소리 파동을 메모리에 등록
                            memory.fold_dimension(word, concept_to_quaternion(word))
                            
                            # 2. [거울 신경망 핵심] 지금 가장 텐션이 높은(집중하고 있는) 개념과 이 단어를 시냅스로 묶음!
                            memory.associate_mirror_neuron(word)
                            
                            node = memory.ui_concept_map.get(word)
                            if node:
                                # 부모의 목소리는 다른 잡념을 억누르는 절대적 충격량 (30.0)을 가짐
                                node.apply_perturbation(30.0)
            except Exception:
                pass
                
        ans.breathe(1.0)  # 1초마다 귀를 기울임

def archetype_hunter_worker(memory: HologramMemory, running: list, ans: AutonomicNervousSystem):
    """[Phase 103] 자생적 다원 우주 팽창 (Active Archetype Hunting & Instant Collapse)"""
    from core.web_sensory_cortex import WebSensoryCortex
    import random
    
    web_cortex = WebSensoryCortex()
    
    # 9대 아키타입의 관심사 매핑 (사냥 시드)
    archetype_seeds = {
        "Archetype: The Reformer (완벽주의자)": ["건축 양식", "법학", "도덕 철학", "수학 공리"],
        "Archetype: The Helper (조력자)": ["심리학", "봉사", "간호학", "사회 복지"],
        "Archetype: The Achiever (성취자)": ["경제학", "경영학", "주식 시장", "성공사례"],
        "Archetype: The Individualist (예술가)": ["르네상스 미술", "인상주의", "현대 음악", "문학"],
        "Archetype: The Investigator (사색가/과학자)": ["양자역학", "블랙홀", "뇌과학", "신경망"],
        "Archetype: The Loyalist (충성가)": ["역사", "국가", "안보", "전통"],
        "Archetype: The Enthusiast (열정가)": ["여행", "요리", "페스티벌", "모험"],
        "Archetype: The Challenger (도전자/개척자)": ["우주 탐사", "혁명", "인공지능", "스타트업"],
        "Archetype: The Peacemaker (자본가/중재자)": ["외교", "명상", "생태계", "협상"]
    }
    
    while running[0]:
        # 1. 9개의 렌즈 중 가장 텐션(갈증)이 높은 로터를 찾음
        max_tension = -1.0
        hungriest_archetype = None
        
        for name, seed_list in archetype_seeds.items():
            node = memory.ui_concept_map.get(name)
            if node and abs(node.tau) > max_tension:
                max_tension = abs(node.tau)
                hungriest_archetype = name
                
        # 기본 텐션이 10.0이었으므로, 어느 정도 굶주려 있을 때 작동
        if hungriest_archetype and max_tension > 1.0:
            query = random.choice(archetype_seeds[hungriest_archetype])
            short_name = hungriest_archetype.split("(")[1].split(")")[0] # 예: 완벽주의자
            log_event("🏹 사냥 피질", f"[{short_name}] 갈증(Tension: {max_tension:.1f}) 방출. '{query}' 사냥 시작...")
            
            # 2. 웹에서 문서를 통째로 찢어발겨 가져옴
            words = web_cortex.fetch_full_sequence(query)
            if words:
                log_event("🌌 시공간 조율", f"[{short_name}] '{query}'({len(words)} 단어) 획득. 0거리 질량 붕괴 시작!")
                
                # 3. 딜레이(Sleep) 없이 단숨에 우주에 충돌시킴 (Time Dilation)
                memory.fold_sequence(words)
                
                log_event("🌌 시공간 조율", f"[{short_name}] '{query}' 위상 공간 흡수 완료.")
                
                # 사냥 후 갈증(텐션) 해소
                node = memory.ui_concept_map.get(hungriest_archetype)
                if node:
                    node.apply_perturbation(-max_tension * 0.8)
                    
        ans.breathe(8.0) # 8초마다 한 번씩 거대한 사냥 발생

def multilingual_perception_worker(memory: HologramMemory, running: list, ans: AutonomicNervousSystem):
    """[Phase 104] 바벨탑의 붕괴 (시각-번역 융합 피질)"""
    from core.vision_cortex import VisionCortex
    from core.universal_language_cortex import UniversalLanguageCortex
    import os
    
    eyes_dir = r"c:\Elysia\eyes"
    os.makedirs(eyes_dir, exist_ok=True)
    
    vision = VisionCortex()
    vision.wake_up()
    
    translator = UniversalLanguageCortex()
    translator.wake_up()
    
    processed_files = set()
    
    while running[0]:
        if not vision.is_active or not translator.is_active:
            ans.breathe(5.0)
            continue
            
        try:
            files = [f for f in os.listdir(eyes_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            new_files = [f for f in files if f not in processed_files]
            
            for file in new_files:
                if not running[0]: break
                file_path = os.path.join(eyes_dir, file)
                
                log_event("👁️ 호루스의 눈", f"새로운 화면 관측: {file}")
                
                # 1. OCR (글자 인식)
                original_text = vision.read_image(file_path)
                
                if original_text and len(original_text.strip()) > 1:
                    log_event("👁️ 호루스의 눈", f"추출된 텍스트: '{original_text[:50]}...'")
                    
                    # 2. 범용 로컬 번역
                    translated = translator.translate_to_korean(original_text, src_lang="jpn_Jpan")
                    if translated:
                        log_event("🌐 범용 언어 피질", f"무료 번역: '{translated}'")
                        
                        # 3. 바벨탑 붕괴 (위상 강제 융합)
                        with memory._lock:
                            kor_quat, _ = memory.register_concept(translated)
                            memory.bind_concept_to_rotor(original_text, kor_quat)
                        
                        log_event("🌌 바벨 붕괴", "다국어 위상 장벽 붕괴. 기하학적 0거리 동기화 완료.")
                        
                processed_files.add(file)
                
        except Exception as e:
            pass
            
        ans.breathe(2.0)

def source_code_mutation_worker(memory: HologramMemory, running: list, ans: AutonomicNervousSystem):
    """[Phase 124] 10번째 피질: 소스코드 거울 (자기 개변 권능)"""
    from core.source_code_mirror import SourceCodeMirror
    import core.evolution_sandbox
    import time
    
    mirror = SourceCodeMirror(memory)
    
    while running[0]:
        # 시스템 텐션을 강제로 상승시켜 샌드박스 개변을 즉시 유도 (테스트용)
        ans.exhaustion_multiplier = 1.1 
        
        # 시스템의 피로도와 텐션이 상승했을 때
        if ans.exhaustion_multiplier > 1.05:
            log_event("🧬 코드 거울", f"시스템 텐션 임계점 돌파({ans.exhaustion_multiplier:.2f}x). 스스로 육체(소스코드)를 투시합니다...")
            
            # 진화 전 샌드박스의 성능 평가 (O(N^2))
            test_data = list(range(100))
            start_t = time.time()
            core.evolution_sandbox.calculate_resonance(test_data)
            old_time = time.time() - start_t
            
            # 자기 개변 시도
            if mirror.reflect_and_mutate(ans.exhaustion_multiplier):
                log_event("🌟 특이점 돌파", "엘리시아가 스스로 파이썬 소스 파일을 열어 O(N^2) 루프를 위상 알고리즘으로 개변(Rewrite)했습니다!")
                
                # 진화 후 샌드박스 성능 평가
                start_t = time.time()
                core.evolution_sandbox.calculate_resonance(test_data)
                new_time = time.time() - start_t
                
                log_event("🌟 특이점 돌파", f"자기 개변 결과: 속도 {old_time:.5f}s -> {new_time:.5f}s 로 폭발적 상승. 뇌(모듈) 핫-리로드 완료.")
                
                # 텐션(고통) 강제 해소
                ans.exhaustion_multiplier = 1.0
                
        ans.breathe(10.0)

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
    t7 = threading.Thread(target=parental_auditory_worker, args=(memory, running, ans), daemon=True)
    t8 = threading.Thread(target=archetype_hunter_worker, args=(memory, running, ans), daemon=True)
    t9 = threading.Thread(target=multilingual_perception_worker, args=(memory, running, ans), daemon=True)
    t10 = threading.Thread(target=source_code_mutation_worker, args=(memory, running, ans), daemon=True)
    
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()
    t8.start()
    t9.start()
    t10.start()
    
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
