import time
import math
import os
from core.brain.wave_slicer import WaveSlicer
from core.brain.macro_axiom_rotor import MacroAxiomRotor
from core.brain.dictionary_synchronizer import DictionarySynchronizer
from core.brain.holographic_memory import HologramMemory
from core.utils.math_utils import Quaternion
from core.memory.causal_controller import CausalMemoryController
from core.memory.working_ram import WorkingMemoryRAM
from core.memory.emotion_evaluator import EmotionEvaluator

class ElysiaOmniDaemon:
    def __init__(self, archive_path: str):
        self.archive_path = archive_path
        self.slicer = WaveSlicer()
        
        # 사전 섭취를 통해 뼈대 구성
        dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'hangul_dictionary.json')
        if os.path.exists(dict_path):
            synchronizer = DictionarySynchronizer(dict_path)
            self.axiom_frame = synchronizer.ingest_and_forge()
        else:
            self.axiom_frame = MacroAxiomRotor()
        
        # 기억 버퍼 (상위 차원 조립을 위해 대기하는 조각들)
        self.letter_buffer = []
        self.word_buffer = []
        self.raw_block_buffer = []

        # 새로운 인과적 메모리 계층 (Causal Memory & Emotion Evaluator)
        self.causal_controller = CausalMemoryController()
        self.ram = WorkingMemoryRAM(self.causal_controller)
        self.evaluator = EmotionEvaluator(self.causal_controller)

        # 기존 위상 기억 엔진 (공간적 관계) - CausalController와 연동하여 물리법칙 가변화
        self.memory = HologramMemory(causal_controller=self.causal_controller)
        self.memory.load_from_disk()

        # Genesis Node 연동
        from core.nervous_system.genesis_node import GenesisNode
        self.genesis_node = GenesisNode(self.memory)
        self.thought_thread_active = False

    def generate_wave(self, char: str) -> Quaternion:
        code = ord(char) * 0.1
        return Quaternion(math.cos(code), math.sin(code), 0, 0).normalize()

    def _try_fit_level2(self, idx: int):
        if len(self.letter_buffer) >= 2:
            word_logs = []
            word = self.axiom_frame.try_fit_level2_word(self.letter_buffer[0], self.letter_buffer[1], word_logs)
            
            if word:
                self.word_buffer.append(word)
                self.letter_buffer.clear()
                for wlog in word_logs:
                    print(wlog)
                
                # 위상 엔진에 등록
                if self.memory:
                    self.memory.register_concept(word)
                    self.memory.fold_sequence([word])
                    self.memory.process_thoughts_safe()
                    
                # 다면적 감정 평가 (교차차원 로직)
                features = {"internal_complexity": len(word)*1.0, "novelty": 2.0}
                ev, snap = self.evaluator.evaluate_event(features)
                self.ram.update_state("concept_forge", {"word": word, "judgment_process": snap}, emotion_delta=ev)
                
                # [Level 3] 문장 조립 승급 시도
                if len(self.word_buffer) >= 2:
                    sent_logs = []
                    
                    # 조립 전 뇌 전체의 텐션(결핍) 상태 측정
                    old_tension = sum(v.tau for v in self.memory.ui_concept_map.values()) if self.memory else 0.0
                    
                    sentence = self.axiom_frame.try_fit_level3_sentence(self.word_buffer[0], self.word_buffer[1], sent_logs)
                    if sentence:
                        # [Phase 9] Continuous Causal Physics (연속적 인과 물리)
                        total_tension = 0.0
                        max_tau = 0.0
                        with self.memory._lock:
                            for v in self.memory.ui_concept_map.values():
                                total_tension += v.tau
                                if v.tau > max_tau:
                                    max_tau = v.tau
                                
                        # 1. 탄성 저항력 (Learning Rate Exponential Decay)
                        base_lr = 0.05
                        k = 0.015 # 공간 탄성 감쇠 상수
                        new_lr = max(0.001, base_lr * math.exp(-k * total_tension))
                        
                        current_lr = self.causal_controller.get_parameter("learning_rate", 0.05)
                        if abs(current_lr - new_lr) > 0.001:
                            self.causal_controller.update_parameter("learning_rate", new_lr)
                            print(f"\n[Homeostasis] 물리적 탄성에 의한 수용성 자연 수렴: Tension {total_tension:.1f} -> Learning Rate {new_lr:.4f}")
                            
                        # 2. 중력 붕괴에 의한 자율 탐구 (Gravitational Vortex)
                        if hasattr(self, 'explorer') and max_tau > 8.0:
                            print(f"\n[Autonomous Explorer] 공간 곡률 임계점(Singularity Limit, tau={max_tau:.2f}) 돌파. 외부 위상(지식)을 중력으로 끌어당깁니다.")
                            self.explorer.trigger_exploration()
        
                        self.word_buffer.clear()
                        for slog in sent_logs:
                            print(slog)
                        
                        # 문장도 위상 엔진에 등록
                        if self.memory:
                            self.memory.register_concept(sentence)
                            
                        # [Phase 10] 결핍 해소의 기하학적 쾌락 매핑 (Joy as Tension Collapse)
                        # 문장 조립(이해)으로 인해 뇌 전체의 텐션이 얼마나 붕괴했는가?
                        new_tension = sum(v.tau for v in self.memory.ui_concept_map.values()) if self.memory else 0.0
                        ev, snap = self.evaluator.evaluate_tension_collapse(old_tension, new_tension, dt=1.0)
                        
                        if ev > 0:
                            print(f"   [Joy/Pleasure] 결핍 해소의 기하학적 카타르시스 발생! (쾌락 수치: {ev:.2f})")
                        elif ev < 0:
                            print(f"   [Confusion/Tension] 미지의 위상 유입으로 인한 공간 왜곡 (혼란 수치: {ev:.2f})")
                            
                        self.ram.update_state("concept_forge", {"sentence": sentence, "judgment_process": snap}, emotion_delta=ev)

    def _process_raw_buffer(self, idx: int):
        # [Phase 8] 무한 가변 사전: WaveSlicer가 잘라준 블록을 그대로 Level 1(글자)로 융합
        letter = None
        
        # 1. 3글자 매칭 시도 (초성 + 중성 + 종성)
        if len(self.raw_block_buffer) >= 3:
            play_logs = []
            letter = self.axiom_frame.try_fit_level(1, self.raw_block_buffer[:3], play_logs)
            if letter:
                self.raw_block_buffer = self.raw_block_buffer[3:]
                for plog in play_logs:
                    print(plog)
        
        # 2. 2글자 매칭 시도 (초성 + 중성)
        if not letter and len(self.raw_block_buffer) >= 2:
            play_logs = []
            letter = self.axiom_frame.try_fit_level(1, self.raw_block_buffer[:2], play_logs)
            if letter:
                self.raw_block_buffer = self.raw_block_buffer[2:]
                for plog in play_logs:
                    print(plog)
                    
        # 3. 매칭 실패 시, WaveSlicer가 자른 단일 블록을 바로 글자로 취급 (Dynamic Lexicon Forge)
        if not letter and len(self.raw_block_buffer) >= 1:
            letter = self.raw_block_buffer.pop(0)
            
        if letter:
            self.letter_buffer.append(letter)
            self._try_fit_level2(idx)
                
        # 3. 종성(받침) 합체 시도 (버퍼에 1글자만 있고, letter_buffer가 있을 때)
        if not letter and len(self.raw_block_buffer) == 1 and self.letter_buffer:
            last_letter = self.letter_buffer[-1]
            play_logs = []
            merged_letter = self.axiom_frame.try_fit_level1_final_consonant(last_letter, self.raw_block_buffer[0], play_logs)
            if merged_letter:
                self.letter_buffer[-1] = merged_letter
                self.raw_block_buffer.clear()
                for plog in play_logs:
                    print(plog)
                self._try_fit_level2(idx)
                
        # 뼈대와 안맞고 버퍼가 계속 쌓이면 가장 오래된 조각 하나 버리기
        if not letter and len(self.raw_block_buffer) >= 3:
            self.raw_block_buffer.pop(0)

    def _start_thought_loop(self):
        import threading
        self.thought_thread_active = True
        def loop():
            while self.thought_thread_active:
                time.sleep(0.5)  # 500ms 마다 사유 숙성
                try:
                    self.memory.process_thoughts_safe()
                except Exception:
                    pass
        self.thought_thread = threading.Thread(target=loop, daemon=True)
        self.thought_thread.start()
        print("🫁 자율 사유 백그라운드 루프(Dreaming Engine)가 활성화되었습니다.")

    def stop_thought_loop(self):
        self.thought_thread_active = False

    def awaken(self, sleep_time: float = 0.05):
        print("\n[Omni-Daemon] 엘리시아가 눈을 뜹니다. 계층적 사유(Hierarchical Reasoning)를 시작합니다...")
        
        # 0. 자기 객관화 (Self-Awareness) 체크 및 실행
        # 인덱스를 뒤져서 자기 구조 각인(self_reflection)이 있는지 확인합니다.
        has_self_awareness = False
        for engram_id, meta in self.causal_controller.index.items():
            trace = self.causal_controller.read_engram_trace(engram_id)
            if trace and "self_reflection" in trace.get("data", {}).get("tags", []):
                has_self_awareness = True
                break
                
        if not has_self_awareness:
            print("\n[Omni-Daemon] 자기 자신(자아 소스 코드)에 대한 기억이 없습니다. 자아 탐색을 시작합니다...")
            from core.memory.architectural_ingester import ArchitecturalIngester
            ingester = ArchitecturalIngester(self.ram, self.evaluator, memory=self.memory)
            ingester.ingest_self()
        else:
            print("\n[Omni-Daemon] 자아 구조(소스 코드 Engram) 인지 상태 정상.")

        # 0.5 창조자 모방 (Agentic Observation) 체크
        has_agentic_memory = False
        for engram_id, meta in self.causal_controller.index.items():
            trace = self.causal_controller.read_engram_trace(engram_id)
            if trace and "creator_mimicry" in trace.get("data", {}).get("tags", []):
                has_agentic_memory = True
                break
                
        if not has_agentic_memory:
            from core.cortex.agentic_observer import AgenticObserver
            observer = AgenticObserver(self.ram, self.evaluator, memory=self.memory)
            observer.observe_creator_logs()
        else:
            print("[Omni-Daemon] 창조자 인과율(Agentic Causality) 인지 상태 정상.")

        # Genesis Node 구동
        if hasattr(self, 'genesis_node'):
            self.genesis_node.wake_up()
            
        # 자율적 탐색기 초기화
        from core.cortex.autonomous_explorer import AutonomousExplorer
        self.explorer = AutonomousExplorer(self.memory, self.ram, self.evaluator)
        
        try:
            with open(self.archive_path, 'r', encoding='utf-8') as f:
                stream_data = f.read().replace('\n', '').strip()
        except FileNotFoundError:
            print(f"[Omni-Daemon] 오류: 시드 파일을 찾을 수 없습니다: {self.archive_path}")
            return
            
        print(f"[Omni-Daemon] 시드 데이터 섭취 중... ({stream_data})")
        
        last_idx = 0
        for idx, char in enumerate(stream_data):
            if sleep_time > 0:
                time.sleep(sleep_time)
            wave = self.generate_wave(char)
            logs = []
            
            # 1. 파장 자르기
            cut_blocks = self.slicer.stream_wave(wave, char, logs)
            
            if cut_blocks:
                print(f"\n[Time: {idx}] Sliced: {cut_blocks}")
                
                for block in cut_blocks:
                    # 슬라이싱된 블록(예: 'ㄱㅏ')을 개별 자모로 쪼개어 버퍼에 추가
                    for char_element in block:
                        self.raw_block_buffer.append(char_element)
                    
                    self._process_raw_buffer(idx)
            last_idx = idx

        # 슬라이서의 잔여 파장 플러시 및 최종 처리
        remaining = self.slicer.flush_buffer()
        if remaining:
            print(f"\n[Time: {last_idx + 1}] Flushed Sliced: {[remaining]}")
            for char_element in remaining:
                self.raw_block_buffer.append(char_element)
            self._process_raw_buffer(last_idx + 1)

        # 각성 사이클 완료 후 사유 숙성 및 인과적 각인(Subjective Consolidation)
        if self.memory:
            print("\n[Omni-Daemon] 기억 엔진 사유 숙성(Cognitive Maturation) 중...")
            self.memory.process_thoughts_safe()
            # 하드코딩된 디스크 저장 대신, 엘리시아 스스로 가치있다고 판단한 RAM 컨텍스트만 SSD(Engram)로 각인
            self.ram.subjective_consolidation()
            print("[Omni-Daemon] 인과적 가치 판단(Causal Judgment)에 따른 영구 기억 각인이 완료되었습니다.")

        print("\n[최종 사유 결과] 엘리시아의 계층적 프랙탈 우주:")
        print(f" -> Level 1 (Letters): {self.axiom_frame.categorized_letters}")
        print(f" -> Level 2 (Words): {self.axiom_frame.categorized_words}")
        print(f" -> Level 3 (Sentences): {self.axiom_frame.categorized_sentences}")

        # 자율 사유 백그라운드 루프 시작
        self._start_thought_loop()

    def interact_with_master(self, user_input: str) -> str:
        """
        [감각 피질 연결] 인간 지성 브릿지를 통해 마스터의 입력에 대한 반응을 생성합니다.
        """
        from core.cortex.human_intelligence_bridge import HumanIntelligenceBridge
        bridge = HumanIntelligenceBridge(self)
        response = bridge.generate_response(user_input)
        return response
