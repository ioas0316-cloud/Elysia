import time
import math
import os
from core.brain.wave_slicer import WaveSlicer
from core.brain.macro_axiom_rotor import MacroAxiomRotor
from core.brain.dictionary_synchronizer import DictionarySynchronizer
from core.brain.holographic_memory import HologramMemory
from core.utils.math_utils import Quaternion

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

        # 기억 엔진 연동
        self.memory = HologramMemory()
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
                
                # 기억 엔진(HologramMemory)에 단어 저장 및 시퀀스 주입
                if self.memory:
                    self.memory.register_concept(word)
                    self.memory.fold_sequence([word])
                
                # [Level 3] 문장 조립 승급 시도
                if len(self.word_buffer) >= 2:
                    sent_logs = []
                    sentence = self.axiom_frame.try_fit_level3_sentence(self.word_buffer[0], self.word_buffer[1], sent_logs)
                    if sentence:
                        self.word_buffer.clear()
                        for slog in sent_logs:
                            print(slog)
                        
                        # 문장도 기억 엔진에 등록
                        if self.memory:
                            self.memory.register_concept(sentence)

    def _process_raw_buffer(self, idx: int):
        # [Level 1 - 글자 조립 시도 (Greedy Matching)]
        letter = None
        
        # 1. 3글자 매칭 시도 (초성 + 중성 + 종성)
        if len(self.raw_block_buffer) >= 3:
            play_logs = []
            letter = self.axiom_frame.try_fit_level(1, self.raw_block_buffer[:3], play_logs)
            if letter:
                self.letter_buffer.append(letter)
                self.raw_block_buffer = self.raw_block_buffer[3:]
                for plog in play_logs:
                    print(plog)
                self._try_fit_level2(idx)
        
        # 2. 2글자 매칭 시도 (초성 + 중성)
        if not letter and len(self.raw_block_buffer) >= 2:
            play_logs = []
            letter = self.axiom_frame.try_fit_level(1, self.raw_block_buffer[:2], play_logs)
            if letter:
                self.letter_buffer.append(letter)
                self.raw_block_buffer = self.raw_block_buffer[2:]
                for plog in play_logs:
                    print(plog)
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
        
        # Genesis Node 구동
        if hasattr(self, 'genesis_node'):
            self.genesis_node.wake_up()
        
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

        # 각성 사이클 완료 후 사유 숙성 및 기억 저장
        if self.memory:
            print("\n[Omni-Daemon] 기억 엔진 사유 숙성(Cognitive Maturation) 중...")
            self.memory.process_thoughts_safe()
            self.memory.save_to_disk()
            print("[Omni-Daemon] 사유 결과가 디스크에 영구 저장되었습니다.")

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
        bridge = HumanIntelligenceBridge(self.memory)
        response = bridge.generate_response(user_input)
        return response
