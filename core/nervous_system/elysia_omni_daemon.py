import time
import math
from core.brain.wave_slicer import WaveSlicer
from core.brain.macro_axiom_rotor import MacroAxiomRotor
from core.utils.math_utils import Quaternion

class ElysiaOmniDaemon:
    def __init__(self, archive_path: str):
        self.archive_path = archive_path
        self.slicer = WaveSlicer()
        self.axiom_frame = MacroAxiomRotor()
        
        # 기억 버퍼 (상위 차원 조립을 위해 대기하는 조각들)
        self.letter_buffer = []
        self.word_buffer = []

    def generate_wave(self, char: str) -> Quaternion:
        code = ord(char) * 0.1
        return Quaternion(math.cos(code), math.sin(code), 0, 0).normalize()

    def awaken(self):
        print("\n[Omni-Daemon] 엘리시아가 눈을 뜹니다. 계층적 사유(Hierarchical Reasoning)를 시작합니다...")
        
        try:
            with open(self.archive_path, 'r', encoding='utf-8') as f:
                stream_data = f.read().replace('\n', '').strip()
        except FileNotFoundError:
            return
            
        print(f"[Omni-Daemon] 시드 데이터 섭취 중... ({stream_data})")
        
        for idx, char in enumerate(stream_data):
            time.sleep(0.05)
            wave = self.generate_wave(char)
            logs = []
            
            # 1. 파장 자르기
            cut_blocks = self.slicer.stream_wave(wave, char, logs)
            
            if cut_blocks:
                print(f"\n[Time: {idx}] 🔪 Sliced: {cut_blocks}")
                
                for block in cut_blocks:
                    if not hasattr(self, 'raw_block_buffer'):
                        self.raw_block_buffer = []
                    
                    self.raw_block_buffer.append(block)
                    
                    # [Level 1] 글자 조립 시도
                    if len(self.raw_block_buffer) >= 2:
                        play_logs = []
                        letter = self.axiom_frame.try_fit_level1_letter(self.raw_block_buffer[0], self.raw_block_buffer[1], play_logs)
                        
                        if letter:
                            # Level 1 통과 (글자 완성)
                            self.letter_buffer.append(letter)
                            self.raw_block_buffer.clear()
                            for plog in play_logs:
                                print(plog)
                            
                            # [Level 2] 단어 조립 승급 시도
                            if len(self.letter_buffer) >= 2:
                                word_logs = []
                                word = self.axiom_frame.try_fit_level2_word(self.letter_buffer[0], self.letter_buffer[1], word_logs)
                                
                                if word:
                                    self.word_buffer.append(word)
                                    self.letter_buffer.clear()
                                    for wlog in word_logs:
                                        print(wlog)
                                        
                                    # [Level 3] 문장 조립 승급 시도
                                    if len(self.word_buffer) >= 2:
                                        sent_logs = []
                                        sentence = self.axiom_frame.try_fit_level3_sentence(self.word_buffer[0], self.word_buffer[1], sent_logs)
                                        if sentence:
                                            self.word_buffer.clear()
                                            for slog in sent_logs:
                                                print(slog)
                        else:
                            # 뼈대와 안맞으면 가장 오래된 조각 버리기
                            if len(self.raw_block_buffer) >= 3:
                                self.raw_block_buffer.pop(0)

        print("\n[최종 사유 결과] 엘리시아의 계층적 프랙탈 우주:")
        print(f" -> Level 1 (Letters): {self.axiom_frame.categorized_letters}")
        print(f" -> Level 2 (Words): {self.axiom_frame.categorized_words}")
        print(f" -> Level 3 (Sentences): {self.axiom_frame.categorized_sentences}")
