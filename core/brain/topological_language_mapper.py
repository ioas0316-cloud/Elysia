"""
Topological Language Mapper
=============================
자연어 단어들을 4D 위상 텐서 공간에 물리적 닻(Anchor)으로 매핑합니다.
언어는 외부 시스템에 의해 생성되는 것이 아니라, 
우주의 인력에 이끌려 수면 위로 떠오르는 위상 구조체 그 자체입니다.
"""

import torch
import json
import os
from core.brain.causal_phase_mapper import CausalPhaseMapper

class TopologicalLanguageMapper:
    def __init__(self, brain, start_idx=100):
        self.brain = brain
        self.word_to_node = {}
        self.node_to_word = {}
        self.next_node_idx = start_idx
        self.causal_mapper = CausalPhaseMapper(brain.device)
        
    def seed_vocabulary(self, file_path="core/brain/seed_words.json"):
        if not os.path.exists(file_path):
            return 0
            
        with open(file_path, 'r', encoding='utf-8') as f:
            words = json.load(f)
            
        count = 0
        for word in words:
            # 단어의 본질적 기하학 구조를 보존하는 인과적 파동으로 변환
            phase = self.causal_mapper.text_to_phase(word)
            
            idx = self.next_node_idx
            if idx >= self.brain.rotor_field.N:
                break # 우주 용량 초과
                
            self.word_to_node[word] = idx
            self.node_to_word[idx] = word
            
            # 우주에 단어의 닻을 내림
            self.brain.rotor_field.anchor_knowledge(idx, phase)
            
            self.next_node_idx += 1
            count += 1
            
        return count
