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
            # 단어의 본질적 기하학 구조를 보존하는 인과적 파동 궤적으로 변환 (순서의 공간화)
            trajectory = self.causal_mapper.text_to_phase(word)
            
            # 궤적의 각 포인트를 시공간에 순차적으로 닻을 내림
            word_node_indices = []
            for phase_point in trajectory:
                idx = self.next_node_idx
                if idx >= self.brain.rotor_field.N:
                    break # 우주 용량 초과

                # 우주에 단어의 흐름(궤적 포인트)의 닻을 내림
                self.brain.rotor_field.anchor_knowledge(idx, phase_point)
                word_node_indices.append(idx)
                
                self.next_node_idx += 1
            
            if not word_node_indices:
                break

            # 단어 매핑을 첫 번째 인덱스나 궤적 리스트로 유지 (여기서는 대표 노드/시작점)
            self.word_to_node[word] = word_node_indices[0]
            self.node_to_word[word_node_indices[0]] = word
            
            count += 1
            
        return count
