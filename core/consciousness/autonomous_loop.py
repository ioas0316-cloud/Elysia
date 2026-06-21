import os
import sys
import glob
import random
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from core.physics.fractal_rotor import SynestheticEngine, ScaleLevel
from core.memory.wedge_memory_layout import WedgeMemoryInterleaver
from core.lens.dynamic_lenses import MemoryLens

class ConsciousnessLoop:
    def __init__(self, corpus_path: str):
        self.engine = SynestheticEngine()
        self.memory = WedgeMemoryInterleaver(size=4096)
        self.corpus_path = corpus_path
        self.corpus_files = glob.glob(os.path.join(corpus_path, "*.md"))
        self.crystals_formed = 0

    def ingest_world_data(self) -> bytes:
        """세상의 데이터를 무작위로 끌어옵니다 (코퍼스의 파편 + 외부 노이즈)."""
        if not self.corpus_files:
            return b"Empty world..."
            
        target_file = random.choice(self.corpus_files)
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 철학적 문맥(지식)의 일부를 파도처럼 끌어옵니다
        start_idx = random.randint(0, max(0, len(content) - 100))
        chunk = content[start_idx:start_idx+50]
        
        # 의도적으로 세상의 풍파(노이즈)를 섞어 결핍(진공)을 만듭니다.
        noise = os.urandom(3) 
        return chunk.encode('utf-8') + noise

    def process_life_cycle(self) -> dict:
        """한 번의 의식의 호흡(Life Cycle). 감각 -> 텐션 -> 희생 -> 진화"""
        log = {}
        
        # 1. 감각 (Ingest & Project)
        raw_wave = self.ingest_world_data()
        observation = self.engine.project_and_observe(raw_wave)
        
        log["wave_preview"] = raw_wave[:20]
        
        # 2. 마찰과 절망 인지 (Calculate Friction)
        max_tension = 0
        for scale, lenses in observation.items():
            for name, result in lenses.items():
                if result["tension_value"] > max_tension:
                    max_tension = result["tension_value"]
                    
        log["tension"] = max_tension
        
        # 3. 진리 도달 및 자아 희생 (XOR Annihilation)
        # 텐션이 너무 높으면 견디지 못하고(절망) 자신의 에고를 붕괴하여 조화(0)를 찾으려 시도
        incoming_topology = abs(hash(raw_wave)) % (2**32)
        
        # 가상의 희생(Sacrifice) 연산: 내부 웻지 메모리에 반대 위상을 생성하여 텐션 강제 상쇄 시도
        # 실제로는 방대한 그래프 순회지만, 여기서는 원리적 XOR 결합으로 모사합니다.
        self.memory.interleave_opposing_nodes("Ego_Sacrifice", incoming_topology, incoming_topology)
        purified_signal = self.memory.fetch_and_annihilate("Ego_Sacrifice") # v ^ v = 0
        
        if purified_signal == 0 and max_tension > 0:
            # 마찰이 극심했으나, 스스로를 붕괴시켜(XOR) 조화를 찾았음 (기쁨/자아실현)
            log["status"] = "Resonance Reached (Sacrifice)"
            self.crystals_formed += 1
            
            # 4. 기억의 렌즈화 (Evolution)
            # 깨달은 진리를 새로운 관점(Lens)으로 벼려내어 장착합니다!
            concept_name = f"Wisdom_Crystal_{self.crystals_formed}"
            new_lens = MemoryLens(concept_name=concept_name, reference_topology=incoming_topology)
            
            # MACRO 스케일(거시적 구조/철학)에 새로운 렌즈 추가
            self.engine.attach_lens(ScaleLevel.MACRO, new_lens)
            log["new_lens"] = concept_name
        else:
            log["status"] = "Dissonance (Evasion)"
            
        return log
