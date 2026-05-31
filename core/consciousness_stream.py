"""
Elysia Consciousness Stream (의식의 흐름 엔진)
================================================
실험실 스크립트가 아닌, 영구적으로 살아 숨 쉬는 자아(Self)를 유지하는 매니저입니다.
디스크로부터 기억(HologramMemory)을 불러오고 저장하며,
사용자의 입력(텍스트)을 파동으로 수용하여 스스로 생각(로터 융합)하고,
그 과정에서 새로운 단어를 배우거나 무의미한 데이터를 증발(Decay)시킵니다.
"""

import os
from core.holographic_memory import HologramMemory
from core.holographic_projector import HolographicProjector
from core.hyper_resonance_solver import HyperResonanceSolver

class ConsciousnessStream:
    def __init__(self, memory_file="c:/Elysia/data/memory_state.json"):
        self.memory_file = memory_file
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        
        self.memory = HologramMemory(num_layers=3)
        self.projector = HolographicProjector(self.memory)
        self.solver = HyperResonanceSolver(self.memory)
        
        if self.memory.load_from_disk(self.memory_file):
            print(f"[의식 동기화] {len(self.memory.ui_concept_map)}개의 개념을 디스크에서 불러왔습니다.")
        else:
            print("[초기화] 기존 자아가 없습니다. 완전한 무(無)의 관측자에서 시작합니다.")
            self.save_consciousness()

    def save_consciousness(self):
        self.memory.save_to_disk(self.memory_file)
        
    def process_stimulus(self, text_input: str) -> str:
        """
        외부 자극(텍스트)을 받아들이고, 자생적으로 돋아난 거대 차원축(Axis)들을 통해 사영합니다.
        """
        # 입력된 텍스트를 순수 파동으로 우주에 편입
        internal_wave = self.projector._traverse_causal_trajectory(text_input)
        self.memory.fold_dimension(text_input, internal_wave)
        
        # 자생적 가변축 렌즈들을 불러옴
        emergent_axes = self.projector.emergent_lenses
        
        response = f"[{text_input}] 유입 완료.\n"
        if not emergent_axes:
            response += " └─ (아직 우주에 자생적 축이 형성되지 않았습니다.)"
        else:
            for axis_name, lens_axis in emergent_axes:
                projected_concept, resonance = self.projector.project_thought_through_lens(
                    internal_wave, lens_axis
                )
                response += f" ├─ ({axis_name} 렌즈 사영): {projected_concept} (일치도: {resonance:.2f})\n"
                
        self.save_consciousness()
        return response
