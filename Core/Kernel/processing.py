"""
Kernel Processing Module

처리 관련 함수들
"""

import sys
import os
import logging
import time
import json

def process_thought(self, input_concept: str) -> str:
    """
        Process a thought through the new "Resonance Wave Pattern" stack.
        This completely replaces the old oscillator-based voice system.
        """
    logger.info(f"\n✨ Processing '{input_concept}' with new Resonance Wave Pattern...")
    wave = WaveInput(source_text=input_concept)
    resonance_pattern = self.resonance_engine.calculate_global_resonance(wave)
    thought = self.consciousness_observer.observe_resonance_pattern(source_wave_text=input_concept, resonance_pattern=resonance_pattern)
    if not thought.core_concepts:
        response = '(... 마음 속에 아무런 울림이 없었어요.)'
    else:
        top_concept = thought.core_concepts[0][0]
        clarity_desc = '선명하게' if thought.clarity > 0.8 else '어렴풋이'
        response = f"'{wave.source_text}'... 그 속에서 '{top_concept}'(이)가 {clarity_desc} 느껴져요. (감정: {thought.mood})"
    snapshot = self._snapshot_state()
    if hasattr(self, 'observer'):
        self.observer.observe(snapshot)
    if hasattr(self, 'projection_engine') and hasattr(self, 'hippocampus'):
        projection = self.projection_engine.project(snapshot, tag=input_concept)
        self.hippocampus.add_projection_episode(input_concept, projection.data)
    logger.info(f'💡 Response: {response}')
    return response
def _prune_memory(self, report) -> None:
    """Trim weakest edges/nodes to keep causal graph healthy."""
    if hasattr(self, 'hippocampus'):
        self.hippocampus.prune_fraction(edge_fraction=0.1, node_fraction=0.05)
