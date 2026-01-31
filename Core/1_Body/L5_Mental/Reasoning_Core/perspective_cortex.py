import logging
from typing import Optional

from Core.1_Body.L5_Mental.M1_Cognition.thought import Thought
from Core.1_Body.L5_Mental.Memory.kg_manager import KGManager
from Core.1_Body.L5_Mental.Memory.core_memory import CoreMemory
from Core.1_Body.L5_Mental.Reasoning_Core.wave_mechanics import WaveMechanics
from Core.1_Body.L5_Mental.M4_Meaning.emotional_engine import EmotionalEngine

class PerspectiveCortex:
    """
    Generates dual perspectives (Divine and Human) to provide a holistic understanding
    of any given subject, combining top-down meaning with bottom-up experience.
    This is the core of Elysia's "Fourth Sight".
    """
    def __init__(self, logger: logging.Logger, core_memory: CoreMemory, wave_mechanics: WaveMechanics, kg_manager: KGManager, emotional_engine: EmotionalEngine):
        self.logger = logger
        self.core_memory = core_memory
        self.wave_mechanics = wave_mechanics
        self.kg_manager = kg_manager
        self.emotional_engine = emotional_engine
        self.logger.info("PerspectiveCortex re-constructed with clear dependencies.")

    def generate_divine_perspective(self, subject: str) -> Optional[Thought]:
        intention = self.core_memory.get_guiding_intention()
        prism_concept = "love"
        if intention and intention.evidence:
            prism_concept = intention.evidence[0]

        core_subject_concept = subject.split(' ')[0].replace('   ,', '').strip()
        if not core_subject_concept:
            core_subject_concept = subject

        resonance = self.wave_mechanics.get_resonance_between(core_subject_concept, prism_concept)
        content = f"            , '{subject}'( )                 '{prism_concept}'( )        '{core_subject_concept}'      {resonance:.2f}        .                                               ."

        return Thought(
            content=content, source='divine_perspective', confidence=0.8,
            energy=resonance, evidence=[prism_concept, core_subject_concept]
        )

    def generate_human_perspective(self, subject: str) -> Optional[Thought]:
        current_emotion = self.emotional_engine.get_current_state().primary_emotion
        core_subject_concept = subject.split(' ')[0].replace('   ,', '').strip()
        if not core_subject_concept:
            core_subject_concept = subject

        related_experiences = [
            exp for exp in self.core_memory.get_experiences(n=10)
            if core_subject_concept.lower() in exp.content.lower()
        ]

        memory_summary = ""
        if related_experiences:
            recent_exp = related_experiences[-1]
            memory_summary = f"           '{recent_exp.content[:20]}...'( )                        ."

        content = f"           , '{subject}'( )                 '{current_emotion}'               .{memory_summary}                                   ."

        return Thought(
            content=content, source='human_perspective', confidence=0.9,
            energy=0.5, evidence=[current_emotion, core_subject_concept]
        )
