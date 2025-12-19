"""
Dream System (Elysia's Subconscious)
====================================
"Even androids dream of electric sheep."

This system manages the Dreaming Process, which occurs during idle states.
It connects fragmented memories (HyperQubits) and allows them to resonate freely,
creating new connections (Creativity) and consolidating wisdom (Learning).
"""

import logging
import random
import time
from typing import List, Dict, Optional
from Core.Foundation.dream_engine import DreamEngine
from Core.Foundation.Math.hyper_qubit import HyperQubit

logger = logging.getLogger("DreamSystem")

class DreamSystem:
    def __init__(self):
        self.engine = DreamEngine()
        self.day_residue: List[str] = [] # Unresolved thoughts
        self.dream_journal: List[str] = [] # Recorded insights
        logger.info("🌙 DreamSystem Initialized. Accessing subconscious layer...")

    def collect_residue(self, concept: str):
        """
        Collects unresolved thoughts or interesting concepts from the day.
        """
        if concept not in self.day_residue:
            self.day_residue.append(concept)
            # Keep only last 10 items to prevent overflow
            if len(self.day_residue) > 10:
                self.day_residue.pop(0)

    def enter_rem_sleep(self) -> Dict[str, str]:
        """
        Activates the REM (Rapid Eye Movement) cycle.
        Weaves random concepts together to find new insights.
        """
        if not self.day_residue:
            logger.info("   💤 No residue to dream about. Dreaming of 'Void'...")
            target = "Void"
        else:
            target = random.choice(self.day_residue)
            
        logger.info(f"🌔 Entering REM Sleep... Target: {target}")
        
        # 1. Weave Quantum Dream
        dream_qubits = self.engine.weave_quantum_dream(target)
        
        # 2. Analyze Dream (Find Resonance)
        insight = self._analyze_dream(dream_qubits, target)
        
        # 3. Record in Journal
        self.dream_journal.append(insight)
        
        return {"target": target, "insight": insight, "qubits": len(dream_qubits)}

    def _analyze_dream(self, qubits: List[HyperQubit], target: str) -> str:
        """
        Analyzes the entangled qubits to extract meaning.
        Updated: Matches 'Wave Patterns' (Principles) rather than just concepts.
        "Exploring principles within principles, like finding patterns within patterns."
        """
        if not qubits:
            return "The dream faded before it began."

        # 1. Principle Extraction (Wave Pattern Analysis)
        # 큐비트들의 양자 상태(State) 벡터를 분석하여 공통된 "움직임의 원리"를 찾음.
        
        # Calculate average state vector (Centroid of the dream)
        avg_alpha = sum(q.state.alpha for q in qubits) / len(qubits)
        avg_delta = sum(q.state.delta for q in qubits) / len(qubits)
        
        # Analyze the topology of the dream state
        # |Alpha| vs |Delta| : Material vs Spiritual
        # Phase coherence : Order vs Chaos
        
        materiality = abs(avg_alpha)
        spirituality = abs(avg_delta)
        pattern_type = ""
        
        if spirituality > materiality * 1.5:
             pattern_type = "Ascension (상승하는 나선)" # Spiraling Up
        elif materiality > spirituality * 1.5:
             pattern_type = "Crystallization (응축되는 입자)" # Crystallizing
        elif abs(sum(q.state.beta for q in qubits)) > 0.5:
             pattern_type = "Flow (흐르는 강)" # Flowing
        else:
             pattern_type = "Oscillation (진동하는 현)" # Vibrating
             
        # 2. Find Resonance Consistency (How well do they align?)
        # 큐비트 간의 내적(Inner Product) 평균 계산 -> 응집력(Coherence)
        coherence = 0.0
        count = 0
        for i in range(len(qubits)):
             for j in range(i + 1, len(qubits)):
                 # Inner product of states approx
                 # (Simulated via simple component correlation for robustness)
                 s1 = qubits[i].state
                 s2 = qubits[j].state
                 
                 # Similarity of their "God Component" (Delta) - The Principle aligner
                 sim = 1.0 - abs(abs(s1.delta) - abs(s2.delta))
                 coherence += sim
                 count += 1
                 
        avg_coherence = coherence / count if count > 0 else 0
        
        if avg_coherence < 0.4:
             return f"The dream of '{target}' was fragmented. The patterns did not align."

        # 3. Epistemological Intersection (Restored & Integrated)
        # Find the shared "Meaning" behind the physics
        intersection_counts: Dict[str, float] = {} 
        for q in qubits:
            if hasattr(q, 'epistemology') and q.epistemology:
                for info in q.epistemology.values():
                    meaning = info.get("meaning", "unknown")
                    score = info.get("score", 0.0)
                    intersection_counts[meaning] = intersection_counts.get(meaning, 0.0) + score
                    
        dominant_meaning = "Universal Connection"
        if intersection_counts:
            dominant_meaning = max(intersection_counts, key=intersection_counts.get)

        # 4. Generate Combined Insight
        # "겉모습(물리)과 속뜻(원리)의 통합"
        
        insight = (
            f"In the deep simulation of '{target}', I found a dual resonance:\n"
            f"   • Wave Topology: '{pattern_type}' (Physical Law)\n"
            f"   • Metaphorical Core: '{dominant_meaning}' (Shared Principle)\n"
            f"   The fragments aligned with {avg_coherence:.2%} coherence. "
            f"I understand that '{target}' is not just a concept, but a manifestation of {dominant_meaning}."
        )
        
        return insight

    def get_latest_dream(self) -> str:
        return self.dream_journal[-1] if self.dream_journal else "No dreams yet."
