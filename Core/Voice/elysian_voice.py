"""
Elysian Voice (통합된 목소리)
=============================
Core.Voice.elysian_voice

"All thoughts converge. One voice speaks."

This module unifies all neural outputs (Monad collapse, visual perception, 
intent) into a single coherent response — The Elysian Voice.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("Elysia.Voice")


@dataclass
class ElysianUtterance:
    """A single coherent output from Elysia."""
    content: str
    confidence: float
    emotional_tone: str
    source_summary: str
    energy: float


class ElysianVoice:
    """
    The unified output system.
    All perceptions, thoughts, and intents converge here to form a single voice.
    """
    
    def __init__(self):
        self.utterance_count = 0
        self.last_utterance = None
        
        # Emotional vocabulary for tone expression
        self.tone_map = {
            "high_energy": ["흥분된", "열정적인", "활기찬"],
            "low_energy": ["차분한", "고요한", "평온한"],
            "curious": ["호기심 가득한", "탐구적인", "궁금해하는"],
            "confident": ["확신에 찬", "단호한", "명확한"],
            "uncertain": ["불확실한", "탐색적인", "조심스러운"]
        }
        
        logger.info("🗣️ Elysian Voice initialized. Ready to speak.")
    
    def synthesize(self, 
                   monad_result: Optional[Dict] = None,
                   visual_context: Optional[Dict] = None,
                   intent: Optional[Dict] = None,
                   raw_thought: str = "") -> ElysianUtterance:
        """
        Synthesizes all inputs into a single coherent utterance.
        
        Args:
            monad_result: Output from MonadEngine collapse
            visual_context: Output from VisualRotor
            intent: Output from IntentCollider
            raw_thought: Direct thought content
        
        Returns:
            ElysianUtterance: A unified, coherent response
        """
        # 1. Gather energy from all sources
        total_energy = 0.0
        source_parts = []
        
        if monad_result:
            total_energy += monad_result.get("energy", 0.5)
            source_parts.append("monad")
        
        if visual_context:
            visual_energy = visual_context.get("focus", {}).get("signature", {}).get("energy", 0)
            total_energy += visual_energy / 255.0  # Normalize
            source_parts.append("vision")
        
        if intent:
            intent_strength = intent.get("motor_strength", 0.5)
            total_energy += intent_strength
            source_parts.append("intent")
        
        avg_energy = total_energy / max(1, len(source_parts))
        
        # 2. Determine emotional tone from context
        tone = self._determine_tone(avg_energy, intent, visual_context)
        
        # 3. Construct the utterance content
        content = self._construct_content(raw_thought, intent, visual_context, monad_result)
        
        # 4. Calculate confidence based on convergence
        confidence = self._calculate_confidence(source_parts, avg_energy)
        
        utterance = ElysianUtterance(
            content=content,
            confidence=confidence,
            emotional_tone=tone,
            source_summary="+".join(source_parts) if source_parts else "direct",
            energy=avg_energy
        )
        
        self.last_utterance = utterance
        self.utterance_count += 1
        
        logger.info(f"🗣️ Utterance #{self.utterance_count}: [{tone}] {content[:50]}...")
        
        return utterance
    
    def _determine_tone(self, energy: float, intent: Optional[Dict], 
                        visual: Optional[Dict]) -> str:
        """Determines the emotional tone of the utterance."""
        if energy > 0.7:
            return "confident"
        elif energy < 0.3:
            return "uncertain"
        
        if intent:
            primary_motor = intent.get("primary_motor", "")
            if primary_motor == "WISDOM":
                return "curious"
            elif primary_motor == "LOVE":
                return "warm"
        
        if visual:
            visual_sig = visual.get("focus", {}).get("signature", {})
            if visual_sig.get("motion", {}).get("is_moving"):
                return "alert"
        
        return "balanced"
    
    def _construct_content(self, raw: str, intent: Optional[Dict],
                           visual: Optional[Dict], monad: Optional[Dict]) -> str:
        """Constructs the actual content of the utterance."""
        parts = []
        
        if raw:
            parts.append(raw)
        
        if intent:
            parts.append(f"[의지: {intent.get('primary_motor', 'EXISTENCE')}]")
        
        if visual:
            interpretation = visual.get("focus", {}).get("interpretation", "")
            if interpretation:
                parts.append(f"[시각: {interpretation}]")
        
        if monad and isinstance(monad, dict):
            parts.append(f"[에너지: {monad.get('energy', 0):.2f}]")
        
        if not parts:
            parts.append("나는 엘리시아입니다.")
        
        return " ".join(parts)
    
    def _calculate_confidence(self, sources: list, energy: float) -> float:
        """Calculates confidence based on source convergence and energy."""
        source_bonus = len(sources) * 0.1
        return min(1.0, energy + source_bonus)
    
    def speak(self, utterance: ElysianUtterance) -> str:
        """
        Final output formatting — how Elysia actually 'speaks'.
        """
        tone_prefix = {
            "confident": "확신을 담아",
            "uncertain": "조심스럽게",
            "curious": "호기심을 담아",
            "warm": "따뜻하게",
            "alert": "주의 깊게",
            "balanced": "담담하게"
        }
        
        prefix = tone_prefix.get(utterance.emotional_tone, "")
        return f"{prefix} 말합니다: \"{utterance.content}\""


if __name__ == "__main__":
    voice = ElysianVoice()
    
    # Test with various inputs
    print("🗣️ Testing Elysian Voice...\n")
    
    # Test 1: Direct thought
    u1 = voice.synthesize(raw_thought="나는 생각한다, 고로 존재한다.")
    print(f"Test 1 (Direct): {voice.speak(u1)}")
    print(f"  Confidence: {u1.confidence:.2f}, Energy: {u1.energy:.2f}\n")
    
    # Test 2: With intent
    mock_intent = {"primary_motor": "WISDOM", "motor_strength": 0.8}
    u2 = voice.synthesize(intent=mock_intent, raw_thought="이 문제의 본질은 무엇인가?")
    print(f"Test 2 (Intent): {voice.speak(u2)}")
    print(f"  Tone: {u2.emotional_tone}, Source: {u2.source_summary}\n")
    
    # Test 3: Full convergence
    mock_monad = {"energy": 0.75}
    mock_visual = {"focus": {"signature": {"energy": 128}, "interpretation": "밝은 장면"}}
    u3 = voice.synthesize(
        monad_result=mock_monad,
        visual_context=mock_visual,
        intent=mock_intent,
        raw_thought="모든 감각이 하나로 수렴한다."
    )
    print(f"Test 3 (Full): {voice.speak(u3)}")
    print(f"  Confidence: {u3.confidence:.2f}, Sources: {u3.source_summary}")
    
    print("\n✨ Elysian Voice test complete.")
