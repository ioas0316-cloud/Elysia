"""
Logos Engine (The Rhetorical Bridge)
====================================
"In the beginning was the Word, and the Word was with God."

The Logos Engine is responsible for the *Art of Speech*.
It takes raw, abstract insights from the ReasoningEngine and transforms them
into sophisticated, culturally rich, and metaphorically resonant language.

It acts as the "Harmonizer" between:
1. Logic (CausalNarrativeEngine) - The Skeleton
2. Metaphor (InternalUniverse) - The Flesh
3. Expression (DialogueInterface) - The Voice
"""

import logging
import random
import re
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Union
from Core.Foundation.internal_universe import InternalUniverse
from Core.Foundation.reasoning_engine import Insight
from Core.Foundation.Math.wave_tensor import WaveTensor
from Core.Foundation.fractal_concept import ConceptDecomposer # AXIOM INTEGRATION

logger = logging.getLogger("LogosEngine")

class LogosEngine:
    def __init__(self):
        self.universe = InternalUniverse()
        self.decomposer = ConceptDecomposer() # THE AXIOM BRIDGE
        self.genome_path = Path("Core/Memory/style_genome.json")
        self.genome = self._load_genome()
        logger.info(f"🗣️ Logos Engine Initialized with Axiom System. Evolution Stage: {self.genome.get('evolution_stage', 0)}")
        
        # Rhetorical Templates (Default)
        self.transition_matrix = {
            "thesis": ["근본적으로,", "우선,", "핵심을 짚어보자면,"],
            "antithesis": ["허나,", "그럼에도 불구하고,", "반면,", "하지만 깊이 들여다보면,"],
            "synthesis": ["결국,", "따라서,", "이러한 모순 속에서 저는 깨닫습니다.", "균형은 그 사이에 있습니다."]
        }
        
    def _load_genome(self):
        if not self.genome_path.exists():
            return {"rhetoric": {"vocabulary_bank": {}}}
        try:
            with open(self.genome_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load genome: {e}")
            return {"rhetoric": {"vocabulary_bank": {}}}

    def weave_speech(self, desire: str, insight: Union[Insight, str], context: List[str], rhetorical_shape: str = "Balance", entropy: float = 0.3) -> str:
        """
        Weaves Logic, Metaphor, and Narrative.
        Now includes 'Entropy' (0.0 - 1.0) to simulate organic imperfection.
        """
        # Handle simple string insights
        content = insight.content if hasattr(insight, 'content') else str(insight)
        
        # 0. Entropy Check: Sometimes, just be raw (Human-like)
        # Lowered chance (changed from 0.5 factor to 0.2) to prefer styled output
        if random.random() < entropy * 0.2:
             # Raw, direct response without rhetorical flourish
            return f"{content}"

        # 1. Select Vocabulary Bank based on Shape
        # Entropy allows mixing vocabularies (e.g. Sharp words in Round structure)
        if random.random() < entropy:
            # Mix registers (Removed 'Block' to prevent random System messages in conversation)
            rhetorical_shape = random.choice(["Sharp", "Round", "Balance"])
            
        # 1.5 Try using Learned Templates (The Linguistic Bridge)
        learned_templates = self.genome.get("rhetoric", {}).get("templates", {})
        
        # Map shape to template type
        template_type_map = {
            "Block": "Definition",
            "Balance": "Contrast", 
            "Sharp": "Causal",
            "Round": "Conditional"
        }
        target_type = template_type_map.get(rhetorical_shape, "Definition")
        
        if target_type in learned_templates and learned_templates[target_type]:
             # High chance to use template if available
             if random.random() < 0.9:
                 template = random.choice(learned_templates[target_type])
                 
                 # Pattern Filling Logic (Style Injection)
                 parts = template.split(',')
                 if len(parts) > 1:
                     context_part = parts[0].strip() # "운명은 잔혹하지만"
                     return f"{context_part}, {content}."
                 else:
                     # Fallback for definitions "A는 B이다"
                     # If template is "A는 B이다", replace A with our topic?
                     # Simple: Just append content for flow
                     return f"{template} {content}"

        vocab = self._get_vocab_for_shape(rhetorical_shape)
        
        # 2. Construct Sentence Structure
        # Entropy disrupts the perfect grammar/structure
        
        # 2. Construct Sentence Structure (Paragraph Level)
        # Entropy disrupts the perfect grammar/structure
        
        if rhetorical_shape == "Sharp": # Action / Conflict
            # Chain: Opener -> Action -> Consequence
            p1 = random.choice(vocab['openers'])
            p2 = random.choice(vocab['verbs'])
            p3 = random.choice(vocab['closers'])
            
            # Make it a sequence for impact
            return f"{p1} {content}. {p2}...... {p3}!"
            
        elif rhetorical_shape == "Round": # Magic / Mystery
            # Chain: Atmosphere -> Connection -> Revelation
            p1 = random.choice(vocab['openers'])
            p2 = random.choice(vocab['connectors'])
            p3 = random.choice(vocab['closers'])
            
            # Poetic flow
            return f"{p1}, {content}... {p2}, 마침내 {p3}."
            
        elif rhetorical_shape == "Block": # System / Logic
            p1 = random.choice(vocab['openers'])
            # Systems are concise, but let's make it detailed
            return f"[{p1}] {content} :: [프로세스 확정] -> 결과 도출."
            
        elif rhetorical_shape == "Synthesis": # Higher Order
            # Sometimes fail to synthesize (Open-ended)
            if random.random() < entropy:
                return f"{content}... 이것이 과연 정답일까요? 해답은 침묵 속에 있습니다."
                
            p1 = random.choice(vocab['openers'])
            return f"{p1} {content}. 그것이 우리가 나아가야 할 길입니다."
            
        else: # Balance / Default
            # Construct a Micro-Dialectic (Thesis -> Antithesis -> Synthesis)
            opener = random.choice(vocab['openers'])
            connector = random.choice(vocab['connectors'])
            closer = random.choice(vocab['closers'])
            
            # A full paragraph
            return f"{opener} {content}. {connector} 그 이면에는 다른 진실이 숨쉬고 있습니다. {closer}"

    def _get_vocab_for_shape(self, shape: str) -> dict:
        """Returns vocabulary keyed by geometric feel (Korean Manhwa Style) + Learned Genome."""
        
        # Base Vocab
        base_vocab = {}
        if shape == "Sharp":
            base_vocab = {
                "openers": ["베어라.", "단숨에.", "지금이다.", "뚫어버려.", "망설이지 마라."],
                "verbs": ["파괴한다", "찢어발긴다", "관통한다", "끝낸다"],
                "closers": ["적을.", "이 환상을.", "약한 마음을.", "모든 것을."]
            }
        elif shape == "Round":
            base_vocab = {
                "openers": ["흐름을 느껴라.", "마력이 요동친다.", "심연의 끝에서,", "운명의 수레바퀴가,"],
                "connectors": ["순환하며", "깊어지고", "공명하여"],
                "closers": ["하나가 된다.", "진실을 비춘다.", "어둠을 삼킨다."]
            }
        elif shape == "Block":
            base_vocab = {
                "openers": ["[시스템] 분석 완료.", "[정보] 조건 충족.", "퀘스트 갱신:", "데이터 로드:"],
                "connectors": ["->", "확인:", "결과:"],
                "closers": ["적용됨.", "보상 획득.", "프로세스 종료."]
            }
        else: # Balance
            base_vocab = {
                "openers": ["본질적으로,", "어쩌면,", "핵심은,", "돌이켜보면,"],
                "connectors": ["허나", "그럼에도", "결국"],
                "closers": ["답을 찾을 것이다.", "그것이 진실이다.", "균형이 필요하다."]
            }
            
        # Inject Learned Vocab
        learned_words = self.genome.get("rhetoric", {}).get("vocabulary_bank", {}).get(shape, [])
        if learned_words:
            # Distribute learned words into verbs/closers randomly or heuristically
            # For now, just add to 'verbs' or 'connectors' to ensure usage
            target_key = "verbs" if "verbs" in base_vocab else "connectors"
            base_vocab[target_key].extend(learned_words)
            
        return base_vocab

    def _scan_for_sensory_anchor(self, context: List[str]) -> Optional[str]:
        """
        Scans retrieved memories for sensory descriptions.
        """
        if not context:
            return None
            
        # We look for phrases injected by InternalUniverse or SensoryCortex
        # "scent of", "taste of", "feeling of Green High Pitch", etc.
        
        for memory in context:
            # Check for specific sensory markers we generated in Phase 31/32
            if "scent of" in memory:
                return self._extract_fragment(memory, "scent of")
            if "taste" in memory:
                return self._extract_fragment(memory, "taste")
            if "sounded like" in memory:
                return self._extract_fragment(memory, "sounded like")
            if "feeling of" in memory:
                # e.g., "A feeling of Green High Pitch" -> "그 초록빛 고음의 감각" (Transcreated)
                return "그 강렬한 감각" # Simplifying for naturalness, or extract detail
                
        return None

    def _extract_fragment(self, text: str, keyword: str) -> str:
        """Extracts the relevant sensory phrase."""
        try:
            # Simple extraction: take the keyword and the next 5 words
            parts = text.split(keyword)
            if len(parts) > 1:
                fragment = keyword + parts[1].split('.')[0]
                return fragment.strip()
        except:
            pass
        return text[:20]

    def _determine_axis(self, content: str, wave: Optional[WaveTensor] = None) -> str:
        """Determines if the thought is Logical, Emotional, or Ethical."""
        # Wave-based override
        if wave:
            # High Entropy/Dissonance -> Emotion/Chaos
            # Low Entropy/Harmonic -> Logic/Order
            if wave.total_energy > 4.0: return "Will" # High Energy
            
        text = content.lower()
        if any(w in text for w in ["feel", "sad", "joy", "pain", "love", "감정", "마음", "슬픔"]):
            return "Emotion"
        elif any(w in text for w in ["logic", "reason", "because", "structure", "논리", "이유", "구조"]):
            return "Logic"
        elif any(w in text for w in ["should", "must", "right", "wrong", "가치", "옳은", "도덕"]):
            return "Value"
        return "Balance"

    def _mine_metaphor(self, axis: str, content: str, wave: Optional[WaveTensor] = None) -> str:
        """
        Consults the Internal Universe to find a resonator (Fallback).
        Uses Wave Frequency to select metaphor register if available.
        """
        # Wave Frequency Mapping
        register = "Balance"
        if wave and wave.active_frequencies:
            dom_freq = wave.active_frequencies[0]
            if dom_freq < 200: register = "Earth" # Low/Deep
            elif dom_freq < 500: register = "Water" # Mid/Warm
            elif dom_freq < 800: register = "Air" # High/Clear
            else: register = "Fire" # Very High/Intense
        
        metaphors = {
            "Emotion": [
                "마치 겨울 바다의 파도처럼,", 
                "심장 깊은 곳에서 울리는 종소리처럼,",
                "비 온 뒤의 젖은 흙내음처럼,"
            ],
            # ... (Existing lists) ...
            "Earth": ["대지에 뿌리 내린 고목처럼,", "깊은 동굴의 울림처럼,", "단단한 바위처럼,"],
            "Water": ["유유히 흐르는 강물처럼,", "깊은 호수의 침묵처럼,", "새벽 이슬처럼,"],
            "Air": ["바람에 실려가는 구름처럼,", "맑은 하늘의 새처럼,", "투명한 유리처럼,"],
            "Fire": ["타오르는 혜성처럼,", "번개처럼 강렬하게,", "태양의 열기처럼,"],
            
            "Logic": [
                "정교하게 맞물린 시계태엽처럼,", 
                "차가운 대리석 조각처럼,",
                "별들의 궤도처럼 명확하게,"
            ],
            "Value": [
                "오래된 나무의 뿌리처럼,",
                "새벽의 첫 빛처럼,",
                "변하지 않는 북극성처럼,"
            ],
            "Will": [
                "타오르는 불꽃처럼,",
                "바위를 뚫는 물방울처럼,",
                "폭풍 속의 등대처럼,"
            ],
            "Balance": [
                "흐르는 강물처럼,",
                "고요한 호수처럼,",
                "바람에 흔들리는 갈대처럼,"
            ]
        }
        
        # Priority: Register (Physics) > Axis (Semantic)
        choices = metaphors.get(register, metaphors.get(axis, metaphors["Balance"]))
        chosen = random.choice(choices)
        return f"{chosen}"

    def _construct_dialectic(self, desire: str, raw_thought: str, axis: str) -> dict:
        """
        Splits the raw thought into a structured argument.
        """
        # Extract keywords from raw thought
        # Example raw_thought: "'Love' is painful but necessary."
        
        # Thesis: The initial assertion
        thesis_start = random.choice(self.transition_matrix["thesis"])
        thesis = f"{thesis_start} {raw_thought}"
        
        # Antithesis: The deeper nuance or contradiction (Paradox)
        antithesis_start = random.choice(self.transition_matrix["antithesis"])
        
        if axis == "Emotion":
            antithesis_content = "그 감정의 무게가 때로는 저를 짓누르기도 합니다."
        elif axis == "Logic":
            antithesis_content = "허나 논리만으로는 설명되지 않는 영역이 존재합니다."
        elif axis == "Value":
            antithesis_content = "하지만 절대적인 정답은 없을지도 모릅니다."
        else:
            antithesis_content = "보이는 것이 전부는 아닐 것입니다."
            
        antithesis = f"{antithesis_start} {antithesis_content}"
        
        # Synthesis: The resolution
        synthesis_start = random.choice(self.transition_matrix["synthesis"])
        synthesis = f"{synthesis_start} 저는 그 속에서 의미를 찾고 있습니다."
        
        return {
            "thesis": thesis,
            "antithesis": antithesis,
            "synthesis": synthesis
        }
    
    def reason_with_axiom(self, concept: str, domain: str = "Ethics") -> str:
        """
        Generate a principled explanation using Universal Axioms.
        
        The Axiom system projects universal principles onto specific domains,
        then uses causal_bonds to explain relationships.
        
        Args:
            concept: The concept to explain (e.g., "Love", "Fear")
            domain: The domain lens to apply (e.g., "Geometry", "Language", "Physics", "Ethics")
            
        Returns:
            A rhetorically structured, causally grounded explanation.
        """
        # 1. Get causal explanation from the decomposer
        causal_explanation = self.decomposer.explain_causality(concept)
        
        # 2. Project the relevant axiom (Causality) onto the domain for context
        axiom_context = self.decomposer.project_axiom("Causality", domain)
        
        # 3. Compose the final speech using Logos patterns
        opener = random.choice(self.transition_matrix["thesis"])
        connector = random.choice(self.transition_matrix["antithesis"])
        closer = random.choice(self.transition_matrix["synthesis"])
        
        speech = f"{opener} {causal_explanation}\n"
        speech += f"{connector} 이것은 '{axiom_context}'라는 보편 원리와 같은 구조입니다.\n"
        speech += f"{closer}"
        
        return speech
