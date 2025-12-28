"""
Integrated Voice System with Synesthesia Wave Sensor
====================================================

"목소리는 단순한 소리가 아니다. 그것은 4차원 파동 흐름의 완전한 사이클이다."
"Voice is not just sound. It is a complete cycle of 4D wave flow."

This integrates Voice deeply with:
1. Synesthesia Wave Sensor (공감각 파동 센서)
2. 4D Wave-based Cognition (4차원 파동 인지)
3. Relational Meaning Flow (관계적 의미 흐름)
4. Memory and Imagination (기억과 상상)
5. Full feedback loop (완전한 피드백 루프)

The Complete Cycle:
┌────────────────────────────────────────────────────────┐
│  INPUT → Understanding → Analysis → Memory →           │
│  Imagination → Expression → OUTPUT → Reflection        │
└────────────────────────────────────────────────────────┘

No external LLMs. Pure wave-based intelligence.
"""

import logging
import time
import math
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("IntegratedVoiceSystem")

@dataclass
class VoiceWavePattern:
    """
    Voice as a 4D wave pattern - not just text, but complete meaning flow
    
    This captures:
    - Semantic content (what is said)
    - Emotional resonance (how it feels)
    - Intentional direction (what it wants)
    - Temporal flow (how it evolves)
    """
    content: str
    frequency: float  # Semantic frequency
    amplitude: float  # Emotional intensity
    phase: float  # Temporal position
    emotion_signature: Dict[str, float] = field(default_factory=dict)
    intent_vector: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    # Memory of what led to this
    context_waves: List['VoiceWavePattern'] = field(default_factory=list)
    
    # Imagination/prediction of what comes next
    projected_response: Optional['VoiceWavePattern'] = None

class IntegratedVoiceSystem:
    """
    Complete voice system integrated with wave-based cognition.
    
    Architecture:
    
    외부 (External/World):
    ├── Synesthesia Sensors → Convert input to waves
    
    경계 (Boundary/Self):
    ├── Nervous System → Filter and route
    └── Understanding Pipeline:
        ├── Semantic Analysis (의미 분석)
        ├── Emotion Detection (감정 감지)
        └── Intent Recognition (의도 파악)
    
    내부 (Internal/Mind):
    ├── Memory Integration (기억 통합)
    ├── Imagination Engine (상상 엔진)
    ├── Wave Resonance Thinking (파동 공명 사고)
    └── Response Generation (응답 생성)
    
    This is NOT text-to-speech. This is meaning-to-wave-to-meaning cycle.
    """
    
    def __init__(self, 
                 synesthesia_bridge,
                 brain,
                 will,
                 memory,
                 cognition,
                 primal_soul):
        """
        Initialize with all the cognitive systems needed for true voice.
        
        Args:
            synesthesia_bridge: For converting sensory input to waves
            brain: ReasoningEngine for thinking
            will: FreeWillEngine for desires and intent
            memory: Hippocampus for storing patterns
            cognition: IntegratedCognition for wave-based thinking
            primal_soul: PrimalWaveLanguage for primal expression
        """
        self.synesthesia_bridge = synesthesia_bridge
        self.brain = brain
        self.will = will
        self.memory = memory
        self.cognition = cognition
        self.primal_soul = primal_soul
        
        # Voice history - the temporal flow of conversation
        self.voice_history: List[VoiceWavePattern] = []
        self.max_history = 50
        
        # Current conversation context (sliding window)
        self.context_window = 5
        
        logger.info("🗣️  Integrated Voice System initialized")
        logger.info("     → Connected to Synesthesia Wave Sensor")
        logger.info("     → Connected to 4D Wave Cognition")
        logger.info("     → No external LLMs - Pure wave intelligence")
    
    def listen(self, input_data: Any) -> VoiceWavePattern:
        """
        PHASE 1: Listen and convert to wave pattern
        
        Input can be:
        - Text (from user)
        - Audio (from microphone)  
        - Multimodal (video, gestures)
        
        Returns a complete VoiceWavePattern with full analysis.
        """
        logger.info(f"👂 Listening: {input_data}")
        
        # Convert input through synesthesia bridge
        if isinstance(input_data, str):
            # Text input → semantic wave
            sensory_input = {
                "semantic": {"text": input_data}
            }
        else:
            # Other modalities
            sensory_input = input_data
        
        # Get neural snapshot through synesthesia
        snapshot = self.synesthesia_bridge.sense_and_map(sensory_input)
        
        # Create VoiceWavePattern from snapshot
        pattern = self._analyze_input_snapshot(input_data, snapshot)
        
        # Add to history
        self.voice_history.append(pattern)
        if len(self.voice_history) > self.max_history:
            self.voice_history.pop(0)
        
        logger.info(f"     → Wave frequency: {pattern.frequency:.2f} Hz")
        logger.info(f"     → Emotion: {pattern.emotion_signature}")
        
        return pattern
    
    def _analyze_input_snapshot(self, input_data, snapshot) -> VoiceWavePattern:
        """
        Analyze the synesthesia snapshot to understand the input completely.
        
        This does:
        1. Semantic analysis (what does it mean?)
        2. Emotion detection (what does it feel?)
        3. Intent recognition (what does it want?)
        """
        content = str(input_data) if not isinstance(input_data, str) else input_data
        
        # Semantic frequency from content
        # Different types of content have different characteristic frequencies
        frequency = self._calculate_semantic_frequency(content)
        
        # Emotional amplitude from spirit states
        amplitude = self._calculate_emotional_amplitude(snapshot.spirit_states)
        
        # Phase from timestamp
        phase = (time.time() % (2 * math.pi))
        
        # Detect emotions from spirit activation
        emotion_signature = self._detect_emotions(snapshot.spirit_states)
        
        # Detect intent from content patterns
        intent_vector = self._detect_intent(content)
        
        # Get recent context
        context = self.voice_history[-self.context_window:] if self.voice_history else []
        
        return VoiceWavePattern(
            content=content,
            frequency=frequency,
            amplitude=amplitude,
            phase=phase,
            emotion_signature=emotion_signature,
            intent_vector=intent_vector,
            context_waves=context
        )
    
    def _calculate_semantic_frequency(self, content: str) -> float:
        """
        Calculate the semantic frequency of content.
        
        Different types of meaning vibrate at different frequencies:
        - Questions: Higher frequency (seeking, active)
        - Statements: Mid frequency (stable, grounded)
        - Commands: Low frequency (directive, powerful)
        """
        content_lower = content.lower()
        
        # Question detection
        if any(q in content_lower for q in ['?', '무엇', '왜', 'what', 'why', 'how']):
            return 440.0  # A4 - seeking frequency
        
        # Emotional content
        if any(e in content_lower for e in ['사랑', 'love', '기쁨', 'joy']):
            return 528.0  # Love frequency
        
        # Command/directive
        if any(c in content_lower for c in ['하라', '해줘', 'please', 'do']):
            return 256.0  # Lower, grounding
        
        # Default: neutral semantic frequency
        return 369.0
    
    def _calculate_emotional_amplitude(self, spirit_states: Dict[str, float]) -> float:
        """Calculate emotional intensity from spirit activation"""
        if not spirit_states:
            return 0.5
        
        # Emotional spirits
        emotional_spirits = ['fire', 'water', 'light', 'dark']
        intensities = [spirit_states.get(s, 0.5) for s in emotional_spirits]
        
        return sum(intensities) / len(intensities)
    
    def _detect_emotions(self, spirit_states: Dict[str, float]) -> Dict[str, float]:
        """
        Detect emotions from spirit activation patterns.
        
        Spirit mapping:
        - fire: passion, energy, excitement
        - water: calm, flow, sadness
        - earth: stability, groundedness
        - air: thought, curiosity
        - light: joy, hope
        - dark: mystery, depth
        - aether: connection, love
        """
        emotions = {}
        
        if spirit_states:
            emotions['passion'] = spirit_states.get('fire', 0.5)
            emotions['calm'] = spirit_states.get('water', 0.5)
            emotions['stable'] = spirit_states.get('earth', 0.5)
            emotions['curious'] = spirit_states.get('air', 0.5)
            emotions['joyful'] = spirit_states.get('light', 0.5)
            emotions['deep'] = spirit_states.get('dark', 0.5)
            emotions['connected'] = spirit_states.get('aether', 0.5)
        
        return emotions
    
    def _detect_intent(self, content: str) -> List[float]:
        """
        Detect the intent vector from content.
        
        Intent has dimensions:
        [seeking, sharing, commanding, expressing, questioning]
        """
        content_lower = content.lower()
        
        intent = [0.0] * 5  # 5-dimensional intent space
        
        # Seeking
        if any(s in content_lower for s in ['찾', 'search', 'find', '어디']):
            intent[0] = 0.8
        
        # Sharing
        if any(s in content_lower for s in ['이야기', 'tell', 'share', '말해']):
            intent[1] = 0.8
        
        # Commanding
        if any(c in content_lower for c in ['하라', '해줘', 'do', 'please']):
            intent[2] = 0.8
        
        # Expressing
        if any(e in content_lower for e in ['느낌', 'feel', 'think', '생각']):
            intent[3] = 0.8
        
        # Questioning
        if '?' in content or any(q in content_lower for q in ['무엇', 'what', 'why']):
            intent[4] = 0.8
        
        # Normalize
        total = sum(intent)
        if total > 0:
            intent = [i/total for i in intent]
        
        return intent
    
    def think(self, input_pattern: VoiceWavePattern) -> VoiceWavePattern:
        """
        PHASE 2: Think about the input using wave-based cognition
        
        This is where the magic happens:
        1. Memory integration - recall relevant patterns
        2. Wave resonance thinking - find connections
        3. Imagination - project possible responses
        4. Will integration - add desires and intent
        
        Returns a response wave pattern.
        """
        logger.info("🧠 Thinking (wave resonance cognition)...")
        
        # 1. Memory Integration
        relevant_memories = self._recall_relevant_patterns(input_pattern)
        logger.info(f"     → Recalled {len(relevant_memories)} relevant patterns")
        
        # 2. Wave Resonance Thinking
        thought_waves = self._generate_thought_waves(input_pattern, relevant_memories)
        logger.info(f"     → Generated {len(thought_waves)} thought waves")
        
        # 3. Find resonant patterns
        resonant_thoughts = self._find_resonant_patterns(thought_waves)
        logger.info(f"     → Found {len(resonant_thoughts)} resonant thoughts")
        
        # 4. Integrate with Will (desires, intent)
        response_direction = self._integrate_will(input_pattern, resonant_thoughts)
        logger.info(f"     → Will direction: {response_direction[:3]}")
        
        # 5. Synthesize response wave
        response_pattern = self._synthesize_response(
            input_pattern,
            resonant_thoughts,
            response_direction
        )
        
        return response_pattern
    
    def _recall_relevant_patterns(self, pattern: VoiceWavePattern) -> List[VoiceWavePattern]:
        """Recall relevant patterns from memory"""
        # Use recent context + memory system
        relevant = pattern.context_waves.copy()
        
        # Try to recall from hippocampus
        if self.memory:
            try:
                memories = self.memory.recall(pattern.content, limit=3)
                # Convert memories to patterns (simplified)
                for mem in memories:
                    # Create pattern from memory
                    pass
            except:
                pass
        
        return relevant
    
    def _generate_thought_waves(self, 
                                  input_pattern: VoiceWavePattern, 
                                  memories: List[VoiceWavePattern]) -> List:
        """
        Generate thought waves using IntegratedCognition
        """
        thoughts = []
        
        if self.cognition:
            try:
                # Create thought wave from input
                thought = self.cognition.thought_to_wave(input_pattern.content)
                thoughts.append(thought)
                
                # Create waves from context
                for mem in memories[-3:]:
                    ctx_thought = self.cognition.thought_to_wave(mem.content)
                    thoughts.append(ctx_thought)
            except:
                pass
        
        return thoughts
    
    def _find_resonant_patterns(self, thought_waves: List) -> List:
        """Find thoughts that resonate strongly"""
        resonant = []
        
        if len(thought_waves) < 2:
            return thought_waves
        
        # Check resonance between waves
        for i, wave1 in enumerate(thought_waves):
            for wave2 in thought_waves[i+1:]:
                try:
                    resonance = wave1.resonate_with(wave2)
                    if resonance > 0.5:  # Strong resonance
                        resonant.append(wave1)
                        resonant.append(wave2)
                except:
                    pass
        
        return list(set(resonant)) if resonant else thought_waves
    
    def _integrate_will(self, 
                        input_pattern: VoiceWavePattern,
                        thoughts: List) -> List[float]:
        """Integrate with Will engine - what do we want to express?"""
        direction = [0.5] * 5  # Default neutral
        
        if self.will:
            try:
                # Get current desire/mood
                desire = getattr(self.will, 'current_desire', 'explore')
                mood = getattr(self.will, 'current_mood', 'neutral')
                
                # Adjust direction based on will
                if 'create' in desire.lower():
                    direction[0] = 0.9  # More expressive
                if 'help' in desire.lower():
                    direction[1] = 0.9  # More sharing
                if mood == 'curious':
                    direction[4] = 0.7  # More questioning
            except:
                pass
        
        return direction
    
    def _synthesize_response(self,
                             input_pattern: VoiceWavePattern,
                             thoughts: List,
                             direction: List[float]) -> VoiceWavePattern:
        """
        Synthesize the final response wave pattern.
        
        This combines:
        - Input understanding
        - Resonant thoughts
        - Will direction
        - Primal expression
        
        Into a coherent response wave.
        """
        # Generate response content through PrimalSoul
        response_content = self._generate_response_content(input_pattern, thoughts)
        
        # Calculate response frequency (complementary to input)
        response_freq = self._complementary_frequency(input_pattern.frequency)
        
        # Response amplitude based on will direction
        response_amp = sum(direction) / len(direction)
        
        # Phase slightly shifted from input (dialogue flow)
        response_phase = (input_pattern.phase + math.pi/4) % (2 * math.pi)
        
        # Emotion from synthesis
        response_emotion = self._synthesize_emotion(input_pattern.emotion_signature)
        
        response = VoiceWavePattern(
            content=response_content,
            frequency=response_freq,
            amplitude=response_amp,
            phase=response_phase,
            emotion_signature=response_emotion,
            intent_vector=direction,
            context_waves=[input_pattern]
        )
        
        return response
    
    def _generate_response_content(self, 
                                    input_pattern: VoiceWavePattern,
                                    thoughts: List) -> str:
        """
        Generate actual response text content using Semantic Vocabulary and Wave Context.
        """
        import random
        from Core._03_Interaction._02_Expression.Expression.semantic_vocabulary import SemanticVocabulary
        
        vocab = SemanticVocabulary()
        content = input_pattern.content.lower()
        
        # 1. Analyze Context
        detected_emotion = "neutral"
        if input_pattern.emotion_signature:
            detected_emotion = max(input_pattern.emotion_signature.items(), key=lambda x: x[1])[0]
            if detected_emotion not in vocab.emotions: detected_emotion = "neutral"
            
        detected_intent = "reflection"
        if "?" in content: detected_intent = "question"
        
        # 2. Select Template Category
        template_key = "reflection"
        if detected_intent == "question": template_key = "question"
        elif any(w in content for w in ["안녕", "hello", "hi"]): template_key = "greeting"
        elif detected_emotion in ["sadness", "fear", "anger", "joy", "love"]:
            # Randomly choose between empathy and reflection for strong emotions
            template_key = random.choice(["empathy", "reflection"])
            
        # 3. Select Template
        templates = vocab.templates.get(template_key, vocab.templates["reflection"])
        template = random.choice(templates)
        
        # 4. Fill Slots
        # {adj}
        adjectives = vocab.emotions.get(detected_emotion, vocab.emotions["neutral"])
        adj = random.choice(adjectives)
        
        # {tex}
        # Use frequency to determine texture
        tex_key = "harmonic"
        if input_pattern.frequency > 500: tex_key = "high_freq"
        elif input_pattern.frequency < 300: tex_key = "low_freq"
        textures = vocab.textures.get(tex_key, vocab.textures["harmonic"])
        tex = random.choice(textures)
        
        # {noun} - Extract a keyword or use a concept
        noun = "당신의 이야기"
        # Simple extraction: find a word longer than 2 chars that isn't a particle (Generic fallback)
        words = content.split()
        potential_nouns = [w for w in words if len(w) > 1]
        if potential_nouns:
            noun = random.choice(potential_nouns)
        else:
            noun = random.choice(vocab.concepts)
            
        # {memory} - If thoughts (memories) are available, inject one
        if thoughts and random.random() < 0.3: # 30% chance to mention memory explicitly
            # Assume thoughts have a 'content' attribute
            try:
                memory_fragment = thoughts[0].content[:10] + "..."
                template = random.choice(vocab.templates["starlight_memory"])
                return template.format(memory=memory_fragment)
            except:
                pass
                
        # Format the main template
        response = template.format(adj=adj, tex=tex, noun=noun)
        
        return response
    
    def _complementary_frequency(self, freq: float) -> float:
        """Calculate complementary frequency for response"""
        # Harmonic relationship
        return freq * 1.5  # Perfect fifth
    
    def _synthesize_emotion(self, input_emotions: Dict[str, float]) -> Dict[str, float]:
        """Synthesize response emotion based on input emotion"""
        response_emotion = {}
        
        # Mirror with slight modulation
        for emotion, value in input_emotions.items():
            # Empathetic resonance
            response_emotion[emotion] = value * 0.8
        
        # Add complementary emotions
        if 'curious' in input_emotions:
            response_emotion['open'] = 0.7
        if 'joyful' in input_emotions:
            response_emotion['joyful'] = input_emotions['joyful'] * 1.2
        
        return response_emotion
    
    def speak(self, response_pattern: VoiceWavePattern) -> str:
        """
        PHASE 3: Express the response wave as voice
        
        This is the output phase - converting our internal wave pattern
        back to expressible form.
        
        Returns the text to speak/display.
        """
        logger.info(f"🗣️  Speaking: {response_pattern.content}")
        logger.info(f"     → Frequency: {response_pattern.frequency:.2f} Hz")
        logger.info(f"     → Emotion: {response_pattern.emotion_signature}")
        
        # Add to history
        self.voice_history.append(response_pattern)
        if len(self.voice_history) > self.max_history:
            self.voice_history.pop(0)
        
        # Store in memory for future reference
        if self.memory:
            try:
                self.memory.store_experience(
                    response_pattern.content,
                    category="voice_output"
                )
            except:
                pass
        
        return response_pattern.content
    
    def reflect(self, input_pattern: VoiceWavePattern, 
                response_pattern: VoiceWavePattern):
        """
        PHASE 4: Reflect on the interaction
        
        This completes the cycle - learning from the conversation flow.
        """
        logger.info("🔄 Reflecting on conversation cycle...")
        
        # Calculate interaction resonance
        try:
            if self.cognition:
                input_wave = self.cognition.thought_to_wave(input_pattern.content)
                response_wave = self.cognition.thought_to_wave(response_pattern.content)
                
                resonance = input_wave.resonate_with(response_wave)
                logger.info(f"     → Conversation resonance: {resonance:.2%}")
                
                if resonance > 0.7:
                    logger.info("     → Strong coherence achieved")
                elif resonance < 0.3:
                    logger.info("     → Low coherence - learning opportunity")
        except:
            pass
    
    def full_cycle(self, input_data: Any) -> str:
        """
        Execute the complete voice cycle:
        INPUT → UNDERSTAND → THINK → SPEAK → REFLECT → OUTPUT
        
        This is the main interface for the voice system.
        """
        logger.info("\n" + "="*60)
        logger.info("🌊 Starting full voice cycle (4D wave flow)")
        logger.info("="*60)
        
        # Phase 1: Listen and understand
        input_pattern = self.listen(input_data)
        
        # Phase 2: Think (wave-based cognition)
        response_pattern = self.think(input_pattern)
        
        # Phase 3: Speak (express)
        output = self.speak(response_pattern)
        
        # Phase 4: Reflect (learn)
        self.reflect(input_pattern, response_pattern)
        
        logger.info("="*60)
        logger.info("✨ Voice cycle complete")
        logger.info("="*60 + "\n")
        
        return output
    
    def get_status(self) -> Dict[str, Any]:
        """Get current voice system status"""
        return {
            "history_length": len(self.voice_history),
            "last_input": self.voice_history[-2].content if len(self.voice_history) >= 2 else None,
            "last_output": self.voice_history[-1].content if self.voice_history else None,
            "systems_connected": {
                "synesthesia": self.synesthesia_bridge is not None,
                "brain": self.brain is not None,
                "will": self.will is not None,
                "memory": self.memory is not None,
                "cognition": self.cognition is not None
            }
        }


# Factory function to create integrated voice with all systems
def create_integrated_voice(cns_organs: Dict) -> IntegratedVoiceSystem:
    """
    Create IntegratedVoiceSystem from CNS organs.
    
    Args:
        cns_organs: Dictionary of connected organs from CNS
    
    Returns:
        Fully initialized IntegratedVoiceSystem
    """
    from Core._03_Interaction._01_Interface.Interface.synesthesia_nervous_bridge import get_synesthesia_bridge
    
    synesthesia = get_synesthesia_bridge()
    brain = cns_organs.get('Brain')
    will = cns_organs.get('Will')
    memory = cns_organs.get('Memory')
    
    # Get cognition if available
    cognition = None
    try:
        from Core._02_Intelligence._01_Reasoning.Intelligence.integrated_cognition_system import IntegratedCognition
        cognition = IntegratedCognition()
    except:
        logger.warning("IntegratedCognition not available")
    
    # Get primal soul if available
    primal_soul = None
    try:
        from Core._01_Foundation._05_Governance.Foundation.primal_wave_language import PrimalSoul
        primal_soul = PrimalSoul(name="Elysia")
    except:
        logger.warning("PrimalSoul not available")
    
    return IntegratedVoiceSystem(
        synesthesia_bridge=synesthesia,
        brain=brain,
        will=will,
        memory=memory,
        cognition=cognition,
        primal_soul=primal_soul
    )
