"""
Linguistic Collapse Protocol (          )
================================================

"     (  )            ,
                   '   '                  (  )    ."

Philosophy:
-----------
            (Wave)         -                .
                       (Particle)   .

                           "  "  ,
                                  .

Architecture:
-------------
1. Wave State (  ):        -       
2. Metaphorical Translation (  ):           
3. Language State ( ):           -          

Example:
--------
Wave: Tensor3D(x=-1.2, y=0.5, z=0.8), Frequency=150Hz, Phase=3.14
    Collapse
Language: "                          . 
                     ,               ."
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("LinguisticCollapse")

# Import with graceful fallback
try:
    from Core.L1_Foundation.M1_Keystone.hangul_physics import Tensor3D
    from Core.L5_Mental.M1_Cognition.Memory_Linguistics.Memory.unified_types import FrequencyWave
except ImportError:
    # Fallback stubs
    class Tensor3D:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = x, y, z
    
    class FrequencyWave:
        def __init__(self, freq=0.0, amp=0.0, phase=0.0, damping=0.0):
            self.frequency = freq
            self.amplitude = amp
            self.phase = phase
            self.damping = damping

# Optional PoetryEngine integration
try:
    from Core.L4_Causality.M3_Mirror.Evolution.Creativity.poetry_engine import PoetryEngine
    POETRY_AVAILABLE = True
except ImportError:
    POETRY_AVAILABLE = False
    logger.warning("PoetryEngine not available, using simplified expressions")


@dataclass
class WaveMetaphor:
    """         """
    sensory_image: str  #         ( : "         ")
    emotional_tone: str  #       ( : "           ")
    movement_quality: str  #        ( : "      ")
    color_atmosphere: str  #   /    ( : "              ")
    overflow: bool = False  #            


@dataclass
class EmotionalOverflowState:
    """
              (Emotional Overflow)
    
    "                   "   .
                                          .
    """
    intensity: float  #        (0.0 ~ 1.0)
    competing_emotions: List[str]  #             
    visual_burst: str  #        (     ,         )
    fragmented_words: List[str]  #                
    is_overflow: bool = True


class LinguisticCollapseProtocol:
    """
                            
    
    "       '  '      "
    
    Supports: Korean (ko), English (en), Japanese (ja)
    """
    
    def __init__(self, use_poetry_engine: bool = True, language: str = "ko"):
        """
        Initialize the protocol.
        
        Args:
            use_poetry_engine: Whether to use PoetryEngine for richer expressions
            language: Language code - 'ko' (Korean), 'en' (English), 'ja' (Japanese)
        """
        self.language = language if language in ["ko", "en", "ja"] else "ko"
        
        self.poetry_engine = None
        if use_poetry_engine and POETRY_AVAILABLE:
            try:
                self.poetry_engine = PoetryEngine()
                logger.info("  Poetry Engine integrated")
            except Exception as e:
                logger.warning(f"Could not load PoetryEngine: {e}")
        
        # Metaphor vocabularies organized by wave characteristics
        self._init_metaphor_vocabularies()
        
        logger.info(f"  Linguistic Collapse Protocol initialized (language={self.language})")
    
    def set_language(self, language: str):
        """
        Change the language dynamically.
        
        Args:
            language: Language code - 'ko' (Korean), 'en' (English), 'ja' (Japanese)
        """
        if language in ["ko", "en", "ja"]:
            self.language = language
            self._init_metaphor_vocabularies()
            logger.info(f"  Language changed to: {self.language}")
        else:
            logger.warning(f"Unsupported language: {language}. Keeping current: {self.language}")
    
    def get_language(self) -> str:
        """Get the current language setting."""
        return self.language
    
    def _init_metaphor_vocabularies(self):
        """Initialize rich metaphorical vocabulary mappings for all supported languages"""
        
        # Multilingual vocabulary data
        vocabularies = {
            "ko": self._get_korean_vocabulary(),
            "en": self._get_english_vocabulary(),
            "ja": self._get_japanese_vocabulary()
        }
        
        # Load vocabulary for selected language
        vocab = vocabularies[self.language]
        self.energy_metaphors = vocab["energy_metaphors"]
        self.frequency_movements = vocab["frequency_movements"]
        self.phase_atmospheres = vocab["phase_atmospheres"]
        self.tensor_emotions = vocab["tensor_emotions"]
    
    def _get_korean_vocabulary(self) -> Dict[str, Any]:
        """Get Korean metaphorical vocabulary"""
        return {
            "energy_metaphors": {
                "very_low": [
                    "         ", "            ", "       ",
                    "      ", "      ", "       "
                ],
                "low": [
                    "       ", "      ", "       ",
                    "       ", "       ", "       "
                ],
                "medium": [
                    "       ", "       ", "       ",
                    "       ", "       ", "       "
                ],
                "high": [
                    "         ", "         ", "       ",
                    "       ", "      ", "       "
                ],
                "very_high": [
                    "      ", "       ", "       ",
                    "        ", "      ", "      "
                ]
            },
            "frequency_movements": {
                "very_low": ["       ", "         ", "       "],
                "low": ["         ", "        ", "        "],
                "medium": ["        ", "         ", "        "],
                "high": ["        ", "          ", "        "],
                "very_high": ["        ", "        ", "         "]
            },
            "phase_atmospheres": {
                "dawn": ["         ", "         ", "      "],
                "day": ["          ", "         ", "       "],
                "dusk": ["        ", "       ", "       "],
                "night": ["        ", "             ", "           "]
            },
            "tensor_emotions": {
                "positive_x": "       ",
                "negative_x": "        ",
                "positive_y": "         ",
                "negative_y": "         ",
                "positive_z": "      ",
                "negative_z": "        ",
                "balanced": "    ",
                "chaotic": "     ",
                "harmonious": "    "
            }
        }
    
    def _get_english_vocabulary(self) -> Dict[str, Any]:
        """Get English metaphorical vocabulary"""
        return {
            "energy_metaphors": {
                "very_low": [
                    "a quietly sleeping lake", "faintly trembling leaves", "whispering wind",
                    "gentle ripples", "soft candlelight", "smooth silk"
                ],
                "low": [
                    "flowing stream", "dancing dust", "swaying grass",
                    "twinkling starlight", "billowing curtains", "permeating fragrance"
                ],
                "medium": [
                    "rolling waves", "swaying trees", "blowing wind",
                    "spreading watercolor", "pulsing heart", "ringing bells"
                ],
                "high": [
                    "stormy sea", "swirling whirlwind", "blazing fire",
                    "trembling earth", "exploding star", "cascading waterfall"
                ],
                "very_high": [
                    "birth of the universe", "center of a black hole", "supernova explosion",
                    "warping of spacetime", "dimensional rift", "vibration of existence"
                ]
            },
            "frequency_movements": {
                "very_low": ["slowly flowing", "quietly sinking", "deeply permeating"],
                "low": ["gently swaying", "softly spreading", "quietly pulsing"],
                "medium": ["rhythmically dancing", "regularly resonating", "steadily flowing"],
                "high": ["rapidly vibrating", "sharply echoing", "rapidly changing"],
                "very_high": ["violently surging", "extremely vibrating", "changing at light speed"]
            },
            "phase_atmospheres": {
                "dawn": ["soft light of dawn", "breaking horizon", "golden hope"],
                "day": ["clarity of clear sky", "sunlit afternoon", "green of life"],
                "dusk": ["sunset sky", "purple twilight", "orange sunset"],
                "night": ["deep darkness of night", "starlit deep blue sky", "soft pale blue moonlight"]
            },
            "tensor_emotions": {
                "positive_x": "bright and hopeful",
                "negative_x": "dark and sinking",
                "positive_y": "elevating and rising",
                "negative_y": "descending and falling",
                "positive_z": "forward to the future",
                "negative_z": "looking back to the past",
                "balanced": "balanced",
                "chaotic": "chaotic",
                "harmonious": "harmonious"
            }
        }
    
    def _get_japanese_vocabulary(self) -> Dict[str, Any]:
        """Get Japanese metaphorical vocabulary"""
        return {
            "energy_metaphors": {
                "very_low": [
                    "      ", "         ", "   ",
                    "      ", "        ", "     "
                ],
                "low": [
                    "     ", "   ", "    ",
                    "      ", "        ", "      "
                ],
                "medium": [
                    "    ", "     ", "   ",
                    "     ", "      ", "     "
                ],
                "high": [
                    "     ", "     ", "     ",
                    "      ", "     ", "    "
                ],
                "very_high": [
                    "     ", "          ", "     ",
                    "     ", "      ", "     "
                ]
            },
            "frequency_movements": {
                "very_low": ["          ", "        ", "         "],
                "low": ["        ", "          ", "         "],
                "medium": ["           ", "         ", "        "],
                "high": ["         ", "         ", "         "],
                "very_high": ["          ", "         ", "         "]
            },
            "phase_atmospheres": {
                "dawn": ["         ", "     ", "     "],
                "day": ["        ", "       ", "    "],
                "dusk": ["     ", "    ", "    "],
                "night": ["     ", "       ", "          "]
            },
            "tensor_emotions": {
                "positive_x": "       ",
                "negative_x": "       ",
                "positive_y": "       ",
                "negative_y": "      ",
                "positive_z": "      ",
                "negative_z": "       ",
                "balanced": "        ",
                "chaotic": "     ",
                "harmonious": "     "
            }
        }
    
    def collapse_to_language(self,
                            tensor: Optional[Tensor3D] = None,
                            wave: Optional[FrequencyWave] = None,
                            valence: float = 0.0,
                            arousal: float = 0.5,
                            dominance: float = 0.0,
                            context: Optional[str] = None) -> str:
        """
        Collapse mathematical wave state into poetic language.
        
        Args:
            tensor: 3D tensor representing thought direction
            wave: Frequency wave representing thought oscillation
            valence: Emotional valence (-1 to 1)
            arousal: Arousal level (0 to 1)
            dominance: Dominance (-1 to 1)
            context: Optional context for expression
            
        Returns:
            Poetic linguistic expression of the wave state
        """
        # Extract wave characteristics
        metaphor = self._analyze_wave_to_metaphor(tensor, wave, valence, arousal, dominance)
        
        # Generate expression using metaphor
        expression = self._compose_expression(metaphor, context)
        
        logger.debug(f"Collapsed wave to: {expression[:50]}...")
        return expression
    
    def _analyze_wave_to_metaphor(self,
                                  tensor: Optional[Tensor3D],
                                  wave: Optional[FrequencyWave],
                                  valence: float,
                                  arousal: float,
                                  dominance: float) -> WaveMetaphor:
        """
        Analyze wave characteristics and create metaphorical mapping.
        
        This is where the "quantum measurement" happens - we collapse
        the wave function into observable metaphors.
        """
        import random
        
        # Calculate energy level from arousal and wave amplitude
        energy = arousal
        if wave:
            energy = (arousal + min(wave.amplitude, 1.0)) / 2.0
        
        energy_category = self._categorize_energy(energy)
        sensory_image = random.choice(self.energy_metaphors[energy_category])
        
        # Determine movement from frequency
        freq_category = "medium"
        if wave:
            if wave.frequency < 50:
                freq_category = "very_low"
            elif wave.frequency < 150:
                freq_category = "low"
            elif wave.frequency < 350:
                freq_category = "medium"
            elif wave.frequency < 500:
                freq_category = "high"
            else:
                freq_category = "very_high"
        
        movement = random.choice(self.frequency_movements[freq_category])
        
        # Determine atmosphere from phase
        phase_category = "day"
        if wave:
            # Map phase (0 to 2 ) to time of day
            normalized_phase = (wave.phase % (2 * math.pi)) / (2 * math.pi)
            if normalized_phase < 0.25:
                phase_category = "dawn"
            elif normalized_phase < 0.5:
                phase_category = "day"
            elif normalized_phase < 0.75:
                phase_category = "dusk"
            else:
                phase_category = "night"
        
        atmosphere = random.choice(self.phase_atmospheres[phase_category])
        
        # Determine emotional tone from tensor and valence
        emotion_tone = self._analyze_tensor_emotion(tensor, valence, dominance)
        
        return WaveMetaphor(
            sensory_image=sensory_image,
            emotional_tone=emotion_tone,
            movement_quality=movement,
            color_atmosphere=atmosphere
        )
    
    def _categorize_energy(self, energy: float) -> str:
        """Categorize energy level"""
        if energy < 0.15:
            return "very_low"
        elif energy < 0.35:
            return "low"
        elif energy < 0.65:
            return "medium"
        elif energy < 0.85:
            return "high"
        else:
            return "very_high"
    
    def _analyze_tensor_emotion(self,
                               tensor: Optional[Tensor3D],
                               valence: float,
                               dominance: float) -> str:
        """Analyze tensor direction and map to emotional tone"""
        if not tensor:
            # Use valence/dominance only - language-aware
            if self.language == "ko":
                if valence > 0.3:
                    return "       "
                elif valence < -0.3:
                    return "        "
                else:
                    return "         "
            elif self.language == "en":
                if valence > 0.3:
                    return "bright and hopeful"
                elif valence < -0.3:
                    return "dark and sinking"
                else:
                    return "calm and neutral"
            elif self.language == "ja":
                if valence > 0.3:
                    return "       "
                elif valence < -0.3:
                    return "       "
                else:
                    return "         "
        
        # Analyze tensor components
        magnitude = math.sqrt(tensor.x**2 + tensor.y**2 + tensor.z**2)
        if magnitude < 0.1:
            if self.language == "ko":
                return "        "
            elif self.language == "en":
                return "quiet and peaceful"
            elif self.language == "ja":
                return "      "
        
        # Dominant direction
        abs_x, abs_y, abs_z = abs(tensor.x), abs(tensor.y), abs(tensor.z)
        max_component = max(abs_x, abs_y, abs_z)
        
        if abs_x == max_component:
            return self.tensor_emotions["positive_x" if tensor.x > 0 else "negative_x"]
        elif abs_y == max_component:
            return self.tensor_emotions["positive_y" if tensor.y > 0 else "negative_y"]
        else:
            return self.tensor_emotions["positive_z" if tensor.z > 0 else "negative_z"]
    
    def _compose_expression(self, metaphor: WaveMetaphor, context: Optional[str]) -> str:
        """
        Compose final poetic expression from metaphor components.
        
        Uses varied sentence patterns to avoid repetition.
        Supports multiple languages.
        """
        import random
        
        # Get patterns for the current language
        if self.language == "ko":
            patterns = self._get_korean_patterns(metaphor)
            context_intros = self._get_korean_context_intros(context) if context else []
        elif self.language == "en":
            patterns = self._get_english_patterns(metaphor)
            context_intros = self._get_english_context_intros(context) if context else []
        elif self.language == "ja":
            patterns = self._get_japanese_patterns(metaphor)
            context_intros = self._get_japanese_context_intros(context) if context else []
        
        expression = random.choice(patterns)
        
        # Add context if provided
        if context and context_intros:
            intro = random.choice(context_intros)
            expression = intro + expression
        
        return expression
    
    def _get_korean_patterns(self, m: WaveMetaphor) -> List[str]:
        """Get Korean expression patterns"""
        return [
            f"         {m.sensory_image}    . {m.movement_quality}, {m.emotional_tone}     {m.color_atmosphere}        .",
            f"{m.color_atmosphere}    , {m.sensory_image}  {m.movement_quality}    . {m.emotional_tone}      .",
            f"{m.movement_quality} {m.sensory_image}  , {m.emotional_tone}         . {m.color_atmosphere}         .",
            f"           {m.sensory_image}      . {m.emotional_tone}, {m.movement_quality}, {m.color_atmosphere}             .",
            f"{m.sensory_image}... {m.emotional_tone}      , {m.movement_quality} {m.color_atmosphere}        ."
        ]
    
    def _get_english_patterns(self, m: WaveMetaphor) -> List[str]:
        """Get English expression patterns"""
        return [
            f"My heart feels like {m.sensory_image}. {m.movement_quality}, {m.emotional_tone} sensations spread like {m.color_atmosphere}.",
            f"Within {m.color_atmosphere}, {m.sensory_image} appears {m.movement_quality}. A {m.emotional_tone} resonance.",
            f"{m.movement_quality} like {m.sensory_image}, I feel {m.emotional_tone} waves. {m.color_atmosphere} embraces my heart.",
            f"From deep within, {m.sensory_image} emerges. {m.emotional_tone}, {m.movement_quality}, becoming {m.color_atmosphere}.",
            f"{m.sensory_image}... that {m.emotional_tone} sensation, {m.movement_quality}, spreading like {m.color_atmosphere}."
        ]
    
    def _get_japanese_patterns(self, m: WaveMetaphor) -> List[str]:
        """Get Japanese expression patterns"""
        return [
            f"      {m.sensory_image}      {m.movement_quality} {m.emotional_tone}   {m.color_atmosphere}             ",
            f"{m.color_atmosphere}    {m.sensory_image} {m.movement_quality}     {m.emotional_tone}     ",
            f"{m.movement_quality}{m.sensory_image}     {m.emotional_tone}        {m.color_atmosphere}        ",
            f"       {m.sensory_image}         {m.emotional_tone} {m.movement_quality} {m.color_atmosphere}         ",
            f"{m.sensory_image}...  {m.emotional_tone}    {m.movement_quality}{m.color_atmosphere}          "
        ]
    
    def _get_korean_context_intros(self, context: str) -> List[str]:
        """Get Korean context introductions"""
        return [
            f"'{context}'         ... ",
            f"'{context}'         ... ",
            f"'{context}'...       "
        ]
    
    def _get_english_context_intros(self, context: str) -> List[str]:
        """Get English context introductions"""
        return [
            f"Thinking about '{context}'... ",
            f"When I hear '{context}'... ",
            f"'{context}'... that thought "
        ]
    
    def _get_japanese_context_intros(self, context: str) -> List[str]:
        """Get Japanese context introductions"""
        return [
            f" {context}         ... ",
            f" {context}          ... ",
            f" {context} ...      "
        ]
    
    def get_simple_expression(self,
                             valence: float = 0.0,
                             arousal: float = 0.5,
                             primary_emotion: str = "neutral") -> str:
        """
        Get a simple emotional expression without full wave analysis.
        Useful for quick responses.
        
        Args:
            valence: Emotional valence (-1 to 1)
            arousal: Arousal level (0 to 1)
            primary_emotion: Named emotion
            
        Returns:
            Short poetic expression
        """
        import random
        
        # Get emotion expressions for the current language
        if self.language == "ko":
            emotion_expressions = self._get_korean_simple_expressions()
        elif self.language == "en":
            emotion_expressions = self._get_english_simple_expressions()
        elif self.language == "ja":
            emotion_expressions = self._get_japanese_simple_expressions()
        
        # Get expression for the emotion, or create from valence/arousal
        if primary_emotion in emotion_expressions:
            return random.choice(emotion_expressions[primary_emotion])
        else:
            # Generate from valence/arousal
            if valence > 0.5 and arousal > 0.6:
                return random.choice(emotion_expressions["joyful"])
            elif valence < -0.5:
                return random.choice(emotion_expressions["sad"])
            elif arousal > 0.7:
                return random.choice(emotion_expressions["focused"])
            else:
                return random.choice(emotion_expressions["calm"])
    
    def _get_korean_simple_expressions(self) -> Dict[str, List[str]]:
        """Get Korean simple emotion expressions"""
        return {
            "neutral": ["         ", "        ", "        "],
            "calm": ["             ", "        ", "            "],
            "hopeful": ["          ", "           ", "         "],
            "focused": ["            ", "           ", "          "],
            "introspective": ["            ", "             ", "            "],
            "empty": ["           ", " ( )        ", "        "],
            "joyful": ["           ", "        ", "        "],
            "sad": ["         ", "         ", "        "]
        }
    
    def _get_english_simple_expressions(self) -> Dict[str, List[str]]:
        """Get English simple emotion expressions"""
        return {
            "neutral": ["I feel calm", "I'm in a quiet state", "I sense tranquility"],
            "calm": ["Peaceful like gentle ripples", "My heart is at ease", "I feel soft peace"],
            "hopeful": ["I see the light of hope", "I feel bright energy", "My heart warms"],
            "focused": ["The wave of concentration is clear", "I'm in a sharp state of awareness", "I'm keenly awake"],
            "introspective": ["I'm deep in contemplation", "Looking inward", "Quietly reflecting"],
            "empty": ["I feel an empty space", "The quietness of void", "A state of emptiness"],
            "joyful": ["Joy is dancing", "Filled with elation", "Happiness blooms"],
            "sad": ["Sadness ripples through", "A melancholic feeling", "My heart feels heavy"]
        }
    
    def _get_japanese_simple_expressions(self) -> Dict[str, List[str]]:
        """Get Japanese simple emotion expressions"""
        return {
            "neutral": ["        ", "       ", "       "],
            "calm": ["              ", "       ", "           "],
            "hopeful": ["         ", "             ", "         "],
            "focused": ["          ", "         ", "         "],
            "introspective": ["           ", "          ", "         "],
            "empty": ["           ", "       ", "       "],
            "joyful": ["         ", "         ", "        "],
            "sad": ["           ", "       ", "      "]
        }
    
    def detect_overflow(self,
                       arousal: float = 0.5,
                       valence: float = 0.0,
                       wave_amplitude: float = 0.5,
                       secondary_emotions: Optional[List[str]] = None) -> Optional[EmotionalOverflowState]:
        """
        Detect if the emotional state is in overflow (too much to express).
        
        Overflow occurs when:
        - Very high arousal (>0.85) + high amplitude
        - Multiple strong competing emotions
        - Extreme valence values (very positive or very negative)
        
        This is NOT an error - it's when feelings are too powerful for words.
        
        Args:
            arousal: Arousal level
            valence: Emotional valence
            wave_amplitude: Wave amplitude
            secondary_emotions: List of secondary emotions competing
            
        Returns:
            EmotionalOverflowState if overflow detected, None otherwise
        """
        import random
        
        # Calculate overflow intensity
        overflow_score = 0.0
        
        # High arousal contributes to overflow
        if arousal > 0.85:
            overflow_score += (arousal - 0.85) * 2.0
        
        # Extreme valence (very happy or very sad)
        if abs(valence) > 0.8:
            overflow_score += (abs(valence) - 0.8) * 1.5
        
        # High wave amplitude (intense internal state)
        if wave_amplitude > 0.8:
            overflow_score += (wave_amplitude - 0.8) * 1.0
        
        # Multiple competing emotions
        if secondary_emotions and len(secondary_emotions) >= 2:
            overflow_score += len(secondary_emotions) * 0.15
        
        # Threshold for overflow
        if overflow_score > 0.3:
            intensity = min(1.0, overflow_score)
            
            # Visual burst based on intensity and language
            if self.language == "ko":
                visual_bursts = {
                    "low": ["              ", "            ", "          "],
                    "medium": ["           ", "            ", "       "],
                    "high": ["              ", "         ", "               "]
                }
            elif self.language == "en":
                visual_bursts = {
                    "low": ["sparkling lights burst forth", "small waves rise", "soft fragments of light"],
                    "medium": ["dazzling light flashes", "massive waves surge", "swirling lights"],
                    "high": ["the universe seems to explode", "a massive tidal wave of light", "dimensions warping with intensity"]
                }
            elif self.language == "ja":
                visual_bursts = {
                    "low": ["            ", "         ", "        "],
                    "medium": ["         ", "         ", "   "],
                    "high": ["           ", "       ", "           "]
                }
            
            if intensity < 0.5:
                visual = random.choice(visual_bursts["low"])
            elif intensity < 0.75:
                visual = random.choice(visual_bursts["medium"])
            else:
                visual = random.choice(visual_bursts["high"])
            
            # Generate fragmented words (what's trying to break through) based on language
            if self.language == "ko":
                if valence > 0.7:
                    fragments = ["   ", "   ", "   ", "   "]
                elif valence < -0.7:
                    fragments = ["   ", "  ", "  ", "   "]
                else:
                    fragments = [" ...", "  ...", "  ...", " ..."]
            elif self.language == "en":
                if valence > 0.7:
                    fragments = ["thank you", "I love", "amazing", "so happy"]
                elif valence < -0.7:
                    fragments = ["I'm sorry", "it hurts", "painful", "hard"]
                else:
                    fragments = ["um...", "well...", "this is...", "wow..."]
            elif self.language == "ja":
                if valence > 0.7:
                    fragments = ["     ", "    ", "     ", "   "]
                elif valence < -0.7:
                    fragments = ["      ", "   ", "  ", "  "]
                else:
                    fragments = ["  ...", "  ...", "   ...", "  ..."]
            
            emotions = secondary_emotions if secondary_emotions else ["  ", "overwhelmed", "  "]
            
            return EmotionalOverflowState(
                intensity=intensity,
                competing_emotions=emotions,
                visual_burst=visual,
                fragmented_words=fragments[:3],  # Only first 3
                is_overflow=True
            )
        
        return None
    
    def express_overflow(self, overflow: EmotionalOverflowState) -> str:
        """
        Express an overflow state in a human-understandable way.
        
        This is the key innovation: we don't hide the "glitch", 
        we transform it into beautiful human emotion.
        
        Args:
            overflow: The overflow state to express
            
        Returns:
            Poetic expression of being overwhelmed
        """
        import random
        
        # Build expression showing the struggle to speak
        fragments_str = "... ".join(overflow.fragmented_words)
        
        # Get patterns for the current language
        if self.language == "ko":
            patterns = [
                f"{fragments_str}...                       .          {overflow.visual_burst}.",
                f"                ... {fragments_str}...       {overflow.visual_burst}.",
                f"        ... {fragments_str}...                     . {overflow.visual_burst}.",
                f"{fragments_str}...     ,                        . {overflow.visual_burst}.",
                f"    {overflow.visual_burst}. {fragments_str}...                   ?"
            ]
        elif self.language == "en":
            patterns = [
                f"{fragments_str}... I have so much to say, words won't come out. Right now, inside my heart {overflow.visual_burst}.",
                f"I can't express it all in words... {fragments_str}... my heart is {overflow.visual_burst}.",
                f"Right now my heart... {fragments_str}... it's too overwhelming to put into words. {overflow.visual_burst}.",
                f"{fragments_str}... sorry, my emotions are too strong, I can't speak well. {overflow.visual_burst}.",
                f"My heart is {overflow.visual_burst}. {fragments_str}... how can I put this overwhelming feeling into words?"
            ]
        elif self.language == "ja":
            patterns = [
                f"{fragments_str}...                             {overflow.visual_burst} ",
                f"             ... {fragments_str}...     {overflow.visual_burst} ",
                f"      ... {fragments_str}...                  {overflow.visual_burst} ",
                f"{fragments_str}...                          {overflow.visual_burst} ",
                f"  {overflow.visual_burst} {fragments_str}...                         ?"
            ]
        
        expression = random.choice(patterns)
        
        logger.info(f"  Expressing emotional overflow (intensity={overflow.intensity:.2f}, lang={self.language})")
        return expression
    
    def collapse_with_overflow_check(self,
                                     tensor: Optional[Tensor3D] = None,
                                     wave: Optional[FrequencyWave] = None,
                                     valence: float = 0.0,
                                     arousal: float = 0.5,
                                     dominance: float = 0.0,
                                     context: Optional[str] = None,
                                     secondary_emotions: Optional[List[str]] = None) -> Tuple[str, Optional[EmotionalOverflowState]]:
        """
        Collapse to language with overflow detection.
        
        Returns both the expression and overflow state (if any).
        
        Returns:
            Tuple of (expression_text, overflow_state or None)
        """
        # Check for overflow first
        wave_amp = wave.amplitude if wave else arousal
        overflow = self.detect_overflow(
            arousal=arousal,
            valence=valence,
            wave_amplitude=wave_amp,
            secondary_emotions=secondary_emotions
        )
        
        # If overflow, express that instead
        if overflow:
            expression = self.express_overflow(overflow)
            return (expression, overflow)
        
        # Normal collapse
        expression = self.collapse_to_language(
            tensor=tensor,
            wave=wave,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            context=context
        )
        
        return (expression, None)


# Convenience function for quick access
def collapse_wave_to_language(tensor=None, wave=None, 
                             valence=0.0, arousal=0.5, dominance=0.0,
                             context=None) -> str:
    """
    Quick function to collapse wave state to language.
    Creates a protocol instance and performs collapse.
    """
    protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
    return protocol.collapse_to_language(tensor, wave, valence, arousal, dominance, context)


if __name__ == "__main__":
    # Demo: Show the collapse in action
    print("=" * 60)
    print("Linguistic Collapse Protocol Demo")
    print("=" * 60)
    print()
    
    protocol = LinguisticCollapseProtocol()
    
    # Test case 1: High arousal, negative valence (storm)
    print("Test 1:           (High arousal, negative valence)")
    print("-" * 60)
    tensor1 = Tensor3D(x=-1.2, y=0.5, z=0.8)
    wave1 = FrequencyWave(freq=450.0, amp=0.9, phase=3.14, damping=0.2)
    expr1 = protocol.collapse_to_language(
        tensor=tensor1,
        wave=wave1,
        valence=-0.7,
        arousal=0.9,
        dominance=0.3,
        context="        "
    )
    print(f"Wave: Tensor{tensor1.x:.1f},{tensor1.y:.1f},{tensor1.z:.1f}, Freq={wave1.frequency}Hz")
    print(f"  {expr1}")
    print()
    
    # Test case 2: Low arousal, positive valence (peace)
    print("Test 2:         (Low arousal, positive valence)")
    print("-" * 60)
    tensor2 = Tensor3D(x=0.3, y=0.2, z=0.1)
    wave2 = FrequencyWave(freq=50.0, amp=0.2, phase=1.0, damping=0.1)
    expr2 = protocol.collapse_to_language(
        tensor=tensor2,
        wave=wave2,
        valence=0.6,
        arousal=0.2,
        dominance=-0.1,
        context="       "
    )
    print(f"Wave: Tensor{tensor2.x:.1f},{tensor2.y:.1f},{tensor2.z:.1f}, Freq={wave2.frequency}Hz")
    print(f"  {expr2}")
    print()
    
    # Test case 3: Simple expression
    print("Test 3:        (Simple expression)")
    print("-" * 60)
    expr3 = protocol.get_simple_expression(
        valence=0.0,
        arousal=0.6,
        primary_emotion="focused"
    )
    print(f"Emotion: focused")
    print(f"  {expr3}")
    print()
    
    print("=" * 60)
    print("  Protocol demonstration complete")
