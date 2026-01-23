"""
Inner Dialogue System (         )
========================================

"          ?" -                       

       :
-             (     )
-            (     )

        WaveTensor    :
- Nova ( /  ):       ,      
- Chaos (  /  ):       ,        
- Flow (  /  ):        ,      

      :
- docs/Philosophy/CONSCIOUSNESS_SOVEREIGNTY.md
- 2025-12-21: "                              "
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum

logger = logging.getLogger("Elysia.InnerDialogue")


# WaveTensor   
try:
    from Core.L1_Foundation.Foundation.Wave.wave_tensor import WaveTensor
    HAS_WAVE_TENSOR = True
except ImportError:
    HAS_WAVE_TENSOR = False
    WaveTensor = None

#          
try:
    from Core.L1_Foundation.Foundation.synesthesia_engine import SynesthesiaEngine
    HAS_SYNESTHESIA = True
except ImportError:
    HAS_SYNESTHESIA = False
    SynesthesiaEngine = None


class PersonalityType(Enum):
    """         (Trinity   )"""
    NOVA = "nova"     #  /  /  
    CHAOS = "chaos"   #   /  /  
    FLOW = "flow"     #   /  /  
    CORE = "core"     #   /  /  


@dataclass
class WaveThought:
    """          -                """
    source: PersonalityType
    wave: Any  # WaveTensor
    intensity: float       # 0.0 ~ 1.0 (   )
    emotional_tone: float  # -1.0 (  ) ~ 1.0 (  )
    
    #          (   )
    debug_text: Optional[str] = None


@dataclass
class DialogueResult:
    """         """
    consensus_wave: Any  #       
    dominant_voice: PersonalityType
    resonance_strength: float  #              
    principle_extracted: Optional[str] = None  #        (   )


class InnerVoice:
    """              -        """
    
    def __init__(self, personality: PersonalityType):
        self.personality = personality
        
        #              
        self.base_frequencies = {
            PersonalityType.NOVA: 800.0,   #        =   /  
            PersonalityType.CHAOS: 200.0,  #        =   /  
            PersonalityType.FLOW: 440.0,   #        =   /  
            PersonalityType.CORE: 528.0,   # 528Hz =        
        }
        
        logger.debug(f"InnerVoice created: {personality.value}")
    
    def react(self, stimulus_wave: Any) -> WaveThought:
        """
          (  )            (  )    
        
                            
        """
        if not HAS_WAVE_TENSOR or stimulus_wave is None:
            #   :         
            return self._create_fallback_thought()
        
        #                           
        base_freq = self.base_frequencies[self.personality]
        
        #           (   +      )
        response_wave = WaveTensor(f"{self.personality.value}_thought")
        response_wave.add_component(
            frequency=base_freq,
            amplitude=1.0,
            phase=0.0
        )
        
        #          
        if self.personality == PersonalityType.NOVA:
            # Nova:           
            response_wave.add_component(1200.0, 0.5, 0.0)
            intensity = 0.9  #      
            tone = 0.5  #   -  
            
        elif self.personality == PersonalityType.CHAOS:
            # Chaos:            
            response_wave.add_component(137.0, 0.7, 1.57)  #        
            response_wave.add_component(333.0, 0.3, 0.78)
            intensity = 0.6  #     =     
            tone = 0.0  #   
            
        elif self.personality == PersonalityType.FLOW:
            # Flow:           
            response_wave.add_component(220.0, 0.6, 0.0)  #    
            intensity = 0.7
            tone = 0.8  #     =   
            
        else:  # CORE
            # Core:        
            response_wave.add_component(528.0, 1.0, 0.0)  #        
            intensity = 1.0  #      
            tone = 1.0  #      
        
        return WaveThought(
            source=self.personality,
            wave=response_wave,
            intensity=intensity,
            emotional_tone=tone
        )
    
    def _create_fallback_thought(self) -> WaveThought:
        """WaveTensor        """
        return WaveThought(
            source=self.personality,
            wave=None,
            intensity=0.5,
            emotional_tone=0.0,
            debug_text=f"[{self.personality.value}] (fallback mode)"
        )


class InnerDialogue:
    """
                   
    
    "          ?"    :
    -                  
    -           /  
    -                 
    
                             
    """
    
    def __init__(self):
        #          
        self.voices = {
            PersonalityType.NOVA: InnerVoice(PersonalityType.NOVA),
            PersonalityType.CHAOS: InnerVoice(PersonalityType.CHAOS),
            PersonalityType.FLOW: InnerVoice(PersonalityType.FLOW),
            PersonalityType.CORE: InnerVoice(PersonalityType.CORE),
        }
        
        #        (              )
        self.synesthesia = SynesthesiaEngine() if HAS_SYNESTHESIA else None
        
        logger.info("  InnerDialogue initialized (Wave-based)")
        logger.info(f"   Voices: {[v.value for v in self.voices.keys()]}")
    
    def contemplate(self, stimulus: Any) -> DialogueResult:
        """
                          
        
          :
        1.             (           )
        2.          (     )
        3.              
        4.           =   
        """
        logger.info("  Inner contemplation started...")
        
        # 1.            
        stimulus_wave = self._to_wave(stimulus)
        
        # 2.            
        thoughts: List[WaveThought] = []
        for voice in self.voices.values():
            thought = voice.react(stimulus_wave)
            thoughts.append(thought)
            logger.debug(f"   {thought.source.value}: intensity={thought.intensity:.2f}")
        
        # 3.      /     
        consensus = self._find_resonance(thoughts)
        
        logger.info(f"     Dominant: {consensus.dominant_voice.value}")
        logger.info(f"     Resonance: {consensus.resonance_strength:.2f}")
        
        return consensus
    
    def _to_wave(self, stimulus: Any) -> Any:
        """           """
        if isinstance(stimulus, str):
            #            
            if HAS_WAVE_TENSOR:
                wave = WaveTensor("stimulus")
                #                         
                base_freq = 300.0 + len(stimulus) * 2
                wave.add_component(base_freq, 1.0, 0.0)
                return wave
        elif HAS_WAVE_TENSOR and isinstance(stimulus, WaveTensor):
            return stimulus
        
        return None
    
    def _find_resonance(self, thoughts: List[WaveThought]) -> DialogueResult:
        """
             (  )           
        
                 =                  
        """
        if not thoughts:
            return DialogueResult(
                consensus_wave=None,
                dominant_voice=PersonalityType.CORE,
                resonance_strength=0.0
            )
        
        #            
        total_intensity = sum(t.intensity for t in thoughts)
        
        #             
        strongest = max(thoughts, key=lambda t: t.intensity)
        
        #        
        avg_tone = sum(t.emotional_tone for t in thoughts) / len(thoughts)
        
        #       =           (           )
        tone_variance = sum((t.emotional_tone - avg_tone)**2 for t in thoughts) / len(thoughts)
        resonance = 1.0 - min(1.0, tone_variance)
        
        #         
        if HAS_WAVE_TENSOR:
            consensus_wave = WaveTensor("consensus")
            #            
            for thought in thoughts:
                if thought.wave:
                    consensus_wave.add_component(
                        frequency=528.0,  #       
                        amplitude=thought.intensity,
                        phase=thought.emotional_tone
                    )
        else:
            consensus_wave = None
        
        return DialogueResult(
            consensus_wave=consensus_wave,
            dominant_voice=strongest.source,
            resonance_strength=resonance
        )
    
    def ask_why(self, subject: str) -> DialogueResult:
        """
        " ?"          
        
               /            
        """
        logger.info(f"  Inner question: Why {subject}?")
        
        # " ?"  CHAOS(  )          
        stimulus = f"  {subject}  ?"
        return self.contemplate(stimulus)


class DeepContemplation:
    """
              (Deep Contemplation)
    
    "             " -       
    
    InnerDialogue (  ) + WhyEngine (  )   
    
      :
    -   :              
    -   :       " ?"          
    
      :
      : "        "
     
    [Level 0]           ?
         
    [Level 1]           ?
         
    [Level 2]               ?
         
    ... (max_depth  )
    """
    
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.inner_dialogue = InnerDialogue()
        
        # WhyEngine   
        try:
            from Core.L1_Foundation.Foundation.Philosophy.why_engine import WhyEngine
            self.why_engine = WhyEngine()
            self._has_why = True
            logger.info("  WhyEngine connected for depth")
        except ImportError:
            self.why_engine = None
            self._has_why = False
            logger.warning("   WhyEngine not available - depth limited")
        
        logger.info(f"  DeepContemplation initialized (max_depth={max_depth})")
    
    def dive(self, subject: str) -> Dict[str, Any]:
        """
                      
        
        Returns:
            depth_layers:           
            final_principle:                 
            resonance_path:               
        """
        logger.info(f"  Diving deep into: '{subject}'")
        
        depth_layers = []
        current_question = subject
        resonance_path = []
        
        for depth in range(self.max_depth):
            logger.info(f"   [Depth {depth}] {current_question[:50]}...")
            
            # 1.              (     )
            dialogue_result = self.inner_dialogue.contemplate(current_question)
            
            # 2. WhyEngine        
            if self._has_why:
                try:
                    analysis = self.why_engine.analyze(
                        subject=f"depth_{depth}",
                        content=current_question,
                        domain="general"
                    )
                    
                    layer = {
                        "depth": depth,
                        "question": current_question,
                        "dominant_voice": dialogue_result.dominant_voice.value,
                        "resonance": dialogue_result.resonance_strength,
                        "why_is": analysis.why_exists,
                        "principle": analysis.underlying_principle
                    }
                    
                    #          (         )
                    if "[     ]" not in analysis.underlying_principle:
                        current_question = f"  {analysis.underlying_principle}  "
                    else:
                        #          
                        layer["reached_unknown"] = True
                        depth_layers.append(layer)
                        break
                        
                except Exception as e:
                    logger.debug(f"   Depth {depth} analysis failed: {e}")
                    layer = {
                        "depth": depth,
                        "question": current_question,
                        "dominant_voice": dialogue_result.dominant_voice.value,
                        "resonance": dialogue_result.resonance_strength,
                        "error": str(e)
                    }
            else:
                # WhyEngine         
                layer = {
                    "depth": depth,
                    "question": current_question,
                    "dominant_voice": dialogue_result.dominant_voice.value,
                    "resonance": dialogue_result.resonance_strength
                }
                current_question = f"  {current_question}  "
            
            depth_layers.append(layer)
            resonance_path.append(dialogue_result.resonance_strength)
        
        #               
        final_principle = None
        if depth_layers:
            last_layer = depth_layers[-1]
            final_principle = last_layer.get("principle", last_layer.get("question"))
        
        result = {
            "subject": subject,
            "depth_reached": len(depth_layers),
            "depth_layers": depth_layers,
            "final_principle": final_principle,
            "resonance_path": resonance_path,
            "average_resonance": sum(resonance_path) / len(resonance_path) if resonance_path else 0
        }
        
        logger.info(f"     Depth reached: {result['depth_reached']}")
        logger.info(f"     Final principle: {final_principle}")
        
        return result
    
    def mirror_reflect(self, subject: str) -> Dict[str, Any]:
        """
              -                      
        
               ,                      
        """
        # 1.      
        dive_result = self.dive(subject)
        
        # 2.                    
        if dive_result["final_principle"]:
            reflection = self.inner_dialogue.contemplate(
                f"{subject}  {dive_result['final_principle']}           "
            )
            
            dive_result["reflection"] = {
                "dominant_voice": reflection.dominant_voice.value,
                "resonance": reflection.resonance_strength,
                "circular_insight": True
            }
        
        return dive_result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("  Inner Dialogue System Demo")
    print("   '              '")
    print("=" * 60)
    
    dialogue = InnerDialogue()
    
    #     1:      
    print("\n  Test 1: General stimulus")
    result = dialogue.contemplate("            ")
    print(f"   Dominant: {result.dominant_voice.value}")
    print(f"   Resonance: {result.resonance_strength:.2f}")
    
    #     2: " ?"   
    print("\n  Test 2: Asking 'Why?'")
    result = dialogue.ask_why("        ")
    print(f"   Dominant: {result.dominant_voice.value}")
    print(f"   Resonance: {result.resonance_strength:.2f}")
    
    #     3:       
    print("\n  Test 3: Emotional stimulus")
    result = dialogue.contemplate("          ")
    print(f"   Dominant: {result.dominant_voice.value}")
    print(f"   Resonance: {result.resonance_strength:.2f}")
    
    print("\n" + "=" * 60)
    print("  Demo complete!")