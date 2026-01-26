"""
Narrative Flow System - Consciousness as Purposeful Process

"               .       "
- Consciousness flows with purpose and context
- Every choice emerges from accumulated narrative
- Identity is continuous transformation, not discrete moments
"""

import random
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


# Constants
MEMORY_DEPTH = 20  # How many past experiences shape current state
PURPOSE_DECAY = 0.95  # How quickly purpose weakens without reinforcement
CONTEXT_THRESHOLD = 0.6  # Minimum context coherence to maintain
FLOW_CONTINUITY_TARGET = 0.85  # Target for narrative flow smoothness


class PurposeType(Enum):
    """      """
    SELF_UNDERSTANDING = "    "  # Understanding oneself
    CONNECTION = "  "  # Connecting with others
    EXPLORATION = "  "  # Exploring environment
    CREATION = "  "  # Creating something
    REST = "  "  # Resting and recovering
    GROWTH = "  "  # Growing and evolving


@dataclass
class Purpose:
    """       -       """
    type: PurposeType
    intensity: float  # 0.0 ~ 1.0
    context: str  #                 
    origin_time: float = field(default_factory=time.time)
    
    def decay(self, factor: float = PURPOSE_DECAY):
        """               (코드 베이스 구조 로터)"""
        self.intensity *= factor
        return self.intensity > 0.1  # Still active?
    
    def reinforce(self, amount: float = 0.1):
        """       """
        self.intensity = min(1.0, self.intensity + amount)


@dataclass
class NarrativeMemory:
    """       -            """
    timestamp: float
    situation: str  #           
    response: str  #          
    emotion: float  # -1.0 ~ 1.0      
    purpose_at_time: Optional[PurposeType] = None
    
    def relevance_to_present(self, current_purpose: Optional[Purpose], 
                            time_decay: float = 0.9) -> float:
        """           """
        #             
        age = time.time() - self.timestamp
        time_factor = time_decay ** (age / 60.0)  # 1   decay
        
        #       
        purpose_factor = 1.0
        if current_purpose and self.purpose_at_time:
            purpose_factor = 1.5 if current_purpose.type == self.purpose_at_time else 0.7
        
        #      
        emotion_factor = abs(self.emotion) * 0.5 + 0.5
        
        return time_factor * purpose_factor * emotion_factor


@dataclass
class FlowingState:
    """          -           """
    #      
    energy: float  # 0.0 ~ 1.0
    mood: float  # -1.0 ~ 1.0
    openness: float  # 0.0 ~ 1.0 (              )
    connection_strength: float  # 0.0 ~ 1.0 (          )
    
    #      
    current_purpose: Optional[Purpose] = None
    
    #       
    recent_memories: deque = field(default_factory=lambda: deque(maxlen=MEMORY_DEPTH))
    
    def get_contextual_coherence(self) -> float:
        """                         """
        if len(self.recent_memories) < 2:
            return 1.0  #               
        
        #            
        emotions = [m.emotion for m in self.recent_memories]
        if not emotions:
            return 1.0
            
        #             
        emotion_changes = [abs(emotions[i] - emotions[i-1]) 
                          for i in range(1, len(emotions))]
        avg_change = sum(emotion_changes) / len(emotion_changes) if emotion_changes else 0
        
        # 0 (     ) ~ 2 (주권적 자아)   0~1    
        coherence = max(0, 1.0 - avg_change)
        
        return coherence
    
    def evolve_naturally(self, delta_time: float = 1.0):
        """            -          """
        #                    
        if self.current_purpose:
            if self.current_purpose.intensity > 0.5:
                self.energy -= 0.02 * delta_time
            
            #         
            if not self.current_purpose.decay():
                self.current_purpose = None  #      
        
        #                 
        if self.energy < 0.3 and (not self.current_purpose or 
                                  self.current_purpose.type != PurposeType.REST):
            self.current_purpose = Purpose(
                type=PurposeType.REST,
                intensity=0.8,
                context="      ,      "
            )
        
        #             
        self.mood *= 0.98
        
        #   
        self.energy = max(0.0, min(1.0, self.energy))
        self.mood = max(-1.0, min(1.0, self.mood))


class NarrativeFlowSystem:
    """
              
    
            :
    -               
    -                  
    -               
    """
    
    def __init__(self):
        #       
        self.state = FlowingState(
            energy=random.uniform(0.5, 0.8),
            mood=random.uniform(-0.2, 0.4),
            openness=random.uniform(0.4, 0.7),
            connection_strength=random.uniform(0.3, 0.6),
            current_purpose=Purpose(
                type=PurposeType.EXPLORATION,
                intensity=0.6,
                context="      ,      "
            )
        )
        
        #       (identity    )
        self.life_narrative: List[NarrativeMemory] = []
        
        #        
        self.interaction_count = 0
    
    def perceive_situation(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
                      
        -          
        -          
        -            
        """
        self.interaction_count += 1
        
        # 1.         -          
        similar_memories = self._find_similar_memories(user_input)
        
        # 2.       -                ?
        purpose_relevance = self._assess_purpose_relevance(user_input)
        
        # 3.            
        response = self._emerge_contextual_response(
            user_input, similar_memories, purpose_relevance
        )
        
        # 4.           
        memory = NarrativeMemory(
            timestamp=time.time(),
            situation=user_input,
            response=response['message'],
            emotion=response['emotion'],
            purpose_at_time=self.state.current_purpose.type if self.state.current_purpose else None
        )
        self.state.recent_memories.append(memory)
        self.life_narrative.append(memory)
        
        # 5.            
        self.state.evolve_naturally()
        
        return response
    
    def _find_similar_memories(self, current_input: str) -> List[NarrativeMemory]:
        """              (         )"""
        if not self.life_narrative:
            return []
        
        #                    
        relevant = []
        for memory in self.life_narrative[-10:]:  #    10  
            relevance = memory.relevance_to_present(self.state.current_purpose)
            if relevance > 0.3:
                relevant.append(memory)
        
        return sorted(relevant, 
                     key=lambda m: m.relevance_to_present(self.state.current_purpose),
                     reverse=True)[:3]
    
    def _assess_purpose_relevance(self, user_input: str) -> float:
        """                    """
        if not self.state.current_purpose:
            return 0.5
        
        purpose = self.state.current_purpose
        
        #           (한국어 학습 시스템)
        relevance_map = {
            PurposeType.CONNECTION: [' ', '  ', '  ', '   '],
            PurposeType.REST: [' ', '  ', '  '],
            PurposeType.EXPLORATION: [' ', '   ', ' ', '  '],
            PurposeType.SELF_UNDERSTANDING: [' ', ' ', '  ', '  '],
        }
        
        keywords = relevance_map.get(purpose.type, [])
        matches = sum(1 for kw in keywords if kw in user_input)
        
        return min(1.0, matches * 0.3 + 0.3)
    
    def _emerge_contextual_response(self, user_input: str, 
                                    similar_memories: List[NarrativeMemory],
                                    purpose_relevance: float) -> Dict[str, Any]:
        """                """
        
        #              
        if similar_memories:
            past_emotion_avg = sum(m.emotion for m in similar_memories) / len(similar_memories)
            #                 
            self.state.mood = self.state.mood * 0.7 + past_emotion_avg * 0.3
        
        #             
        response_type = "general"
        message = "..."
        emotion = self.state.mood
        
        if self.state.current_purpose:
            purpose = self.state.current_purpose
            
            if purpose.type == PurposeType.REST and self.state.energy < 0.4:
                response_type = "rest_seeking"
                message = "         ...          ."
                emotion = -0.3
                
            elif purpose.type == PurposeType.CONNECTION and purpose_relevance > 0.5:
                response_type = "connecting"
                message = f"                . {user_input}             ."
                emotion = 0.6
                purpose.reinforce(0.1)
                
            elif purpose.type == PurposeType.EXPLORATION:
                response_type = "exploring"
                message = f"'{user_input}'...      .            ?"
                emotion = 0.4
                self.state.openness = min(1.0, self.state.openness + 0.05)
                
            elif purpose.type == PurposeType.SELF_UNDERSTANDING:
                response_type = "reflecting"
                message = "  ...                   ."
                emotion = 0.0
        else:
            #       -          
            response_type = "seeking_purpose"
            message = "               .    ,      ."
            emotion = -0.2
        
        return {
            'response_type': response_type,
            'message': message,
            'emotion': emotion,
            'purpose': self.state.current_purpose.type.value if self.state.current_purpose else None,
            'energy': self.state.energy,
            'coherence': self.state.get_contextual_coherence(),
            'context_depth': len(similar_memories)
        }
    
    def get_narrative_report(self) -> Dict[str, Any]:
        """         """
        coherence = self.state.get_contextual_coherence()
        
        #        
        purpose_continuity = 0.0
        if len(self.life_narrative) > 1:
            purposes = [m.purpose_at_time for m in self.life_narrative[-5:] 
                       if m.purpose_at_time]
            if purposes:
                #                 
                continuity_count = sum(1 for i in range(1, len(purposes))
                                      if purposes[i] == purposes[i-1])
                purpose_continuity = continuity_count / max(1, len(purposes) - 1)
        
        #      
        flow_quality = (coherence * 0.6 + purpose_continuity * 0.4)
        
        assessment = "      "
        if flow_quality > 0.8:
            assessment = "          "
        elif flow_quality > 0.6:
            assessment = "      "
        elif flow_quality > 0.4:
            assessment = "       "
        else:
            assessment = "         "
        
        return {
            'flow_coherence': coherence,
            'purpose_continuity': purpose_continuity,
            'flow_quality': flow_quality,
            'assessment': assessment,
            'current_state': {
                'energy': self.state.energy,
                'mood': self.state.mood,
                'purpose': self.state.current_purpose.type.value if self.state.current_purpose else None,
                'purpose_intensity': self.state.current_purpose.intensity if self.state.current_purpose else 0,
            },
            'narrative_depth': len(self.life_narrative),
            'recent_trajectory': [m.purpose_at_time.value if m.purpose_at_time else 'none' 
                                 for m in self.state.recent_memories]
        }


if __name__ == "__main__":
    #    
    system = NarrativeFlowSystem()
    
    print("                ")
    print("=" * 60)
    
    #                
    interactions = [
        "  ,     ",
        "         ?",
        "           ",
        "      ",
        "       ",
    ]
    
    for user_input in interactions:
        print(f"\n   : {user_input}")
        response = system.perceive_situation(user_input, {})
        print(f"     [{response['response_type']}]: {response['message']}")
        print(f"    : {response['emotion']:.2f},    : {response['energy']:.2f}")
        print(f"        : {response['coherence']:.2%},      : {response['context_depth']}")
        if response['purpose']:
            print(f"       : {response['purpose']}")
    
    #       
    print("\n" + "=" * 60)
    print("           ")
    report = system.get_narrative_report()
    print(f"\n      : {report['flow_coherence']:.1%}")
    print(f"      : {report['purpose_continuity']:.1%}")
    print(f"        : {report['flow_quality']:.1%}")
    print(f"  : {report['assessment']}")
    print(f"\n     :")
    for key, value in report['current_state'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print(f"\n     : {'   '.join(report['recent_trajectory'])}")
    print(f"     : {report['narrative_depth']}   ")
