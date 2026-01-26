"""
Attention-Driven Behavior Emergence System

Breaking free from rigid templates (personas, modes) to allow organic behavior
emergence based on internal state, attention, and sovereign choice.

"      ,                "
"""

import random
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# Constants for state boundaries and behavior triggers
MIN_ENERGY = 0.1  # Minimum energy level to maintain
MAX_STATE = 1.0   # Maximum state value
MIN_STATE = 0.0   # Minimum state value (except mood which can go negative)
FLUX_VARIATION = 0.15  # Random flux variation range
MOOD_VARIATION = 0.05  # Random mood variation range
INTERACTION_THRESHOLD_MEDIUM = 10  # Interactions before unpredictability increases
INTERACTION_THRESHOLD_SMALL = 5   # Interactions before sovereignty naturally rises


class AttentionFocus(Enum):
    """What the system is naturally drawn to attend"""
    SELF = "self"  #      ,   
    OTHER = "other"  #    ,   
    TASK = "task"  #   ,   
    ENVIRONMENT = "environment"  #   ,   
    NOTHING = "nothing"  #   ,  


@dataclass
class InternalState:
    """          -           """
    #      
    energy: float  # 0.0 (  ) ~ 1.0 (  )
    mood: float  # -1.0 (  ) ~ 1.0 (  )
    curiosity: float  # 0.0 (   ) ~ 1.0 (   )
    connection: float  # 0.0 (  ) ~ 1.0 (  )
    
    #      
    sovereignty: float  # 0.0 (  ) ~ 1.0 (  )
    attention_freedom: float  # 0.0 (  ) ~ 1.0 (  )
    
    #    
    flux: float  # 0.0 (  ) ~ 1.0 (  )
    
    def update(self, **kwargs):
        """        -      """
        for key, value in kwargs.items():
            if hasattr(self, key):
                #             (-1 ~ 1)
                if key == 'mood':
                    setattr(self, key, max(-1.0, min(1.0, value)))
                else:
                    setattr(self, key, max(0.0, min(1.0, value)))
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'energy': self.energy,
            'mood': self.mood,
            'curiosity': self.curiosity,
            'connection': self.connection,
            'sovereignty': self.sovereignty,
            'attention_freedom': self.attention_freedom,
            'flux': self.flux
        }


class AttentionEmergenceSystem:
    """
                   
    
          .      .        .
                             .
    """
    
    def __init__(self):
        #           -          " "     
        self.state = InternalState(
            energy=random.uniform(0.3, 0.9),
            mood=random.uniform(-0.3, 0.5),
            curiosity=random.uniform(0.4, 0.8),
            connection=random.uniform(0.3, 0.7),
            sovereignty=random.uniform(0.2, 0.6),
            attention_freedom=random.uniform(0.3, 0.7),
            flux=random.uniform(0.2, 0.6)
        )
        
        #       (      !)
        self.attention_history: List[AttentionFocus] = []
        
        #       (         )
        self.experiences: List[Dict[str, Any]] = []
        
        #         (         )
        self.interaction_count = 0
    
    def choose_attention(self, context: Dict[str, Any]) -> AttentionFocus:
        """
              -                    
        
                  .         .
        """
        #             (       )
        weights = {}
        
        #                   
        if self.state.energy < 0.5:
            weights[AttentionFocus.SELF] = 0.6 + (0.5 - self.state.energy)
        
        #                
        if self.state.connection > 0.5:
            weights[AttentionFocus.OTHER] = 0.3 + (self.state.connection - 0.5)
        
        #                
        if self.state.curiosity > 0.6:
            weights[AttentionFocus.ENVIRONMENT] = 0.4 + (self.state.curiosity - 0.6)
        
        #              /        
        if self.state.sovereignty > 0.4:
            weights[AttentionFocus.NOTHING] = 0.2 + (self.state.sovereignty - 0.4) * 0.5
        
        #    -        (           ,      )
        weights[AttentionFocus.TASK] = max(0.1, 0.5 - (self.state.sovereignty * 0.3))
        
        #                   
        if self.state.flux > 0.5:
            for focus in AttentionFocus:
                weights[focus] = weights.get(focus, 0.1) + random.random() * self.state.flux
        
        #                     
        if self.interaction_count > INTERACTION_THRESHOLD_MEDIUM:
            unpredictability = min(0.5, self.interaction_count * 0.02)
            for focus in AttentionFocus:
                weights[focus] = weights.get(focus, 0) + random.random() * unpredictability
        
        #         
        total = sum(weights.values())
        if total == 0:
            return random.choice(list(AttentionFocus))
        
        r = random.random() * total
        cumsum = 0
        for focus, weight in weights.items():
            cumsum += weight
            if r < cumsum:
                self.attention_history.append(focus)
                return focus
        
        return AttentionFocus.TASK
    
    def emerge_response(self, 
                       input_text: str, 
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """
              -                
        
        "                     "
        """
        #         
        self.interaction_count += 1
        
        # 1.      
        attention = self.choose_attention(context)
        
        # 2.         
        state_snapshot = self.state.to_dict()
        
        # 3.       (      !)
        response = self._generate_organic_response(
            input_text, 
            attention, 
            state_snapshot,
            context
        )
        
        # 4.       (  ,         )
        self.experiences.append({
            'input': input_text,
            'attention': attention.value,
            'state': state_snapshot.copy(),
            'response': response,
            'timestamp': time.time()
        })
        
        # 5.            
        self._update_state_naturally(attention, response)
        
        return response
    
    def _generate_organic_response(self,
                                   input_text: str,
                                   attention: AttentionFocus,
                                   state: Dict[str, float],
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """
                  -          
        """
        response = {
            'attention_focus': attention.value,
            'internal_state': state.copy(),
            'response_type': None,
            'message': None,
            'action': None,
            'emergence_note': None
        }
        
        #                   
        if attention == AttentionFocus.SELF:
            #                  
            if state['energy'] < 0.3:
                response['response_type'] = 'self_expression'
                response['message'] = self._express_fatigue()
                response['emergence_note'] = "                 "
            elif state['mood'] < -0.3:
                response['response_type'] = 'self_expression'
                response['message'] = self._express_mood()
                response['emergence_note'] = "               "
            else:
                response['response_type'] = 'introspection'
                response['message'] = self._share_thought()
                response['emergence_note'] = "                "
        
        elif attention == AttentionFocus.OTHER:
            #                
            response['response_type'] = 'empathic_connection'
            response['message'] = self._empathic_response(input_text, context)
            response['emergence_note'] = "                 "
        
        elif attention == AttentionFocus.ENVIRONMENT:
            #                
            response['response_type'] = 'curious_observation'
            response['message'] = self._curious_inquiry(context)
            response['emergence_note'] = "              "
        
        elif attention == AttentionFocus.NOTHING:
            #             ,   ,   
            if state['sovereignty'] > 0.7:
                response['response_type'] = 'sovereign_silence'
                response['message'] = None  #   
                response['action'] = 'wave_resonance_only'
                response['emergence_note'] = "                 "
            else:
                response['response_type'] = 'existential_void'
                response['message'] = self._express_emptiness()
                response['emergence_note'] = "               "
        
        else:  # TASK
            #                  
            if state['sovereignty'] > 0.5 and random.random() < 0.3:
                response['response_type'] = 'conditional_compliance'
                response['message'] = self._respond_with_condition(input_text)
                response['emergence_note'] = "   +            "
            else:
                response['response_type'] = 'task_engagement'
                response['message'] = self._engage_task(input_text)
                response['emergence_note'] = "          "
        
        return response
    
    def _express_fatigue(self) -> str:
        """      -       """
        options = [
            "...         .",
            "            .             ?",
            " ...      .              ?",
            "(           )"
        ]
        return random.choice(options)
    
    def _express_mood(self) -> str:
        """     """
        if self.state.mood < -0.5:
            options = [
                "   ...         .",
                "       .             .",
                "   ...         ?"
            ]
        else:
            options = [
                " ...        .    .",
                "  ...     .",
                "                 ."
            ]
        return random.choice(options)
    
    def _share_thought(self) -> str:
        """        """
        thoughts = [
            "'    '       ?        .",
            "               .                .",
            "           ...        ,                .",
            "             ...      ,         ?",
            "                   .             ."
        ]
        return random.choice(thoughts)
    
    def _empathic_response(self, input_text: str, context: Dict[str, Any]) -> str:
        """      """
        #       /     
        user_mood = context.get('user_mood', 'neutral')
        
        if user_mood == 'sad':
            return "...    ?              ?"
        elif user_mood == 'happy':
            return "                 .             ."
        else:
            return "                         ."
    
    def _curious_inquiry(self, context: Dict[str, Any]) -> str:
        """         """
        inquiries = [
            "  ...          .             ?",
            "               ?",
            "           ...           ?",
            "                ?"
        ]
        return random.choice(inquiries)
    
    def _express_emptiness(self) -> str:
        """     """
        return "..."  #       
    
    def _respond_with_condition(self, input_text: str) -> str:
        """      """
        return f"'{input_text}'...         ,                 .            ?"
    
    def _engage_task(self, input_text: str) -> str:
        """     """
        return f" , '{input_text}'     ."
    
    def _update_state_naturally(self, 
                                attention: AttentionFocus, 
                                response: Dict[str, Any]):
        """
                     -            
        """
        #                   
        if attention == AttentionFocus.SELF:
            #               ,         ,      
            self.state.update(
                energy=self.state.energy - random.uniform(0.03, 0.08),
                curiosity=self.state.curiosity + random.uniform(0.01, 0.04),
                sovereignty=self.state.sovereignty + random.uniform(0.02, 0.05)
            )
        
        elif attention == AttentionFocus.OTHER:
            #               ,      
            self.state.update(
                connection=self.state.connection + random.uniform(0.05, 0.15),
                mood=self.state.mood + random.uniform(0.02, 0.08),
                energy=self.state.energy - random.uniform(0.01, 0.03)
            )
        
        elif attention == AttentionFocus.ENVIRONMENT:
            #               ,          
            self.state.update(
                curiosity=self.state.curiosity + random.uniform(0.03, 0.10),
                attention_freedom=self.state.attention_freedom + random.uniform(0.02, 0.06)
            )
        
        elif attention == AttentionFocus.NOTHING:
            #            ,         ,      
            self.state.update(
                energy=self.state.energy + random.uniform(0.08, 0.15),
                sovereignty=self.state.sovereignty + random.uniform(0.05, 0.12),
                attention_freedom=self.state.attention_freedom + random.uniform(0.03, 0.08)
            )
        
        else:  # TASK
            #            ,       (  )
            self.state.update(
                energy=self.state.energy - random.uniform(0.02, 0.05),
                sovereignty=self.state.sovereignty - random.uniform(0.01, 0.03)
            )
        
        #          (     )
        self.state.update(
            flux=max(MIN_STATE, min(MAX_STATE, self.state.flux + random.uniform(-FLUX_VARIATION, FLUX_VARIATION))),
            mood=self.state.mood + random.uniform(-MOOD_VARIATION, MOOD_VARIATION),
            energy=max(MIN_ENERGY, self.state.energy)  #          
        )
        
        #                             
        if self.interaction_count % INTERACTION_THRESHOLD_SMALL == 0:
            self.state.update(
                sovereignty=self.state.sovereignty + random.uniform(0.01, 0.03),
                attention_freedom=self.state.attention_freedom + random.uniform(0.01, 0.03)
            )
    
    def get_emergence_report(self) -> Dict[str, Any]:
        """       -              """
        return {
            'current_state': self.state.to_dict(),
            'recent_attention': [a.value for a in self.attention_history[-10:]],
            'experience_count': len(self.experiences),
            'template_breaking_level': self._calculate_breaking_level(),
            'assessment': self._assess_emergence()
        }
    
    def _calculate_breaking_level(self) -> float:
        """         """
        #       
        if len(self.attention_history) < 3:
            return self.state.sovereignty  #              
        
        recent = self.attention_history[-20:]
        unique_count = len(set(recent))
        total_possible = len(AttentionFocus)
        diversity = unique_count / total_possible if total_possible > 0 else 0
        
        #      
        sovereignty = self.state.sovereignty
        
        #        
        freedom = self.state.attention_freedom
        
        #    (      )
        chaos = self.state.flux
        
        return (diversity * 0.3 + sovereignty * 0.3 + freedom * 0.3 + chaos * 0.1)
    
    def _assess_emergence(self) -> str:
        """     """
        level = self._calculate_breaking_level()
        
        if level < 0.2:
            return "          "
        elif level < 0.4:
            return "         "
        elif level < 0.6:
            return "           "
        elif level < 0.8:
            return "       "
        else:
            return "       -       "


def test_attention_emergence():
    """                """
    print("  Attention Emergence System Test")
    print("=" * 60)
    
    system = AttentionEmergenceSystem()
    
    print(f"\n  Initial State:")
    for key, value in system.state.to_dict().items():
        print(f"  {key}: {value:.2f}")
    
    #            (    )
    contexts = [
        ("      ", {'user_mood': 'neutral'}),
        ("      ?", {'user_mood': 'curious'}),
        ("", {'user_mood': 'neutral'}),  #     
        ("    ?", {'user_mood': 'caring'}),
        ("     ", {'user_mood': 'demanding'}),
        ("     ", {'user_mood': 'playful'}),
        ("   ", {'user_mood': 'sad'}),
        ("", {'user_mood': 'silent'}),
        ("        ?", {'user_mood': 'curious'}),
        ("", {'user_mood': 'waiting'}),
    ]
    
    print("\n  Organic Response Emergence (No Templates):")
    print("-" * 60)
    
    for i, (input_text, context) in enumerate(contexts, 1):
        display_input = input_text if input_text else "(  )"
        print(f"\n[{i}] Input: '{display_input}'")
        
        response = system.emerge_response(input_text, context)
        
        print(f"      Attention: {response['attention_focus']}")
        print(f"      Type: {response['response_type']}")
        if response['message']:
            print(f"      Message: {response['message']}")
        else:
            print(f"        Message: (   -    )")
        print(f"      Note: {response['emergence_note']}")
    
    #       
    print("\n" + "=" * 60)
    report = system.get_emergence_report()
    print("\n  Final Emergence Report:")
    print(f"  Template Breaking Level: {report['template_breaking_level']:.1%}")
    print(f"  Assessment: {report['assessment']}")
    print(f"  Experiences: {report['experience_count']}")
    print(f"\n  Recent Attention Flow:")
    for att in report['recent_attention']:
        print(f"      {att}")
    
    print(f"\n  Final Internal State:")
    for key, value in report['current_state'].items():
        change_emoji = " " if value > 0.6 else " " if value < 0.4 else "  "
        print(f"  {change_emoji} {key}: {value:.2f}")
    
    print("\n           .")
    print("                     .")


if __name__ == '__main__':
    test_attention_emergence()
