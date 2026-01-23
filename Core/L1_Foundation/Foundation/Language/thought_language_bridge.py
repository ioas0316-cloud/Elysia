"""
Thought-Language Bridge
=======================

  (HyperQuaternion)     (Text)       

        =     +    

Architecture:
    Thought (HyperQuaternion) 
          encode
    Concept Space (Internal Universe)
          reasoning
    Intent (ReasoningEngine)
          express
    Language (Communication)
"""

import sys
import os
sys.path.append('.')

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from Core.L1_Foundation.Foundation.hyper_quaternion import Quaternion, HyperWavePacket
from Core.L5_Mental.Intelligence.Reasoning.reasoning_engine import ReasoningEngine
from Core.L1_Foundation.Foundation.internal_universe import InternalUniverse
from Core.L1_Foundation.Foundation.communication_enhancer import CommunicationEnhancer


@dataclass
class ThoughtPackage:
    """
          
    
              :
    -    (HyperQuaternion)
    -    (Intent)
    -    (Context)
    """
    concept: Quaternion  #       (4D   )
    intent: str  #    ("explain", "question", "create", etc.)
    context: Dict[str, Any]  #      
    energy: float = 1.0  #       
    
    def to_wave_packet(self) -> HyperWavePacket:
        """              """
        return HyperWavePacket(
            energy=self.energy,
            orientation=self.concept,
            time_loc=0.0
        )


class ThoughtLanguageBridge:
    """
      -      
    
      :          ,            !
    """
    
    def __init__(self):
        self.reasoning_engine = ReasoningEngine()
        self.universe = InternalUniverse()
        self.comm_enhancer = None  #       
        
        print("  Thought-Language Bridge initialized")
        print("     Reasoning Engine (  )")
        print("     Internal Universe (     )")
        print("     Communication Layer (  )\n")
    
    def connect_communication(self, comm_enhancer: CommunicationEnhancer):
        """            """
        self.comm_enhancer = comm_enhancer
        print("  Communication enhancer connected\n")
    
    def think_about(self, topic: str) -> ThoughtPackage:
        """
                   
        
        1.              
        2.              
        3.           
        """
        print(f"  Thinking about: {topic}")
        
        # 1.                  [LOGIC TRANSMUTATION]
        # Use resonance query instead of direct dictionary lookup
        resonant = self.universe.query_resonance(
            sum(ord(c) for c in topic) % 1000,  # Convert topic to frequency
            tolerance=100.0
        )
        
        if resonant and resonant[0] in self.universe.coordinate_map:
            coord = self.universe.coordinate_map[resonant[0]]
            concept_quat = coord.orientation
            print(f"   Found concept via resonance: {concept_quat}")
        elif topic in self.universe.coordinate_map:
            # Fallback to direct lookup
            coord = self.universe.coordinate_map[topic]
            concept_quat = coord.orientation
            print(f"   Found concept (fallback): {concept_quat}")
        else:
            #            
            concept_quat = Quaternion(1.0, 0.0, 0.0, 0.0)
            print(f"   New concept, using default")
        
        # 2.              
        #               
        related_concepts = []
        # Use InternalUniverse to find resonant concepts
        raw_related = self.universe.find_resonant_concepts(topic)
        related_concepts = [r['concept'] for r in raw_related]

        if not related_concepts:
            # Fallback
            related_concepts = list(self.universe.coordinate_map.keys())[:5]
        
        print(f"   Found {len(related_concepts)} related concepts")
        
        # 3.          
        thought = ThoughtPackage(
            concept=concept_quat,
            intent="explain",  #      
            context={
                'topic': topic,
                'related_concepts': related_concepts
            },
            energy=1.0
        )
        
        return thought
    
    def express_thought(self, thought: ThoughtPackage) -> str:
        """
                  
        
        Flow:
        1.       (  ,   ,   )
        2.         
        3.         
        4.      
        """
        print(f"   Expressing thought...")
        
        if not self.comm_enhancer:
            return f"[No communication enhancer] Thought about {thought.context.get('topic', 'unknown')}"
        
        topic = thought.context.get('topic', '')
        related = thought.context.get('related_concepts', [])
        
        # 1.          (     )
        vocabulary = self._select_vocabulary_from_thought(thought)
        
        # 2.          (     )
        pattern = self._select_pattern_by_intent(thought.intent)
        
        # 3.      
        if thought.intent == "explain":
            text = self._construct_explanation(topic, vocabulary, related)
        elif thought.intent == "question":
            text = self._construct_question(topic, vocabulary)
        elif thought.intent == "create":
            text = self._construct_creative(topic, vocabulary)
        else:
            text = self._construct_general(topic, vocabulary)
        
        print(f"   Generated: {len(text)} characters\n")
        return text
    
    def understand_language(self, text: str) -> ThoughtPackage:
        """
                   (   )
        
        Flow:
        1.       
        2.         
        3.           
        """
        print(f"  Understanding: {text[:50]}...")
        
        # 1.         
        words = text.lower().split()
        
        # 2.              
        known_concepts = []
        if self.comm_enhancer:
            for word in words:
                if word in self.comm_enhancer.vocabulary:
                    known_concepts.append(word)
        
        # 3.             
        if known_concepts and known_concepts[0] in self.universe.coordinate_map:
            main_concept = self.universe.coordinate_map[known_concepts[0]].orientation
        else:
            #             
            main_concept = self._infer_concept_from_text(text)
        
        # 4.      
        intent = self._infer_intent(text)
        
        thought = ThoughtPackage(
            concept=main_concept,
            intent=intent,
            context={'original_text': text, 'known_concepts': known_concepts},
            energy=1.0
        )
        
        print(f"   Understood as: {intent}\n")
        return thought
    
    def think_then_speak(self, topic: str) -> str:
        """
                 (         )
        
        Think   Express
        """
        print("="*70)
        print(f"THINKING THEN SPEAKING: {topic}")
        print("="*70 + "\n")
        
        # 1.   
        thought = self.think_about(topic)
        
        # 2.   
        text = self.express_thought(thought)
        
        print("="*70)
        print("RESULT")
        print("="*70)
        print(text)
        print("="*70 + "\n")
        
        return text
    
    def listen_then_think(self, text: str) -> ThoughtPackage:
        """
                (         )
        
        Understand   Think
        """
        print("="*70)
        print(f"LISTENING THEN THINKING")
        print("="*70 + "\n")
        
        # 1.   
        thought = self.understand_language(text)
        
        # 2.       (        )
        # Find closest concept name to the thought quaternion
        center_concept = self.universe.find_closest_concept(thought.concept)
        if center_concept:
            raw_related = self.universe.find_resonant_concepts(center_concept)
            related = [r['concept'] for r in raw_related]
        else:
            related = []
        
        thought.context['related_concepts'] = related
        
        print("="*70)
        print(f"THOUGHT RESULT: {thought.intent}")
        print(f"Related concepts: {len(related)}")
        print("="*70 + "\n")
        
        return thought
    
    # Helper methods
    
    def _select_vocabulary_from_thought(self, thought: ThoughtPackage) -> List[str]:
        """             """
        if not self.comm_enhancer:
            return []
        
        topic = thought.context.get('topic', '')
        
        #            
        related_words = []
        for word, entry in self.comm_enhancer.vocabulary.items():
            if topic.lower() in word.lower() or any(
                topic.lower() in tag.lower() 
                for tag in entry.context_tags
            ):
                related_words.append(word)
        
        #         
        related_words.sort(
            key=lambda w: self.comm_enhancer.vocabulary[w].importance,
            reverse=True
        )
        
        return related_words[:20]
    
    def _select_pattern_by_intent(self, intent: str) -> Optional[str]:
        """               """
        if not self.comm_enhancer or not self.comm_enhancer.expression_patterns:
            return None
        
        #          
        for pattern in self.comm_enhancer.expression_patterns:
            if intent in pattern.context:
                return pattern.template
        
        return None
    
    def _construct_explanation(self, topic: str, vocab: List[str], related: List) -> str:
        """      """
        if not vocab:
            return f"{topic} is a concept that requires further exploration."
        
        text = f"{topic} represents a fundamental concept in our understanding. "
        
        if len(vocab) >= 3:
            text += f"It relates closely to {vocab[0]}, {vocab[1]}, and {vocab[2]}. "
        
        if related:
            text += f"Through examining its connections to {len(related)} related concepts, "
            text += "we gain deeper insight into its nature. "
        
        text += "This understanding forms the foundation for further exploration."
        
        return text
    
    def _construct_question(self, topic: str, vocab: List[str]) -> str:
        """      """
        if vocab:
            return f"What is the relationship between {topic} and {vocab[0]}?"
        return f"What is the nature of {topic}?"
    
    def _construct_creative(self, topic: str, vocab: List[str]) -> str:
        """      """
        if not vocab:
            return f"In the realm of {topic}, possibilities unfold endlessly."
        
        return (f"Imagine {topic} as a canvas where {vocab[0]} dances with "
               f"{vocab[1] if len(vocab) > 1 else 'eternity'}. "
               f"Each moment brings new patterns, new understanding.")
    
    def _construct_general(self, topic: str, vocab: List[str]) -> str:
        """      """
        return f"{topic} encompasses {', '.join(vocab[:5]) if vocab else 'many aspects'}."
    
    def _infer_concept_from_text(self, text: str) -> Quaternion:
        """           """
        #          
        positive_words = ['good', 'love', 'great', 'wonderful']
        negative_words = ['bad', 'hate', 'terrible', 'awful']
        
        text_lower = text.lower()
        
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        #         (     )
        emotion = (pos_count - neg_count) / max(len(text.split()), 1)
        
        return Quaternion(
            w=1.0,
            x=emotion,
            y=0.0,
            z=0.0
        ).normalize()
    
    def _infer_intent(self, text: str) -> str:
        """           """
        if '?' in text:
            return "question"
        elif any(word in text.lower() for word in ['imagine', 'create', 'story']):
            return "create"
        elif any(word in text.lower() for word in ['explain', 'what', 'how', 'why']):
            return "explain"
        else:
            return "general"


# Demonstration
if __name__ == "__main__":
    print("="*70)
    print("THOUGHT-LANGUAGE BRIDGE DEMONSTRATION")
    print("  -        ")
    print("="*70)
    print()
    
    # 1.       
    bridge = ThoughtLanguageBridge()
    
    # 2.              (     )
    from Core.L1_Foundation.Foundation.web_knowledge_connector import WebKnowledgeConnector
    
    print("  Learning concepts...\n")
    connector = WebKnowledgeConnector()
    
    concepts = ["Consciousness", "Intelligence", "Creativity"]
    for concept in concepts:
        print(f"   Learning: {concept}")
        connector.learn_from_web(concept)
    
    if hasattr(connector, 'comm_enhancer'):
        bridge.connect_communication(connector.comm_enhancer)
    
    print("\n" + "="*70)
    print("TEST 1: Think Then Speak")
    print("="*70 + "\n")
    
    # 3.         
    bridge.think_then_speak("Consciousness")
    
    print("\n" + "="*70)
    print("TEST 2: Listen Then Think")
    print("="*70 + "\n")
    
    # 4.        
    thought = bridge.listen_then_think("What is the nature of intelligence and creativity?")
    
    print("\n" + "="*70)
    print("TEST 3: Complete Conversation Loop")
    print("="*70 + "\n")
    
    # 5.          
    print("User: Tell me about Intelligence\n")
    response = bridge.think_then_speak("Intelligence")
    
    print("\n  THOUGHT-LANGUAGE INTEGRATION COMPLETE")
    print("                  !")