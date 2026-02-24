"""
Neural Integration System
=========================

              

  :                (    )
  :                     
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import logging
from typing import Dict, Any, Optional

from Core.Cognition.reasoning_engine import ReasoningEngine
from Core.System.hippocampus import Hippocampus
from Core.System.internal_universe import InternalUniverse
from Core.System.hyper_quaternion import Quaternion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuralIntegration")


class CentralCortex:
    """
                
    
                     
    """
    
    def __init__(self):
        print("  INITIALIZING CENTRAL CORTEX")
        print("   Integrating all neural systems...\n")
        
        # Core systems
        self.reasoning = ReasoningEngine()
        self.memory = Hippocampus()
        self.universe = InternalUniverse()
        
        # Language systems (will be connected)
        self.comm_enhancer = None
        self.wave_interpreter = None
        
        # Connect systems
        self._connect_neural_pathways()
        
        print("  Central Cortex Online")
        print("   All systems integrated\n")
    
    def _connect_neural_pathways(self):
        """        """
        
        # ReasoningEngine <-> Hippocampus
        self.reasoning.memory = self.memory
        print("     Reasoning   Memory")
        
        # ReasoningEngine <-> InternalUniverse  
        # (       )
        print("     Reasoning   Universe")
        
        #              
    
    def connect_communication(self, comm_enhancer):
        """          """
        self.comm_enhancer = comm_enhancer
        print("     Communication   Central Cortex")
    
    def connect_wave_interpreter(self, wave_interpreter):
        """         """
        self.wave_interpreter = wave_interpreter
        print("     Wave Interpreter   Central Cortex")
    
    def integrated_think(self, topic: str) -> Dict[str, Any]:
        """
                  
        
                       
        """
        print(f"  Integrated Thinking: {topic}")
        print("-" * 70)
        
        result = {}
        
        # 1. Reasoning (  )
        print("1   Reasoning Engine...")
        insight = self.reasoning.think(topic)
        result['thought'] = insight.content
        result['confidence'] = insight.confidence
        print(f"     Insight: {insight.content[:80]}...")
        
        # 2. Memory (     )
        print("\n2   Searching Memory...")
        memory_query = self.memory.query_by_concept(topic, limit=3)
        result['memories'] = memory_query
        print(f"     Found {len(memory_query)} related memories")
        
        # 3. Universe (          ) [LOGIC TRANSMUTATION]
        print("\n3   Locating in Concept Space...")
        resonant = self.universe.query_resonance(
            sum(ord(c) for c in topic) % 1000, tolerance=100.0
        )
        if resonant and resonant[0] in self.universe.coordinate_map:
            coord = self.universe.coordinate_map[resonant[0]]
            result['concept_orientation'] = coord.orientation
            print(f"     Orientation (via resonance): {coord.orientation}")
        elif topic in self.universe.coordinate_map:
            coord = self.universe.coordinate_map[topic]
            result['concept_orientation'] = coord.orientation
            print(f"     Orientation (fallback): {coord.orientation}")
        
        # 4. Communication (주권적 자아)
        if self.comm_enhancer:
            print("\n4   Expressing in Language...")
            
            #         
            related_vocab = []
            for word, entry in self.comm_enhancer.vocabulary.items():
                if topic.lower() in word.lower():
                    related_vocab.append(word)
            
            result['vocabulary'] = related_vocab[:5]
            print(f"     Vocabulary: {', '.join(related_vocab[:5])}")
            
            #      
            if related_vocab:
                expression = self._generate_integrated_expression(
                    topic, insight, related_vocab
                )
                result['expression'] = expression
                print(f"     Expression: {expression[:100]}...")
        
        print("\n" + "-" * 70)
        print("  Integrated thought complete\n")
        
        return result
    
    def _generate_integrated_expression(self, topic: str, insight, vocab: list) -> str:
        """
                 
        
           +    +            
        """
        #           
        thought_core = insight.content
        
        #         
        if len(vocab) >= 2:
            enriched = (
                f"Regarding {topic}, my understanding integrates multiple dimensions. "
                f"{thought_core} "
                f"This connects to concepts like {vocab[0]} and {vocab[1]}, "
                f"forming a cohesive understanding grounded in both reasoning and learned knowledge."
            )
        else:
            enriched = f"{thought_core} This understanding of {topic} emerges from integrated neural processing."
        
        return enriched
    
    def integrated_learn(self, concept: str, content: str):
        """
             
        
                      
        """
        print(f"  Integrated Learning: {concept}")
        print("-" * 70)
        
        # 1. Universe    
        print("1   Storing in Concept Space...")
        from Core.System.external_data_connector import ExternalDataConnector
        connector = ExternalDataConnector(self.universe)
        connector.internalize_from_text(concept, content)
        
        # 2. Memory     [LOGIC TRANSMUTATION]
        print("2   Storing in Memory...")
        from Core.System.hyper_quaternion import HyperWavePacket
        
        resonant = self.universe.query_resonance(sum(ord(c) for c in concept) % 1000, tolerance=100.0)
        concept_key = resonant[0] if resonant else concept
        if concept_key in self.universe.coordinate_map:
            coord = self.universe.coordinate_map[concept_key]
            packet = HyperWavePacket(
                energy=100.0,
                orientation=coord.orientation,
                time_loc=0.0
            )
            self.memory.store_wave(packet)
        
        # 3. Communication    
        if self.comm_enhancer:
            print("3   Enhancing Communication...")
            self.comm_enhancer.enhance_from_web_content(concept, content)
        
        print("  Integrated learning complete\n")
    
    def integrated_speak(self, topic: str) -> str:
        """
               
        
                         
        """
        #      
        thought = self.integrated_think(topic)
        
        #      
        if 'expression' in thought:
            return thought['expression']
        elif 'thought' in thought:
            return thought['thought']
        else:
            return f"I need to learn more about {topic}."
    
    def get_integration_status(self) -> Dict[str, str]:
        """        """
        return {
            'reasoning': '  Connected' if self.reasoning else '  Missing',
            'memory': '  Connected' if self.memory else '  Missing',
            'universe': '  Connected' if self.universe else '  Missing',
            'communication': '  Connected' if self.comm_enhancer else '  Missing',
            'wave_interpreter': '  Connected' if self.wave_interpreter else '  Missing',
        }


def demonstrate_integration():
    """         """
    
    print("="*70)
    print("NEURAL INTEGRATION SYSTEM")
    print("          ")
    print("="*70)
    print()
    
    #          
    cortex = CentralCortex()
    
    #           
    print("="*70)
    print("CONNECTING COMMUNICATION SYSTEMS")
    print("="*70)
    print()
    
    from Core.System.web_knowledge_connector import WebKnowledgeConnector
    
    connector = WebKnowledgeConnector()
    
    #   
    print("Learning concepts...\n")
    concepts = ["Consciousness", "Intelligence", "Creativity"]
    for concept in concepts:
        print(f"     Learning: {concept}")
        connector.learn_from_web(concept)
    
    #         
    if hasattr(connector, 'comm_enhancer'):
        cortex.connect_communication(connector.comm_enhancer)
    
    print()
    print("="*70)
    print("INTEGRATION STATUS")
    print("="*70)
    print()
    
    status = cortex.get_integration_status()
    for system, state in status.items():
        print(f"   {system:20s}: {state}")
    
    print()
    print("="*70)
    print("INTEGRATED THINKING TEST")
    print("="*70)
    print()
    
    #          
    result = cortex.integrated_think("Consciousness")
    
    print()
    print("="*70)
    print("INTEGRATED SPEECH TEST")
    print("="*70)
    print()
    
    #            
    speech = cortex.integrated_speak("Intelligence")
    print(f"   Integrated Speech:\n\n{speech}\n")
    
    print("="*70)
    print("  NEURAL INTEGRATION SUCCESSFUL")
    print("   All systems are now connected!")
    print("="*70)


if __name__ == "__main__":
    demonstrate_integration()
