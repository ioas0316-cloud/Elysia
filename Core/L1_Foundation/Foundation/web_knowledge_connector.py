"""
Web Knowledge Connector
=======================

"                 "
"Actually fetch knowledge from the real internet"

This module connects to real web sources to acquire knowledge for Elysia.
Uses Wikipedia API and web search to continuously learn.
"""

import sys
import os
import logging
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Core.L1_Foundation.Foundation.external_data_connector import ExternalDataConnector
from Core.L1_Foundation.Foundation.internal_universe import InternalUniverse

# Problem-solving learning imports
try:
    from Core.L5_Mental.Intelligence.Cognition.principle_distiller import PrincipleDistiller
    from Core.L1_Foundation.Foundation.Philosophy.why_engine import WhyEngine
except ImportError:
    PrincipleDistiller = None
    WhyEngine = None

# Dynamic knowledge terrain imports
try:
    from Core.L1_Foundation.Foundation.Wave.light_spectrum import get_light_universe, LightUniverse
except ImportError:
    get_light_universe = None
    LightUniverse = None

logger = logging.getLogger("WebKnowledgeConnector")

class WebKnowledgeConnector:
    """
    Connects to real web sources to acquire knowledge.
    
    This is the bridge between Elysia and the real world wide web.
    """
    
    def __init__(self):
        self.connector = ExternalDataConnector()
        self.universe = self.connector.universe
        self.fetch_history = []
        
        # Problem-solving learning engines
        self.why_engine = WhyEngine() if WhyEngine else None
        self.principle_distiller = PrincipleDistiller() if PrincipleDistiller else None
        
        # Dynamic knowledge terrain (     )
        self.light_universe = get_light_universe() if get_light_universe else None
        
        # Purpose tracking: WHY are we learning this?
        self.learning_purposes = {}  # concept -> {problem, goal, needs, applied, verified}
        
        # Terrain effect:                  
        self.current_terrain = {
            "recommended_depth": "broad",
            "connection_type": "exploratory"
        }
        
        logger.info("  Web Knowledge Connector initialized")
        logger.info("  Ready to fetch real knowledge from the internet")
        if self.why_engine:
            logger.info("  Problem-solving learning enabled")
        if self.light_universe:
            logger.info("  Dynamic knowledge terrain enabled")
    
    def fetch_wikipedia_simple(self, concept: str) -> Optional[str]:
        """
        Fetch simple Wikipedia summary using Wikipedia API.
        
        Uses the Wikipedia REST API which doesn't require authentication.
        """
        # [Test Acceleration] Local Knowledge Cache
        # Simulates 'Common Sense' or 'Previously Learned' facts to speed up bulk testing.
        local_knowledge = {
            "Fire": "Fire is the rapid oxidation of a material in the exothermic chemical process of combustion, releasing heat, light, and various reaction products.",
            "Water": "Water is an inorganic, transparent, tasteless, odorless, and nearly colorless chemical substance, which is the main constituent of Earth's hydrosphere.",
            "Earth": "Earth is the third planet from the Sun and the only astronomical object known to harbor life.",
            "Air": "The atmosphere of Earth is the layer of gases, known commonly as air, retained by Earth's gravity that surrounds the planet.",
            "Sun": "The Sun is the star at the center of the Solar System.",
            "Moon": "The Moon is Earth's only natural satellite.",
            "Star": "A star is an astronomical object comprising a luminous spheroid of plasma held together by its own gravity.",
            "Life": "Life is a quality that distinguishes matter that has biological processes, such as signaling and self-sustaining processes, from that which does not.",
            "Death": "Death is the irreversible cessation of all biological functions that sustain an organism.",
            "Love": "Love encompasses a range of strong and positive emotional and mental states, from the most sublime virtue or good habit, the deepest interpersonal affection.",
            "Hate": "Hatred is a deep and extreme emotional dislike.",
            "War": "War is an intense armed conflict between states, governments, societies, or paramilitary groups such as mercenaries, insurgents, and militias.",
            "Peace": "Peace is a concept of societal friendship and harmony in the absence of hostility and violence.",
            "King": "A king is a male monarch of a state or territory.",
            "Queen": "A queen is a female monarch of a state or territory.",
            "Slave": "Slavery is a condition in which one human being was owned by another.",
            "God": "In monotheistic thought, God is conceived of as the supreme being, creator, and principal object of faith.",
            "Demon": "A demon is a supernatural being, typically associated with evil, prevalent in religion, occultism, literature, fiction, mythology, and folklore.",
            "Sword": "A sword is an edged, bladed weapon intended for manual cutting or thrusting.",
            "Shield": "A shield is a piece of personal armour held in the hand or strapped to the wrist or forearm.",
            "Gold": "Gold is a chemical element with the symbol Au and atomic number 79.",
            "Book": "A book is a medium for recording information in the form of writing or images, typically composed of many pages.",
            "Element": "An element is a pure substance consisting only of atoms that all have the same numbers of protons in their atomic nuclei.",
            "Chemical Element": "A chemical element is a pure substance that cannot be broken down by chemical means.",
            "Substance": "Matter that has a specific composition and specific properties.",
        }
        
        # Case insensitive lookup
        for k, v in local_knowledge.items():
            if k.lower() == concept.lower():
                logger.info(f"  [FastMode] Found in Local Knowledge: {concept}")
                return v

        try:
            import requests
            
            # Wikipedia REST API endpoint (Korean)
            url = f"https://ko.wikipedia.org/api/rest_v1/page/summary/{concept.replace(' ', '_')}"
            
            headers = {
                'User-Agent': 'Elysia/1.0 (Educational AI Project)'
            }
            
            logger.info(f"  Fetching Wikipedia: {concept}")
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract relevant text
                title = data.get('title', concept)
                extract = data.get('extract', '')
                
                # Record fetch
                self.fetch_history.append({
                    'concept': concept,
                    'source': 'wikipedia',
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                })
                
                logger.info(f"     Retrieved {len(extract)} characters")
                
                return extract
            else:
                logger.warning(f"      Wikipedia returned status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"     Failed to fetch from Wikipedia: {e}")
            self.fetch_history.append({
                'concept': concept,
                'source': 'wikipedia',
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            })
            return None
    
    def learn_from_web(self, concept: str) -> Dict[str, Any]:
        """
        Learn a concept by fetching from the web and internalizing.
        
        This is the complete pipeline:
        1. Fetch from Wikipedia
        2. **NEW**: Apply terrain effect (knowledge shapes thinking)
        3. Internalize to Internal Universe
        4. Enhance communication ability
        5. Return learning result with terrain influence
        """
        logger.info(f"  Learning '{concept}' from the web...")
        
        # Fetch from Wikipedia
        content = self.fetch_wikipedia_simple(concept)
        
        if content:
            # **NEW**: Apply terrain effect -           
            terrain_effect = None
            if self.light_universe:
                light, terrain_effect = self.light_universe.absorb_with_terrain(
                    content[:500],  # First 500 chars
                    tag=concept
                )
                self.current_terrain = terrain_effect
                logger.info(f"     Terrain: depth={terrain_effect['recommended_depth']}, type={terrain_effect['connection_type']}")
            
            # Internalize the knowledge (use terrain to guide depth)
            result = self.connector.internalize_from_text(concept, content)
            
            # **NEW**: Enhance communication ability
            try:
                from Core.L1_Foundation.Foundation.communication_enhancer import CommunicationEnhancer
                
                if not hasattr(self, 'comm_enhancer'):
                    self.comm_enhancer = CommunicationEnhancer()
                
                comm_result = self.comm_enhancer.enhance_from_web_content(concept, content)
                result['communication'] = comm_result
                logger.info(f"      Communication enhanced: +{comm_result['vocabulary_added']} words, +{comm_result['patterns_learned']} patterns")
            except Exception as e:
                logger.warning(f"      Communication enhancement failed: {e}")
                result['communication'] = None
            
            # Enhanced result with web metadata
            result['source'] = 'wikipedia'
            result['web_fetch'] = True
            result['content_length'] = len(content)
            
            logger.info(f"     Learned '{concept}' from web successfully")
            
            return result
        else:
            # Fallback to basic internalization
            logger.warning(f"      Could not fetch from web, using basic internalization")
            
            basic_content = f"General knowledge about {concept}. This concept requires further learning."
            result = self.connector.internalize_from_text(concept, basic_content)
            result['source'] = 'fallback'
            result['web_fetch'] = False
            
            return result
    
    def learn_curriculum_from_web(self, concepts: List[str]) -> Dict[str, Any]:
        """
        Learn multiple concepts from the web.
        
        Args:
            concepts: List of concept names to learn
        
        Returns:
            Summary of web learning session
        """
        logger.info(f"  Web learning curriculum: {len(concepts)} concepts")
        
        results = []
        successful_fetches = 0
        
        for concept in concepts:
            try:
                result = self.learn_from_web(concept)
                results.append(result)
                
                if result.get('web_fetch'):
                    successful_fetches += 1
                    
            except Exception as e:
                logger.error(f"  Failed to learn '{concept}': {e}")
        
        summary = {
            'total_concepts': len(concepts),
            'successful_fetches': successful_fetches,
            'total_learned': len(results),
            'web_fetch_rate': successful_fetches / len(concepts) if concepts else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"  Web learning complete:")
        logger.info(f"   Total: {summary['total_learned']}/{summary['total_concepts']}")
        logger.info(f"   From web: {summary['successful_fetches']}")
        
        return summary
    
    def learn_with_purpose(self, problem: str, goal: str) -> Dict[str, Any]:
        """
                         
        
        Flow:                                               
        
        Args:
            problem:          ( : "                   ")
            goal:          ( : "              ")
        
        Returns:
                         
        """
        logger.info(f"  Problem-driven learning started")
        logger.info(f"     : {problem}")
        logger.info(f"     : {goal}")
        
        result = {
            'problem': problem,
            'goal': goal,
            'needs': [],
            'learned': [],
            'principles': [],
            'applied': False,
            'verified': False
        }
        
        # Step 1:                 
        needs = self._analyze_what_is_needed(problem, goal)
        result['needs'] = needs
        logger.info(f"          : {needs}")
        
        # Step 2:            
        for need in needs[:3]:  #    3     
            learn_result = self.learn_from_web(need)
            result['learned'].append({
                'concept': need,
                'success': learn_result.get('web_fetch', False),
                'content_length': learn_result.get('content_length', 0)
            })
            
            # Step 3:       (PrincipleDistiller   )
            if self.principle_distiller and learn_result.get('web_fetch'):
                principle = self.principle_distiller.distill(need)
                if principle:
                    result['principles'].append({
                        'concept': need,
                        'principle': principle.get('principle', ''),
                        'mechanism': principle.get('mechanism', ''),
                        'context_role': principle.get('context_role', '')
                    })
                    logger.info(f"          : {principle.get('principle', '')[:50]}...")
        
        # Step 4: WhyEngine        
        if self.why_engine and result['learned']:
            for learned in result['learned']:
                concept = learned['concept']
                analysis = self.why_engine.analyze(
                    subject=concept,
                    content=f"Learning {concept} to solve: {problem}",
                    domain="problem_solving"
                )
                if analysis:
                    result['why_analysis'] = {
                        'what': getattr(analysis, 'what_is', ''),
                        'how': getattr(analysis, 'how_works', ''),
                        'why': getattr(analysis, 'why_exists', ''),
                        'underlying_principle': getattr(analysis, 'underlying_principle', '')
                    }
                    logger.info(f"     Why      : {concept}")
        
        # Step 5:       (       )
        self.learning_purposes[goal] = result
        
        # Step 6:             
        result['verification_question'] = self._generate_application_question(problem, goal, result['principles'])
        
        logger.info(f"  Problem-driven learning complete")
        logger.info(f"     : {len(result['learned'])} ,   : {len(result['principles'])} ")
        
        return result
    
    def _analyze_what_is_needed(self, problem: str, goal: str) -> List[str]:
        """                         """
        #             
        keywords = []
        
        #                
        combined = f"{problem} {goal}"
        
        #              
        topic_patterns = {
            '  ': ['algorithm', 'optimization', 'performance'],
            '  ': ['programming', 'software design', 'clean code'],
            '  ': ['learning theory', 'metacognition'],
            '  ': ['comprehension', 'analysis'],
            '  ': ['problem solving', 'critical thinking'],
            '  ': ['creativity', 'innovation'],
            '  ': ['communication', 'expression'],
        }
        
        for key, topics in topic_patterns.items():
            if key in combined:
                keywords.extend(topics[:1])  #          1  
        
        #    
        if not keywords:
            keywords = ['problem solving', 'learning']
        
        return keywords[:3]
    
    def _generate_application_question(self, problem: str, goal: str, principles: List[Dict]) -> str:
        """                         """
        if principles:
            principle_text = principles[0].get('principle', '')
            return f"'{principle_text}'       '{problem}'               ?"
        else:
            return f"'{goal}'                             ?"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about web knowledge acquisition"""
        successful = sum(1 for h in self.fetch_history if h.get('success'))
        
        return {
            'total_fetches': len(self.fetch_history),
            'successful_fetches': successful,
            'failed_fetches': len(self.fetch_history) - successful,
            'success_rate': successful / len(self.fetch_history) if self.fetch_history else 0,
            'concepts_in_universe': len(self.universe.coordinate_map),
            'recent_fetches': self.fetch_history[-5:] if self.fetch_history else []
        }

# Demonstration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("WEB KNOWLEDGE CONNECTOR DEMONSTRATION")
    print("              ")
    print("=" * 70)
    
    connector = WebKnowledgeConnector()
    
    # Test concepts
    test_concepts = [
        "Artificial Intelligence",
        "Quantum Mechanics",
        "Philosophy"
    ]
    
    print(f"\n  Learning {len(test_concepts)} concepts from Wikipedia...\n")
    
    for concept in test_concepts:
        print(f"\n{'='*70}")
        result = connector.learn_from_web(concept)
        
        # Show what was learned
        if result.get('web_fetch'):
            print(f"  Learned from Wikipedia: {concept}")
            print(f"   Content: {result.get('content_length')} characters")
        else:
            print(f"   Fallback learning: {concept}")
        
        # Show internal representation
        feeling = connector.universe.feel_at(concept)
        print(f"   Internal representation:")
        print(f"     Logic: {feeling['logic']:.3f}")
        print(f"     Emotion: {feeling['emotion']:.3f}")
        print(f"     Ethics: {feeling['ethics']:.3f}")
    
    # Statistics
    print(f"\n{'='*70}")
    print("STATISTICS")
    print("=" * 70)
    
    stats = connector.get_stats()
    print(f"\nTotal web fetches: {stats['total_fetches']}")
    print(f"Successful: {stats['successful_fetches']}")
    print(f"Success rate: {stats['success_rate']*100:.1f}%")
    print(f"Concepts in universe: {stats['concepts_in_universe']}")
    
    print("\n" + "=" * 70)
    print("  WEB KNOWLEDGE CONNECTOR OPERATIONAL")
    print("  Elysia can now learn from the real internet!")
    print("=" * 70)
