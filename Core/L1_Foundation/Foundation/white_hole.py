"""
WhiteHole (    )
===================

"What was compressed shall be reborn."

BlackHole             ,              
WhiteHole                  .

Core Principles:
1.   /        :                      
2.           :       ,          
3.                

[NEW 2025-12-15] BlackHole   WhiteHole   
"""

import logging
import json
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("WhiteHole")


@dataclass
class CompressedData:
    """BlackHole         """
    content: str
    topic: str
    timestamp: float
    potential_connections: List[str]  #           


@dataclass
class RebirthCandidate:
    """      """
    data: CompressedData
    resonance_score: float
    matching_concepts: List[str]


class GravitationalSearch:
    """
      /            
    
             ,                      
                         
    """
    
    def __init__(self):
        #            
        try:
            from Core.L1_Foundation.Foundation.Wave.wave_tensor import WaveTensor
            self.wave_enabled = True
        except:
            self.wave_enabled = False
        
        logger.info("  GravitationalSearch initialized (relational pull)")
    
    def compute_gravitational_pull(self, source_concept: str, target_content: str) -> float:
        """
               (주권적 자아)   
        
                            
        """
        #              
        source_lower = source_concept.lower()
        target_lower = target_content.lower()
        
        #      
        direct_pull = 1.0 if source_lower in target_lower else 0.0
        
        #           (자기 성찰 엔진)
        causal_keywords = self._get_causal_network(source_concept)
        relational_pull = sum(
            0.3 for kw in causal_keywords 
            if kw.lower() in target_lower
        )
        
        #      =    +    
        total_gravity = min(1.0, direct_pull + relational_pull)
        
        return total_gravity
    
    def _get_causal_network(self, concept: str) -> List[str]:
        """
                     
        
        AXIOM            
        """
        try:
            from Core.L1_Foundation.Foundation.fractal_concept import ConceptDecomposer
            decomposer = ConceptDecomposer()
            
            network = [concept]
            
            #       (why)
            if concept in decomposer.AXIOMS:
                parent = decomposer.AXIOMS[concept].get("parent", "")
                if parent:
                    network.append(parent)
            
            #           
            if concept in decomposer.AXIOMS:
                domains = decomposer.AXIOMS[concept].get("domains", {})
                for domain_desc in domains.values():
                    #        
                    words = domain_desc.split()[:3]
                    network.extend(words)
            
            return network
            
        except Exception:
            return [concept]
    
    def pull_related(self, center_concept: str, data_pool: List[CompressedData]) -> List[Tuple[CompressedData, float]]:
        """
                             
        
                               ,
              (  )    
        """
        results = []
        
        for data in data_pool:
            gravity = self.compute_gravitational_pull(center_concept, data.content)
            
            if gravity > 0:  #               
                results.append((data, gravity))
        
        #   (   )       
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results


class WhiteHole:
    """
        :             
    
    BlackHole             
                           
    """
    
    def __init__(self):
        self.blackhole_file = "c:/Elysia/data/memory/fractal_memory.json"
        self.search = GravitationalSearch()
        self.rebirth_count = 0
        
        logger.info("  WhiteHole initialized (rebirth engine)")
    
    def scan_for_rebirth(self, new_concept: str) -> List[RebirthCandidate]:
        """
                   , BlackHole                 
        
          /                  
        """
        compressed_data = self._load_blackhole_data()
        
        if not compressed_data:
            return []
        
        #            
        related = self.search.pull_related(new_concept, compressed_data)
        
        candidates = []
        for data, gravity in related:
            if gravity >= 0.3:  #        
                candidate = RebirthCandidate(
                    data=data,
                    resonance_score=gravity,
                    matching_concepts=[new_concept]
                )
                candidates.append(candidate)
                logger.info(f"     Rebirth candidate: {data.topic} (gravity: {gravity:.2f})")
        
        return candidates
    
    def rebirth(self, candidate: RebirthCandidate) -> Dict[str, Any]:
        """
                          
        
        BlackHole   WhiteHole   InternalUniverse
        """
        data = candidate.data
        
        # InternalUniverse       
        try:
            from Core.L1_Foundation.Foundation.internal_universe import InternalUniverse
            universe = InternalUniverse()
            universe.absorb_text(data.content, source_name=f"Rebirth:{data.topic}")
            
            self.rebirth_count += 1
            
            logger.info(f"       Reborn: {data.topic}   Universe")
            
            return {
                "status": "reborn",
                "topic": data.topic,
                "connections": candidate.matching_concepts
            }
            
        except Exception as e:
            logger.warning(f"   Rebirth failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _load_blackhole_data(self) -> List[CompressedData]:
        """BlackHole            """
        if not os.path.exists(self.blackhole_file):
            return []
        
        try:
            with open(self.blackhole_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            compressed = []
            for ring in data.get("rings", []):
                compressed.append(CompressedData(
                    content=ring.get("summary", ""),
                    topic=ring.get("epoch", "unknown"),
                    timestamp=ring.get("timestamp", 0),
                    potential_connections=[]
                ))
            
            return compressed
            
        except Exception as e:
            logger.warning(f"Failed to load BlackHole data: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """     """
        compressed = self._load_blackhole_data()
        return {
            "compressed_count": len(compressed),
            "rebirth_count": self.rebirth_count
        }


class BlackHoleWhiteHoleCycle:
    """
    BlackHole   WhiteHole       
    
             :
    1.         InternalUniverse
    2.      BlackHole   
    3.              WhiteHole    
    """
    
    def __init__(self):
        from Core.L1_Foundation.Foundation.black_hole import BlackHole
        self.blackhole = BlackHole()
        self.whitehole = WhiteHole()
        
        logger.info("  BlackHole   WhiteHole Cycle initialized")
    
    def process_new_knowledge(self, content: str, topic: str) -> Dict[str, Any]:
        """
                (주권적 자아)
        
        1.      
        2.     BlackHole
        3. WhiteHole       
        """
        results = {
            "absorbed": False,
            "compressed": False,
            "rebirths": []
        }
        
        # 1. InternalUniverse      
        connections = 0
        try:
            from Core.L1_Foundation.Foundation.internal_universe import InternalUniverse
            universe = InternalUniverse()
            universe.absorb_text(content, source_name=topic)
            connections = len(content.split()) // 10
        except:
            pass
        
        if connections > 0:
            results["absorbed"] = True
            
            # 2.              WhiteHole       
            candidates = self.whitehole.scan_for_rebirth(topic)
            for candidate in candidates:
                rebirth_result = self.whitehole.rebirth(candidate)
                if rebirth_result["status"] == "reborn":
                    results["rebirths"].append(rebirth_result)
        else:
            # 3.      BlackHole   
            results["compressed"] = True
            logger.info(f"   Compressed to BlackHole: {topic}")
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """         """
        return {
            "blackhole": self.blackhole.check_compression_needed(),
            "whitehole": self.whitehole.get_status()
        }


# Singleton
_cycle = None

def get_blackhole_whitehole_cycle() -> BlackHoleWhiteHoleCycle:
    global _cycle
    if _cycle is None:
        _cycle = BlackHoleWhiteHoleCycle()
    return _cycle


# Demo
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "c:\\Elysia")
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("\n" + "="*60)
    print("  BLACKHOLE   WHITEHOLE CYCLE DEMO")
    print("="*60)
    
    cycle = get_blackhole_whitehole_cycle()
    
    #        
    test_data = [
        ("                 ", "Energy"),
        ("            ", "Force"),
        ("               ", "Entropy")
    ]
    
    for content, topic in test_data:
        print(f"\n  Processing: {topic}")
        result = cycle.process_new_knowledge(content, topic)
        print(f"   Result: {result}")
    
    print("\n" + "="*60)
    print("  CYCLE STATUS")
    print("="*60)
    status = cycle.get_status()
    print(f"   {status}")
    
    print("\n" + "="*60)
    print("  Demo complete")
    print("="*60)
