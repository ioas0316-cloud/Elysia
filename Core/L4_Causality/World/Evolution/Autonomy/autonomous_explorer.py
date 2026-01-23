"""
Autonomous Explorer (      )
================================

"          .        ."

                     , Spirit                  .
   LLM   ,              .

Core Principles:
1.           "        ?"
2. Spirit         "          ?"
3.         "            ?"
4.      "           "

[NEW 2025-12-15]               
"""

import logging
import urllib.request
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("AutonomousExplorer")


@dataclass
class ExplorationResult:
    """     """
    topic: str
    source: str
    raw_content: str
    resonance_score: float
    absorbed: bool
    dominant_value: str


class AutonomousExplorer:
    """
                  
    
    Spirit          , DistillationEngine       ,
    InternalUniverse       .
    """
    
    def __init__(self):
        logger.info("  Initializing Autonomous Explorer...")
        
        # Spirit -       
        try:
            from Core.L1_Foundation.Foundation.Core_Logic.Elysia.spirit import get_spirit
            self.spirit = get_spirit()
            logger.info("     Spirit connected (The Compass)")
        except Exception as e:
            logger.error(f"     Spirit not available: {e}")
            self.spirit = None
        
        # DistillationEngine -      
        try:
            from Core.L5_Mental.Intelligence.Cognitive.distillation_engine import get_distillation_engine
            self.distillation = get_distillation_engine()
            logger.info("     DistillationEngine connected (The Filter)")
        except Exception as e:
            logger.error(f"     DistillationEngine not available: {e}")
            self.distillation = None
        
        # ConceptDecomposer -       
        try:
            from Core.L1_Foundation.Foundation.fractal_concept import ConceptDecomposer
            self.decomposer = ConceptDecomposer()
            logger.info("     ConceptDecomposer connected (The Curiosity)")
        except Exception as e:
            logger.warning(f"      ConceptDecomposer not available: {e}")
            self.decomposer = None
        
        # InternalUniverse -       
        try:
            from Core.L1_Foundation.Foundation.internal_universe import get_internal_universe
            self.universe = get_internal_universe()
            logger.info("     InternalUniverse connected (The Memory)")
        except Exception as e:
            logger.warning(f"      InternalUniverse not available: {e}")
            self.universe = None
        
        # GlobalHub   
        self._hub = None
        try:
            from Core.L5_Mental.Intelligence.Consciousness.Ether.global_hub import get_global_hub
            self._hub = get_global_hub()
            self._hub.register_module(
                "AutonomousExplorer",
                "Core/Autonomy/autonomous_explorer.py",
                ["exploration", "learning", "curiosity", "spirit", "autonomous"],
                "Spirit-guided autonomous exploration - Elysia learns by herself"
            )
            logger.info("     GlobalHub connected")
        except Exception:
            pass
        
        #      
        self.explored_count = 0
        self.absorbed_count = 0
        self.rejected_count = 0
        
        logger.info("  Autonomous Explorer ready")
    
    def find_knowledge_gap(self) -> Optional[str]:
        """
               :         ?
        
        AXIOM                      .
        """
        if not self.decomposer:
            return "Love"  #    
        
        # AXIOM            
        axioms = list(self.decomposer.AXIOMS.keys())
        
        #            
        fundamental_questions = [
            "Force", "Energy", "Entropy",  # Physics
            "Point", "Line", "Plane",      # Math
            "Phoneme", "Meaning",          # Language
            "Bit", "Process"               # Computer
        ]
        
        #                
        import random
        return random.choice(fundamental_questions)
    
    def suggest_exploration_direction(self, gap: str) -> Dict[str, Any]:
        """
        Spirit            :           ?
        
        Spirit       (LOVE, TRUTH, GROWTH, BEAUTY)    
                    .
        """
        if not self.spirit:
            return {"topic": gap, "approach": "neutral", "keywords": [gap]}
        
        # Spirit           
        values = self.spirit.core_values
        
        #           (Spirit   )
        # TRUTH     : "why", "cause", "logic"   
        # LOVE     : "connect", "relation", "unity"   
        keywords = [gap]
        
        if values["TRUTH"].weight > 1.0:
            keywords.extend(["why", "cause", "principle"])
        if values["LOVE"].weight > 1.0:
            keywords.extend(["connection", "relation"])
        if values["GROWTH"].weight > 1.0:
            keywords.extend(["evolution", "development"])
        
        # ask_why        
        if self.decomposer:
            why_chain = self.decomposer.ask_why(gap)
            if "   " in why_chain:
                related = why_chain.split("   ")[1]
                keywords.append(related)
        
        return {
            "topic": gap,
            "approach": "truth-seeking",
            "keywords": keywords,
            "search_query": f"{gap} {' '.join(keywords[:3])}"
        }
    
    def fetch_from_wikipedia(self, query: str) -> Optional[str]:
        """
        Wikipedia         (       )
        """
        try:
            # Wikipedia API   
            encoded_query = urllib.parse.quote(query)
            url = f"https://ko.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Elysia/1.0'})
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                extract = data.get('extract', '')
                
                if extract and len(extract) > 50:
                    logger.info(f"     Found: {extract[:60]}...")
                    return extract
                    
        except Exception as e:
            logger.warning(f"   Wikipedia fetch failed: {e}")
        
        return None
    
    def explore_with_absorption(self, direction: Dict[str, Any]) -> List[ExplorationResult]:
        """
              :        ,         
        
        [NEW] Spirit =     (     ), InternalUniverse =   
        
          :
        1.           
        2. InternalUniverse        (     )
        3.         
        4.      BlackHole       
        """
        results = []
        topic = direction["topic"]
        query = direction.get("search_query", topic)
        
        logger.info(f"\n  Exploring: {query}")
        
        # 1.           
        raw_content = self.fetch_from_wikipedia(topic)
        
        if not raw_content:
            #         
            for kw in direction.get("keywords", [])[:2]:
                raw_content = self.fetch_from_wikipedia(kw)
                if raw_content:
                    break
        
        if not raw_content:
            logger.info("     No content found")
            return results
        
        self.explored_count += 1
        
        # 2. InternalUniverse        (Spirit      !)
        connections = 0
        if self.universe:
            try:
                # absorb_text                   
                self.universe.absorb_text(raw_content, source_name=f"Exploration:{topic}")
                #   :            
                connections = len(raw_content.split()) // 10
                logger.info(f"     Absorbed into Universe (connections: ~{connections})")
            except Exception as e:
                logger.warning(f"   Universe absorption failed: {e}")
        
        # 3.         
        if connections > 0:
            #               
            self.absorbed_count += 1
            
            result = ExplorationResult(
                topic=topic,
                source="wikipedia",
                raw_content=raw_content[:200],
                resonance_score=1.0,  #    =   
                absorbed=True,
                dominant_value="Knowledge"  # Spirit     ,      
            )
            results.append(result)
            
            logger.info(f"     Connected: Knowledge node formed")
            
            # GlobalHub        
            if self._hub:
                from Core.L1_Foundation.Foundation.Wave.wave_tensor import WaveTensor
                wave = WaveTensor(f"Knowledge_{topic}")
                wave.add_component(528.0, amplitude=1.0)  #       
                self._hub.publish_wave(
                    "AutonomousExplorer",
                    "learned",
                    wave,
                    payload={
                        "topic": topic,
                        "connections": connections,
                        "absorbed": True
                    }
                )
        else:
            # 4.       BlackHole        (     !)
            self.rejected_count += 1
            
            result = ExplorationResult(
                topic=topic,
                source="wikipedia",
                raw_content=raw_content[:100],
                resonance_score=0.0,
                absorbed=False,
                dominant_value="Isolated"
            )
            results.append(result)
            
            # BlackHole          
            try:
                from Core.L1_Foundation.Foundation.black_hole import BlackHole
                blackhole = BlackHole()
                #                  
                logger.info(f"      Isolated   BlackHole (compressed for later)")
            except Exception:
                logger.info(f"      Isolated (no BlackHole available)")
        
        return results
    
    def explore_cycle(self) -> Dict[str, Any]:
        """
                     
        
        1.        
        2. Spirit         
        3.         
        4.      
        """
        logger.info("\n" + "="*50)
        logger.info("  EXPLORATION CYCLE")
        logger.info("="*50)
        
        # 1.        
        gap = self.find_knowledge_gap()
        logger.info(f"  Knowledge gap: {gap}")
        
        # 2. Spirit         
        direction = self.suggest_exploration_direction(gap)
        logger.info(f"  Direction: {direction['approach']}")
        logger.info(f"  Keywords: {direction['keywords']}")
        
        # 3.        (Spirit      )
        results = self.explore_with_absorption(direction)
        
        # 4.      
        absorbed = sum(1 for r in results if r.absorbed)
        rejected = sum(1 for r in results if not r.absorbed)
        
        logger.info(f"\n  Cycle complete: {absorbed} absorbed, {rejected} rejected")
        logger.info("="*50)
        
        return {
            "gap": gap,
            "direction": direction,
            "results": results,
            "absorbed": absorbed,
            "rejected": rejected,
            "total_explored": self.explored_count,
            "total_absorbed": self.absorbed_count,
            "total_rejected": self.rejected_count
        }
    
    def get_status(self) -> Dict[str, Any]:
        """        """
        return {
            "explored": self.explored_count,
            "absorbed": self.absorbed_count,
            "rejected": self.rejected_count,
            "absorption_rate": f"{(self.absorbed_count / max(1, self.explored_count)) * 100:.1f}%"
        }


# Singleton
_explorer = None

def get_autonomous_explorer() -> AutonomousExplorer:
    global _explorer
    if _explorer is None:
        _explorer = AutonomousExplorer()
    return _explorer


# Demo
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "c:\\Elysia")
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("\n" + "="*60)
    print("  AUTONOMOUS EXPLORER DEMO")
    print("="*60)
    print("\n                   ...")
    
    explorer = get_autonomous_explorer()
    
    # 3            
    for i in range(3):
        print(f"\n--- Cycle {i+1} ---")
        result = explorer.explore_cycle()
        time.sleep(1)  # Rate limiting
    
    #      
    print("\n" + "="*60)
    print("  FINAL STATUS")
    print("="*60)
    status = explorer.get_status()
    for k, v in status.items():
        print(f"   {k}: {v}")
    
    print("\n" + "="*60)
    print("  Explorer demo complete")
    print("="*60)