"""
ThoughtSpace (      )
=============================

"                 .         ." - Elysia

     :
1.             (No Instant Response)
2.                 (Thought Maturation)
3.              (Error Contemplation)
4.                 (Contextual Adaptation)

       :
-           
-            
-                
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger("Elysia.ThoughtSpace")


class ThoughtState(Enum):
    """     """
    GATHERING = "gathering"      #        
    CONTEMPLATING = "contemplating"  #     
    SYNTHESIZING = "synthesizing"    #     
    READY = "ready"              #         
    ERROR_ANALYZING = "error_analyzing"  #        


@dataclass
class ThoughtShape:
    """
           -                

           (protrusions)        (recesses)
                    
    """
    protrusions: List[str] = field(default_factory=list)  #        (      )
    recesses: List[str] = field(default_factory=list)      #       (     )

    def fits_with(self, other: 'ThoughtShape') -> float:
        """                ? (0.0 ~ 1.0)"""
        if not self.protrusions or not other.recesses:
            return 0.0

        #                           ?
        fits = 0
        for p in self.protrusions:
            for r in other.recesses:
                #         (    overlap)
                if p.lower() in r.lower() or r.lower() in p.lower():
                    fits += 1

        max_possible = max(len(self.protrusions), len(other.recesses))
        return min(1.0, fits / max_possible) if max_possible > 0 else 0.0


@dataclass
class ThoughtParticle:
    """
          -                

    [        ]
    - shape:          (   )
    - illumination:       (   )
    - axis_alignment:           
    """
    id: str
    content: Any                    #   ,   ,     
    source: str                     #         (memory, perception, reasoning)
    resonance: float = 0.5          #            
    weight: float = 1.0             #    
    timestamp: datetime = field(default_factory=datetime.now)

    # [NEW]      
    shape: ThoughtShape = field(default_factory=ThoughtShape)

    # [NEW]       
    illumination: float = 0.5       #       (0=  , 1=  )

    # [NEW]        
    axis_alignment: float = 0.0     #            

    def age_seconds(self) -> float:
        """       ( )"""
        return (datetime.now() - self.timestamp).total_seconds()

    def can_connect_to(self, other: 'ThoughtParticle') -> float:
        """
                      ? (     )
        """
        return self.shape.fits_with(other.shape)

    def illuminate(self, amount: float = 0.2):
        """      (      )"""
        self.illumination = min(1.0, self.illumination + amount)

    def fade(self, amount: float = 0.1):
        """       (      )"""
        self.illumination = max(0.0, self.illumination - amount)


@dataclass
class ErrorTrace:
    """      -               """
    error_type: str                 #       (ImportError, TypeError, LogicError)
    error_message: str              #       
    context: str                    #         
    attempted_action: str           #       
    cause_analysis: str = ""        #      
    learned_principle: str = ""     #      
    prevention_strategy: str = ""   #      
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ContemplationResult:
    """     """
    synthesis: str                  #       
    confidence: float               #    
    contributing_thoughts: List[str]  #        
    time_in_gap: float              #           ( )
    error_insights: List[str] = field(default_factory=list)  #           


class ThoughtSpace:
    """       (The Gap)

                    .
               ,     ,     .

         :
    1.          (Particle Gathering)
    2.          (Resonance Linking)
    3.       (Error Contemplation)
    4.        (Adaptive Synthesis)
    """

    def __init__(self, maturation_threshold: float = 1.0):
        """
        Args:
            maturation_threshold:               ( )
        """
        #                 
        self.active_particles: List[ThoughtParticle] = []

        #      
        self.state: ThoughtState = ThoughtState.GATHERING

        #         
        self.gap_entered_at: Optional[datetime] = None

        #        ( )
        self.maturation_threshold = maturation_threshold

        #       (     )
        self.error_history: List[ErrorTrace] = []

        #       (     )
        self.error_patterns: Dict[str, List[str]] = {}  # error_type -> [     ]

        #         
        self.contemplation_log: List[ContemplationResult] = []

        logger.info("ThoughtSpace initialized - The Gap is open")

    # =========================================================================
    # 1.      /  
    # =========================================================================

    def enter_gap(self, stimulus: str = "") -> None:
        """       -      

        Args:
            stimulus:           
        """
        self.active_particles.clear()
        self.state = ThoughtState.GATHERING
        self.gap_entered_at = datetime.now()

        #             
        if stimulus:
            self.add_thought_particle(
                content=stimulus,
                source="stimulus",
                weight=1.5  #           
            )

        logger.info(f"  Entered The Gap: '{stimulus[:50]}...' if stimulus else 'empty")

    def exit_gap(self) -> ContemplationResult:
        """        -      

        Returns:
                 
        """
        result = self.synthesize()
        self.active_particles.clear()
        self.gap_entered_at = None

        #   
        self.contemplation_log.append(result)
        if len(self.contemplation_log) > 100:
            self.contemplation_log = self.contemplation_log[-50:]

        logger.info(f"  Exited The Gap with synthesis (confidence: {result.confidence:.2f})")
        return result

    # =========================================================================
    # 2.         
    # =========================================================================

    def add_thought_particle(
        self,
        content: Any,
        source: str,
        weight: float = 1.0
    ) -> ThoughtParticle:
        """        

        Args:
            content:      
            source:    (memory, perception, reasoning, error)
            weight:    

        Returns:
                  
        """
        particle_id = hashlib.md5(
            f"{content}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]

        particle = ThoughtParticle(
            id=particle_id,
            content=content,
            source=source,
            weight=weight,
        )

        #               
        particle.resonance = self._calculate_resonance(particle)

        self.active_particles.append(particle)

        #        
        if len(self.active_particles) >= 3:
            self.state = ThoughtState.CONTEMPLATING

        logger.debug(f"  Particle added: {source} (resonance: {particle.resonance:.2f})")
        return particle

    def _calculate_resonance(self, new_particle: ThoughtParticle) -> float:
        """                    """
        if not self.active_particles:
            return 0.5

        #       :              
        same_source = sum(
            1 for p in self.active_particles if p.source == new_particle.source
        )
        source_resonance = same_source / max(1, len(self.active_particles))

        #        :               
        if self.active_particles:
            latest = max(self.active_particles, key=lambda p: p.timestamp)
            time_diff = (new_particle.timestamp - latest.timestamp).total_seconds()
            temporal_resonance = max(0, 1.0 - (time_diff / 60.0))  # 1    
        else:
            temporal_resonance = 0.5

        return (source_resonance + temporal_resonance) / 2

    # =========================================================================
    # 3.       (Error Contemplation) -   !
    # =========================================================================

    def contemplate_error(
        self,
        error: Exception,
        context: str,
        attempted_action: str
    ) -> ErrorTrace:
        """                

        Args:
            error:       
            context:         
            attempted_action:       

        Returns:
                  (     )
        """
        self.state = ThoughtState.ERROR_ANALYZING

        error_type = type(error).__name__
        error_message = str(error)

        #         
        trace = ErrorTrace(
            error_type=error_type,
            error_message=error_message,
            context=context,
            attempted_action=attempted_action,
        )

        #      
        trace.cause_analysis = self._analyze_error_cause(error_type, error_message, context)

        #      
        trace.learned_principle = self._extract_principle(trace)

        #      
        trace.prevention_strategy = self._devise_prevention(trace)

        #   
        self.error_history.append(trace)

        #      
        if error_type not in self.error_patterns:
            self.error_patterns[error_type] = []
        self.error_patterns[error_type].append(trace.cause_analysis)

        #                   
        self.add_thought_particle(
            content=f"Error Insight: {trace.learned_principle}",
            source="error",
            weight=2.0  #              
        )

        logger.info(f"  Error contemplated: {error_type}")
        logger.info(f"   Learned: {trace.learned_principle}")

        return trace

    def _analyze_error_cause(
        self,
        error_type: str,
        error_message: str,
        context: str
    ) -> str:
        """        

                    ,         .
        """
        causes = []

        # ImportError   
        if error_type == "ImportError" or error_type == "ModuleNotFoundError":
            if "No module named" in error_message:
                module_name = error_message.split("'")[-2] if "'" in error_message else "unknown"
                causes.append(f"   '{module_name}'                   ")
                causes.append("                    ")
            elif "cannot import name" in error_message:
                causes.append("                        import   ")

        # AttributeError   
        elif error_type == "AttributeError":
            if "has no attribute" in error_message:
                causes.append("         /        -         ")
                causes.append("None              ")

        # TypeError   
        elif error_type == "TypeError":
            if "argument" in error_message:
                causes.append("                  ")
            elif "not subscriptable" in error_message:
                causes.append("             (None, int  )")

        # FileNotFoundError   
        elif error_type == "FileNotFoundError":
            causes.append("                         ")
            causes.append("      vs            ")

        #   
        if not causes:
            causes.append(f"             : {error_type}")
            causes.append(f"         : {error_message[:100]}")

        #          
        if error_type in self.error_patterns:
            past_causes = self.error_patterns[error_type]
            if past_causes:
                causes.append(f"                 : {past_causes[-1]}")

        return " | ".join(causes)

    def _extract_principle(self, trace: ErrorTrace) -> str:
        """          """
        error_type = trace.error_type

        #                     
        principles = {
            "ImportError": "                   ",
            "ModuleNotFoundError": "                     ",
            "AttributeError": "                    ",
            "TypeError": "         -        ",
            "FileNotFoundError": "                   ",
            "KeyError": "                   ",
            "IndexError": "                        ",
            "ValueError": "                   ",
        }

        base_principle = principles.get(
            error_type,
            "                   "
        )

        return f"{base_principle} (  : {trace.context[:50]})"

    def _devise_prevention(self, trace: ErrorTrace) -> str:
        """        """
        error_type = trace.error_type

        strategies = {
            "ImportError": "try-except  import      ,            ",
            "ModuleNotFoundError": "bootstrap_guardian.py       requirements.txt   ",
            "AttributeError": "hasattr()    getattr(obj, 'attr', default)   ",
            "TypeError": "      + isinstance()      ",
            "FileNotFoundError": "Path.exists()        ",
            "KeyError": "dict.get(key, default)   ",
            "IndexError": "len()        ",
            "ValueError": "           ",
        }

        return strategies.get(
            error_type,
            "                   "
        )

    # =========================================================================
    # 4.    (Synthesis)
    # =========================================================================

    def synthesize(self) -> ContemplationResult:
        """                  

        Returns:
                  
        """
        self.state = ThoughtState.SYNTHESIZING

        if not self.active_particles:
            return ContemplationResult(
                synthesis="         -          ",
                confidence=0.0,
                contributing_thoughts=[],
                time_in_gap=0.0,
            )

        #      
        time_in_gap = 0.0
        if self.gap_entered_at:
            time_in_gap = (datetime.now() - self.gap_entered_at).total_seconds()

        #          
        sorted_particles = sorted(
            self.active_particles,
            key=lambda p: p.weight * p.resonance,
            reverse=True
        )

        #   
        contributing = [str(p.content)[:50] for p in sorted_particles[:5]]

        #         
        error_insights = [
            str(p.content) for p in sorted_particles
            if p.source == "error"
        ]

        #    :              
        avg_resonance = sum(p.resonance for p in self.active_particles) / len(self.active_particles)
        particle_factor = min(1.0, len(self.active_particles) / 5)
        maturity_factor = min(1.0, time_in_gap / self.maturation_threshold)

        confidence = (avg_resonance + particle_factor + maturity_factor) / 3

        #       (             )
        synthesis_parts = []
        for p in sorted_particles[:3]:
            content_str = str(p.content)
            if len(content_str) > 100:
                content_str = content_str[:100] + "..."
            synthesis_parts.append(f"[{p.source}] {content_str}")

        synthesis = "   ".join(synthesis_parts) if synthesis_parts else "     "

        self.state = ThoughtState.READY

        return ContemplationResult(
            synthesis=synthesis,
            confidence=confidence,
            contributing_thoughts=contributing,
            time_in_gap=time_in_gap,
            error_insights=error_insights,
        )

    # =========================================================================
    # 5.      
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """        """
        return {
            "state": self.state.value,
            "active_particles": len(self.active_particles),
            "time_in_gap": (
                (datetime.now() - self.gap_entered_at).total_seconds()
                if self.gap_entered_at else 0.0
            ),
            "error_history_count": len(self.error_history),
            "known_error_patterns": list(self.error_patterns.keys()),
            "contemplation_count": len(self.contemplation_log),
        }

    def get_recent_error_insights(self, n: int = 5) -> List[Dict[str, str]]:
        """           """
        recent = self.error_history[-n:] if self.error_history else []
        return [
            {
                "error_type": e.error_type,
                "learned_principle": e.learned_principle,
                "prevention_strategy": e.prevention_strategy,
            }
            for e in recent
        ]

    # =========================================================================
    # 6.         (Plasma Direction) -       
    # =========================================================================

    def get_thought_direction(self) -> Dict[str, float]:
        """
                       

        "                       "
        """
        if not self.active_particles:
            return {"exploration": 0.1}  #   :      

        #                     
        source_weights = {}
        for p in self.active_particles:
            if p.source not in source_weights:
                source_weights[p.source] = 0.0
            source_weights[p.source] += p.weight * p.resonance

        #    
        total = sum(source_weights.values())
        if total > 0:
            source_weights = {k: v/total for k, v in source_weights.items()}

        return source_weights

    def what_if(self, changes: Dict[str, Any], scenario_name: str = "") -> Dict[str, Any]:
        """
               ? (What-If      )

                              
                      

        Args:
            changes: {"add": [   ], "remove": [id ], "modify_weight": {id: new_weight}}
            scenario_name:        

        Returns:
                    
        """
        import copy

        #         
        simulated_particles = copy.deepcopy(self.active_particles)
        reasoning = []

        #      
        if "add" in changes:
            for content in changes["add"]:
                new_id = hashlib.md5(f"whatif_{content}".encode()).hexdigest()[:8]
                simulated_particles.append(ThoughtParticle(
                    id=new_id,
                    content=content,
                    source="what_if",
                    weight=1.0,
                    resonance=0.5
                ))
                reasoning.append(f"+   : {content[:30]}...")

        #      
        if "remove" in changes:
            before_count = len(simulated_particles)
            simulated_particles = [p for p in simulated_particles if p.id not in changes["remove"]]
            removed_count = before_count - len(simulated_particles)
            reasoning.append(f"-   : {removed_count}    ")

        #       
        if "modify_weight" in changes:
            for pid, new_weight in changes["modify_weight"].items():
                for p in simulated_particles:
                    if p.id == pid:
                        old_weight = p.weight
                        p.weight = new_weight
                        reasoning.append(f"     : {p.content[:20]}... {old_weight:.1f}   {new_weight:.1f}")

        #      
        if not simulated_particles:
            predicted_synthesis = "     -           "
            predicted_confidence = 0.0
        else:
            sorted_particles = sorted(
                simulated_particles,
                key=lambda p: p.weight * p.resonance,
                reverse=True
            )
            synthesis_parts = [f"[{p.source}] {str(p.content)[:50]}" for p in sorted_particles[:3]]
            predicted_synthesis = "   ".join(synthesis_parts)
            predicted_confidence = sum(p.resonance for p in simulated_particles) / len(simulated_particles)

        result = {
            "scenario": scenario_name or "what_if",
            "reasoning": reasoning,
            "predicted_synthesis": predicted_synthesis,
            "predicted_confidence": predicted_confidence,
            "simulated_particle_count": len(simulated_particles),
            "original_particle_count": len(self.active_particles)
        }

        logger.info(f"  What-If: {scenario_name or 'unnamed'}   confidence {predicted_confidence:.2f}")
        return result

    def explore_futures(self, variable: str, values: List[Any] = None) -> List[Dict[str, Any]]:
        """
                 

              (     )                    

        Args:
            variable:       ("add_thought", "remove_error", etc.)
            values:       
        """
        if values is None:
            values = ["love", "fear", "curiosity"]

        futures = []

        for val in values:
            if variable == "add_thought":
                scenario = self.what_if({"add": [val]}, f"add_{val}")
            elif variable == "weight_boost":
                #            val    
                if self.active_particles:
                    scenario = self.what_if(
                        {"modify_weight": {self.active_particles[0].id: float(val)}},
                        f"weight_{val}"
                    )
                else:
                    scenario = {"error": "no particles"}
            else:
                scenario = self.what_if({"add": [f"{variable}:{val}"]}, f"{variable}_{val}")

            futures.append({
                "value": val,
                "result": scenario
            })

        logger.info(f"  Explored {len(futures)} futures for '{variable}'")
        return futures

    def understand_particle(self, particle_id: str) -> Dict[str, Any]:
        """
          (  )       

                     ?              ?
        """
        target = None
        for p in self.active_particles:
            if p.id == particle_id:
                target = p
                break

        if not target:
            return {"error": f"   '{particle_id}'           ."}

        #               
        same_source = [p for p in self.active_particles if p.source == target.source and p.id != particle_id]

        return {
            "name": str(target.content)[:50],
            "source": target.source,
            "weight": target.weight,
            "resonance": target.resonance,
            "age_seconds": target.age_seconds(),
            "related_particles": [str(p.content)[:30] for p in same_source[:3]],
            "interpretation": f"'{target.source}'        ,     {target.resonance:.2f}             "
        }

    def reflect_on_gap(self) -> str:
        """
                  -                
        """
        if not self.active_particles:
            return "          .          ."

        #   
        direction = self.get_thought_direction()
        main_direction = max(direction.items(), key=lambda x: x[1]) if direction else ("unknown", 0)

        #      
        avg_resonance = sum(p.resonance for p in self.active_particles) / len(self.active_particles)
        oldest = min(self.active_particles, key=lambda p: p.timestamp)
        newest = max(self.active_particles, key=lambda p: p.timestamp)

        reflection = f"""
          
{'='*50}

       :
        : {len(self.active_particles)}
        : {avg_resonance:.2f}
     : {self.state.value}

       :
       : {main_direction[0]} ({main_direction[1]:.2f})

       :
         : {str(oldest.content)[:30]}... ({oldest.age_seconds():.1f}   )
        : {str(newest.content)[:30]}...

    :
          '{main_direction[0]}'              .
        {'  ' if avg_resonance > 0.5 else '  '}         {'    ' if avg_resonance > 0.5 else '      '}.
"""

        logger.info(reflection)
        return reflection

    # =========================================================================
    # 7.        (Divergent Expansion) -              
    # =========================================================================

    def expand_thought(self, thought: ThoughtParticle) -> List[ThoughtParticle]:
        """
                               

                           
                    
        """
        new_thoughts = []
        content_str = str(thought.content)

        #       :           
        words = content_str.split()

        for i, word in enumerate(words[:3]):  #    3    
            #         (  )
            new_id = hashlib.md5(f"expand_{thought.id}_{word}".encode()).hexdigest()[:8]

            #      :            "      "   
            new_shape = ThoughtShape(
                protrusions=[word],  #         
                recesses=[w for w in words if w != word][:2]  #          
            )

            new_particle = ThoughtParticle(
                id=new_id,
                content=f"  {word} (     )",
                source="expansion",
                weight=thought.weight * 0.8,  #      
                resonance=thought.resonance,
                shape=new_shape,
                illumination=thought.illumination * 0.7,  #        
            )
            new_thoughts.append(new_particle)

        if new_thoughts:
            logger.info(f"  Expanded: {content_str[:20]}...   {len(new_thoughts)} branches")

        return new_thoughts

    def diverge_all(self, max_depth: int = 3) -> int:
        """
                       (     )

        Returns:            
        """
        if max_depth <= 0:
            return 0

        new_particles = []
        for p in self.active_particles:
            branches = self.expand_thought(p)
            new_particles.extend(branches)

        self.active_particles.extend(new_particles)

        logger.info(f"  Diverged: {len(new_particles)} new thoughts from {len(self.active_particles) - len(new_particles)} seeds")
        return len(new_particles)

    # =========================================================================
    # 8.        (Gravity Attention) -          
    # =========================================================================

    def apply_gravity_attention(self, intention: str):
        """
                 :             

          (intention)             
                        
        """
        intention_lower = intention.lower()
        intention_words = set(intention_lower.split())

        illuminated_count = 0
        faded_count = 0

        for particle in self.active_particles:
            content_lower = str(particle.content).lower()
            content_words = set(content_lower.split())

            #     =      
            overlap = intention_words & content_words
            alignment = len(overlap) / max(1, len(intention_words))

            particle.axis_alignment = alignment

            #   :          
            if alignment > 0.3:
                particle.illuminate(0.3 * alignment)
                illuminated_count += 1
            else:
                particle.fade(0.2)
                faded_count += 1

        logger.info(f"   Gravity Attention: {illuminated_count} illuminated, {faded_count} faded")
        logger.info(f"   Intention: '{intention}'")

    def get_illuminated_thoughts(self, threshold: float = 0.5) -> List[ThoughtParticle]:
        """               (         )"""
        return [p for p in self.active_particles if p.illumination >= threshold]

    def get_dark_thoughts(self, threshold: float = 0.3) -> List[ThoughtParticle]:
        """          (      )"""
        return [p for p in self.active_particles if p.illumination < threshold]

    # =========================================================================
    # 9.       (Boundary Inclusion) -            
    # =========================================================================

    def filter_by_intention(self, intention: str) -> List[ThoughtParticle]:
        """
                       (        )

        "        " =               
        """
        self.apply_gravity_attention(intention)
        return self.get_illuminated_thoughts()

    # =========================================================================
    # 10.          (Puzzle Connection)
    # =========================================================================

    def find_puzzle_connections(self, threshold: float = 0.3) -> List[Tuple[ThoughtParticle, ThoughtParticle, float]]:
        """
                         

                                  
        """
        connections = []

        for i, p1 in enumerate(self.active_particles):
            for p2 in self.active_particles[i+1:]:
                fit_score = p1.can_connect_to(p2)
                if fit_score >= threshold:
                    connections.append((p1, p2, fit_score))

        connections.sort(key=lambda x: x[2], reverse=True)

        if connections:
            logger.info(f"  Found {len(connections)} puzzle connections")

        return connections

    def sovereign_select(self, intention: str) -> Optional[ThoughtParticle]:
        """
              :                  

          (  )       (  )     
        """
        self.apply_gravity_attention(intention)

        #         =         
        illuminated = self.get_illuminated_thoughts(threshold=0.4)

        if not illuminated:
            logger.info("        :          ")
            return None

        #            (  )
        chosen = max(illuminated, key=lambda p: p.illumination)

        logger.info(f"        : '{str(chosen.content)[:30]}...' (illumination: {chosen.illumination:.2f})")
        return chosen

# =============================================================================
# Demo
# =============================================================================


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  ThoughtSpace Demo")
    print("   \"             \"")
    print("=" * 60)

    space = ThoughtSpace(maturation_threshold=0.5)

    # 1.      
    print("\n[1]      :")
    space.enter_gap("         ?")

    # 2.         
    print("\n[2]         :")
    space.add_thought_particle("        ", source="memory")
    space.add_thought_particle("      ", source="reasoning")
    space.add_thought_particle("              ", source="memory")

    # 3.      
    print("\n[3]       (ImportError   ):")
    try:
        import nonexistent_module  #       
    except ImportError as e:
        trace = space.contemplate_error(
            error=e,
            context="                  ",
            attempted_action="import nonexistent_module"
        )
        print(f"     : {trace.cause_analysis}")
        print(f"     : {trace.learned_principle}")
        print(f"     : {trace.prevention_strategy}")

    # 4.   
    print("\n[4]   :")
    result = space.exit_gap()
    print(f"     : {result.synthesis}")
    print(f"      : {result.confidence:.2f}")
    print(f"        : {result.time_in_gap:.2f} ")
    if result.error_insights:
        print(f"        : {result.error_insights}")

    # 5.   
    print("\n[5]   :")
    status = space.get_status()
    print(f"        : {status['known_error_patterns']}")

    print("\n  ThoughtSpace Demo complete!")
