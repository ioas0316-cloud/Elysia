"""
Conflict Resolver Engine (        )
========================================

"                   ,               ."

                              .

     :
- Memory: "       "
- Vision: "           "
    : "          ,             "
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from Core.L1_Foundation.Foundation.Wave.wave_interference import (
    Wave, WaveInterference, InterferenceResult, InterferenceType
)

logger = logging.getLogger("ConflictResolver")


class ConflictType(Enum):
    """     """
    NONE = "none"                    #      
    SEMANTIC = "semantic"            #        (        )
    INTENSITY = "intensity"          #       (     ,       )
    TEMPORAL = "temporal"            #        (   vs   )
    CONTEXTUAL = "contextual"        #       (   vs      )


class ResolutionStrategy(Enum):
    """     """
    DOMINANT = "dominant"            #           
    MERGE = "merge"                  #   /  
    CONTEXTUAL = "contextual"        #         
    UNCERTAIN = "uncertain"          #        
    DEFER = "defer"                  #      


@dataclass
class ConflictOutput:
    """          """
    value: Any                       #     
    source: str                      #          
    confidence: float = 0.5          #     (0-1)
    timestamp: float = 0.0           #      
    context: str = ""                #      


@dataclass
class ResolvedOutput:
    """      """
    value: Any                           #     
    confidence: float                    #       
    strategy: ResolutionStrategy         #       
    conflict_type: ConflictType          #          
    sources: List[str] = field(default_factory=list)  #       
    explanation: str = ""                #      
    alternatives: List[Any] = field(default_factory=list)  #    
    uncertainty: float = 0.0             #     


class ConflictResolver:
    """
                   
    
                                  .
    
    Usage:
        resolver = ConflictResolver()
        outputs = [
            ConflictOutput("red apple", "Memory", 0.8),
            ConflictOutput("green apple", "Vision", 0.9),
        ]
        result = resolver.resolve(outputs)
    """
    
    def __init__(self):
        self.interference_engine = WaveInterference()
        self.resolution_history: List[Dict] = []
    
    def resolve(self, outputs: List[ConflictOutput]) -> ResolvedOutput:
        """
                       .
        
        Args:
            outputs:              
            
        Returns:
            ResolvedOutput:       
        """
        if not outputs:
            return ResolvedOutput(
                value=None,
                confidence=0.0,
                strategy=ResolutionStrategy.UNCERTAIN,
                conflict_type=ConflictType.NONE,
                explanation="No outputs to resolve"
            )
        
        if len(outputs) == 1:
            return ResolvedOutput(
                value=outputs[0].value,
                confidence=outputs[0].confidence,
                strategy=ResolutionStrategy.DOMINANT,
                conflict_type=ConflictType.NONE,
                sources=[outputs[0].source],
                explanation="Single output, no conflict"
            )
        
        # 1.         
        conflict_type = self.detect_conflict_type(outputs)
        
        if conflict_type == ConflictType.NONE:
            #       -               
            best = max(outputs, key=lambda o: o.confidence)
            return ResolvedOutput(
                value=best.value,
                confidence=best.confidence,
                strategy=ResolutionStrategy.DOMINANT,
                conflict_type=ConflictType.NONE,
                sources=[o.source for o in outputs],
                explanation="No conflict detected, using highest confidence"
            )
        
        # 2.             
        waves = self._outputs_to_waves(outputs)
        
        # 3.      
        interference_result = self.interference_engine.calculate_interference(waves)
        
        # 4.                   
        strategy, resolved_value, explanation = self._select_resolution_strategy(
            outputs, interference_result, conflict_type
        )
        
        # 5.      
        result = ResolvedOutput(
            value=resolved_value,
            confidence=interference_result.confidence,
            strategy=strategy,
            conflict_type=conflict_type,
            sources=[o.source for o in outputs],
            explanation=explanation,
            alternatives=[o.value for o in outputs if o.value != resolved_value],
            uncertainty=interference_result.uncertainty
        )
        
        #        
        self._record_resolution(outputs, result)
        
        logger.info(
            f"   Conflict resolved: {conflict_type.value}   {strategy.value} "
            f"(conf={result.confidence:.2f})"
        )
        
        return result
    
    def detect_conflict(self, outputs: List[ConflictOutput]) -> bool:
        """
                    .
        
        Returns:
            True if conflict exists, False otherwise
        """
        return self.detect_conflict_type(outputs) != ConflictType.NONE
    
    def detect_conflict_type(self, outputs: List[ConflictOutput]) -> ConflictType:
        """
                    .
        
        Args:
            outputs:    
            
        Returns:
            ConflictType:          
        """
        if len(outputs) < 2:
            return ConflictType.NONE
        
        values = [str(o.value).lower() for o in outputs]
        confidences = [o.confidence for o in outputs]
        timestamps = [o.timestamp for o in outputs]
        contexts = [o.context for o in outputs]
        
        #                
        if len(set(values)) == 1:
            #          
            if max(confidences) - min(confidences) > 0.3:
                return ConflictType.INTENSITY
            return ConflictType.NONE
        
        #            
        if any(c for c in contexts) and len(set(contexts)) > 1:
            return ConflictType.CONTEXTUAL
        
        #            
        if timestamps and max(timestamps) - min(timestamps) > 3600:  # 1        
            return ConflictType.TEMPORAL
        
        #   :       
        return ConflictType.SEMANTIC
    
    def _outputs_to_waves(self, outputs: List[ConflictOutput]) -> List[Wave]:
        """            """
        waves = []
        for output in outputs:
            #               
            value_hash = abs(hash(str(output.value))) % 1000
            frequency = 432.0 + value_hash * 0.5  # 432-932Hz   
            
            #          
            amplitude = output.confidence
            
            #         
            context_hash = abs(hash(output.context)) % 628  # 0 - 2  * 100
            phase = context_hash / 100.0
            
            wave = Wave(
                frequency=frequency,
                amplitude=amplitude,
                phase=phase,
                source=output.source,
                confidence=output.confidence
            )
            waves.append(wave)
        
        return waves
    
    def _select_resolution_strategy(
        self,
        outputs: List[ConflictOutput],
        interference: InterferenceResult,
        conflict_type: ConflictType
    ) -> Tuple[ResolutionStrategy, Any, str]:
        """
                          
        
        Returns:
            (strategy, resolved_value, explanation)
        """
        #      :             
        if interference.interference_type == InterferenceType.CONSTRUCTIVE:
            #                   ,         
            primary = max(outputs, key=lambda o: o.confidence)
            secondary = [o for o in outputs if o != primary]
            
            if conflict_type == ConflictType.CONTEXTUAL:
                #      
                value = self._merge_contextual(primary, secondary)
                explanation = f"Merged {primary.source} (primary) with contextual info from {[s.source for s in secondary]}"
            else:
                value = primary.value
                explanation = f"{primary.source} confirmed by {[s.source for s in secondary]} (constructive interference)"
            
            return ResolutionStrategy.MERGE, value, explanation
        
        #      :                    
        elif interference.interference_type == InterferenceType.DESTRUCTIVE:
            if interference.confidence < 0.3:
                #                  
                dominant = max(outputs, key=lambda o: o.confidence)
                value = f"[Uncertain] Possibly {dominant.value}"
                explanation = "High uncertainty due to destructive interference"
                return ResolutionStrategy.UNCERTAIN, value, explanation
            else:
                #           
                dominant = max(outputs, key=lambda o: o.confidence)
                explanation = f"Destructive interference: {dominant.source} dominant over {[o.source for o in outputs if o != dominant]}"
                return ResolutionStrategy.DOMINANT, dominant.value, explanation
        
        #      :            
        else:
            if conflict_type == ConflictType.TEMPORAL:
                #         
                newest = max(outputs, key=lambda o: o.timestamp)
                explanation = f"Temporal conflict: using most recent from {newest.source}"
                return ResolutionStrategy.CONTEXTUAL, newest.value, explanation
            
            elif conflict_type == ConflictType.CONTEXTUAL:
                #      
                value = self._create_contextual_response(outputs)
                explanation = "Contextual separation of conflicting outputs"
                return ResolutionStrategy.CONTEXTUAL, value, explanation
            
            else:
                #      
                options = ", ".join([f"{o.source}:{o.value}" for o in outputs])
                value = f"[Multiple possibilities: {options}]"
                explanation = "Mixed interference, decision deferred"
                return ResolutionStrategy.DEFER, value, explanation
    
    def _merge_contextual(self, primary: ConflictOutput, secondary: List[ConflictOutput]) -> Any:
        """         """
        base_value = str(primary.value)
        
        for s in secondary:
            if s.context:
                base_value += f" (also: {s.value} in {s.context} context)"
        
        return base_value
    
    def _create_contextual_response(self, outputs: List[ConflictOutput]) -> Any:
        """             """
        parts = []
        for output in sorted(outputs, key=lambda o: o.confidence, reverse=True):
            if output.context:
                parts.append(f"In {output.context}: {output.value}")
            else:
                parts.append(f"Generally: {output.value}")
        
        return " | ".join(parts)
    
    def _record_resolution(self, outputs: List[ConflictOutput], result: ResolvedOutput):
        """        """
        self.resolution_history.append({
            "inputs": [{"source": o.source, "value": str(o.value)[:50]} for o in outputs],
            "result": str(result.value)[:100],
            "strategy": result.strategy.value,
            "conflict_type": result.conflict_type.value
        })
        
        #    100     
        if len(self.resolution_history) > 100:
            self.resolution_history = self.resolution_history[-100:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """        """
        if not self.resolution_history:
            return {"total": 0, "strategies": {}, "conflict_types": {}}
        
        strategies = {}
        conflict_types = {}
        
        for record in self.resolution_history:
            s = record["strategy"]
            c = record["conflict_type"]
            strategies[s] = strategies.get(s, 0) + 1
            conflict_types[c] = conflict_types.get(c, 0) + 1
        
        return {
            "total": len(self.resolution_history),
            "strategies": strategies,
            "conflict_types": conflict_types
        }


# =============    =============

def demo_conflict_resolution():
    """        """
    print("=" * 60)
    print("   Conflict Resolver Demo")
    print("=" * 60)
    
    resolver = ConflictResolver()
    
    # 1.       
    print("\n[1] Semantic Conflict (      )")
    print("-" * 40)
    outputs1 = [
        ConflictOutput("Apple is red", "Memory", 0.7),
        ConflictOutput("This apple is green", "Vision", 0.9, context="current observation"),
    ]
    result1 = resolver.resolve(outputs1)
    print(f"   Memory: 'Apple is red' (conf=0.7)")
    print(f"   Vision: 'This apple is green' (conf=0.9)")
    print(f"   Resolved: {result1.value}")
    print(f"   Strategy: {result1.strategy.value}")
    print(f"   Explanation: {result1.explanation}")
    
    # 2.    (  )
    print("\n[2] Confirmation (  )")
    print("-" * 40)
    outputs2 = [
        ConflictOutput("The sky is blue", "Memory", 0.8),
        ConflictOutput("Sky appears blue", "Vision", 0.85),
    ]
    result2 = resolver.resolve(outputs2)
    print(f"   Memory + Vision agree on 'sky is blue'")
    print(f"   Resolved: {result2.value}")
    print(f"   Confidence: {result2.confidence:.2f} (boosted)")
    
    # 3.       
    print("\n[3] Temporal Conflict (      )")
    print("-" * 40)
    import time
    old_time = time.time() - 7200  # 2    
    outputs3 = [
        ConflictOutput("Weather: Sunny", "OldForecast", 0.6, timestamp=old_time),
        ConflictOutput("Weather: Rainy", "CurrentSensor", 0.8, timestamp=time.time()),
    ]
    result3 = resolver.resolve(outputs3)
    print(f"   Old (2hr ago): 'Sunny' vs Current: 'Rainy'")
    print(f"   Resolved: {result3.value}")
    print(f"   Strategy: {result3.strategy.value}")
    
    #   
    print("\n" + "=" * 60)
    print("  Resolution Statistics:")
    stats = resolver.get_statistics()
    print(f"   Total resolutions: {stats['total']}")
    print(f"   Strategies used: {stats['strategies']}")
    print("=" * 60)
    print("  Demo Complete!")


if __name__ == "__main__":
    import sys
    
    if "--demo" in sys.argv:
        demo_conflict_resolution()
    else:
        print("Usage: python conflict_resolver.py --demo")