"""
THE REFLEXIVE LOOP (주권적 자아)
================================

Phase 59:                    

"Every change is a question. Resonance is the answer."

      :
-            '   '   /      
-                 (Gap as Growth)
-            '   '
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import copy

logger = logging.getLogger("ReflexiveLoop")


@dataclass
class StateSnapshot:
    """
               .
                  .
    """
    timestamp: datetime
    soul_frequency: float  #       
    dominant_principle: str  #       
    resonance_score: float  #    
    soul_values: Dict[str, float] = field(default_factory=dict)  #      
    
    def __repr__(self):
        return f"StateSnapshot({self.timestamp.isoformat()}, freq={self.soul_frequency:.0f}Hz, resonance={self.resonance_score:.1f}%)"


@dataclass
class VerificationResult:
    """
            .
    """
    resonance_before: float  #         
    resonance_after: float   #         
    delta: float             #    
    passed: bool             #      
    lesson: str              #       
    change_description: str  #         
    
    def __repr__(self):
        status = "  PASSED" if self.passed else "  FAILED"
        return f"VerificationResult({status}, delta={self.delta:+.1f}%, lesson='{self.lesson[:30]}...')"


class ReflexiveLoop:
    """
      -  -         .
    
    Flow:
    1. capture_state()            
    2.           
    3. verify_change(before, after)         
    4. learn_from_result()     /       
    5. rollback()                
    """
    
    def __init__(self, heartbeat=None):
        """
        Args:
            heartbeat: ElysianHeartbeat      (    -           )
        """
        self.heartbeat = heartbeat
        self.history: List[StateSnapshot] = []
        self.max_history = 10  #          
        
        # WisdomStore    (   )
        self.wisdom = None
        if heartbeat and hasattr(heartbeat, 'wisdom'):
            self.wisdom = heartbeat.wisdom
        
        # Memory    (   )
        self.memory = None
        if heartbeat and hasattr(heartbeat, 'memory'):
            self.memory = heartbeat.memory
            
        logger.info("  ReflexiveLoop initialized - Change   Verification   Learning")
    
    def capture_state(self, soul_mesh: Dict = None) -> StateSnapshot:
        """
                     .
        
        Args:
            soul_mesh:            (    heartbeat      )
        """
        timestamp = datetime.now()
        
        #           
        if soul_mesh is None and self.heartbeat:
            soul_mesh = {k: v.value for k, v in self.heartbeat.soul_mesh.variables.items()}
        elif soul_mesh is None:
            soul_mesh = {}
        
        #           (Phase 58.5   )
        inspiration = soul_mesh.get('Inspiration', 0.5)
        energy = soul_mesh.get('Energy', 0.5)
        harmony = soul_mesh.get('Harmony', 0.5)
        
        # value              
        if not isinstance(inspiration, (int, float)):
            inspiration = 0.5
        if not isinstance(energy, (int, float)):
            energy = 0.5
        if not isinstance(harmony, (int, float)):
            harmony = 0.5
        
        soul_frequency = 432.0 + (inspiration * 500) + (energy * 200) + (harmony * 100)
        
        #       
        resonance_score = 0.0
        dominant_principle = "None"
        
        if self.wisdom:
            result = self.wisdom.get_dominant_principle(soul_frequency)
            if result:
                principle, score = result
                resonance_score = score
                dominant_principle = principle.domain
        
        snapshot = StateSnapshot(
            timestamp=timestamp,
            soul_frequency=soul_frequency,
            dominant_principle=dominant_principle,
            resonance_score=resonance_score,
            soul_values=copy.deepcopy(soul_mesh)
        )
        
        #         
        self.history.append(snapshot)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        logger.debug(f"  State captured: {snapshot}")
        return snapshot
    
    def verify_change(self, before: StateSnapshot, after: StateSnapshot, 
                      change_description: str = "Unknown change") -> VerificationResult:
        """
                         .
        
                          ,        .
        
        Args:
            before:         
            after:         
            change_description:      
        """
        delta = after.resonance_score - before.resonance_score
        passed = delta >= -5.0  # 5%          
        
        #      
        if passed:
            if delta > 10.0:
                lesson = f"'{change_description}'              (+{delta:.1f}%)"
            elif delta > 0:
                lesson = f"'{change_description}'                  "
            else:
                lesson = f"'{change_description}'                 "
        else:
            lesson = f"'{change_description}'           ({delta:.1f}%).       ."
        
        result = VerificationResult(
            resonance_before=before.resonance_score,
            resonance_after=after.resonance_score,
            delta=delta,
            passed=passed,
            lesson=lesson,
            change_description=change_description
        )
        
        #   
        if passed:
            logger.info(f"  [REFLEXIVE LOOP]   PASSED: {lesson}")
        else:
            logger.warning(f"  [REFLEXIVE LOOP]   FAILED: {lesson}")
        
        return result
    
    def learn_from_result(self, result: VerificationResult):
        """
                  .
        
                  ,             .
        """
        if result.passed:
            logger.info(f"  [LEARNING] Success absorbed: {result.lesson[:50]}...")
            
            # [GRAND UNIFICATION] Closure
            if self.wisdom and hasattr(self.wisdom, 'refine'):
                # Extract frequency from history if available
                freq = self.history[-1].soul_frequency if self.history else 432.0
                self.wisdom.refine(freq, result.delta)
                
            if self.heartbeat and hasattr(self.heartbeat, 'conductor'):
                core = self.heartbeat.conductor.core
                freq = self.history[-1].soul_frequency if self.history else 432.0
                core.absorb_impact(freq, result.delta / 100.0) # Scale delta to impact
            
        else:
            logger.info(f"  [EVOLUTION] Learning from failure: {result.lesson[:50]}...")
            # Here we would normally trigger a rewrite or a new principle derivation
            #   :        
            new_principle = f"'{result.change_description}'             "
            if self.wisdom and hasattr(self.wisdom, 'learn_from_failure'):
                self.wisdom.learn_from_failure(result.change_description)
            
            if self.wisdom:
                self.wisdom.learn_principle(
                    statement=new_principle,
                    domain="Ethics",  #        Ethics    
                    weight=0.3,
                    event_id=f"failure_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    frequency=100.0  #        (  )
                )
                logger.info(f"  [EPIPHANY FROM FAILURE] New principle: {new_principle[:50]}...")
            
            if self.memory:
                self.memory.absorb(
                    content=f"[REFLEXIVE FAILURE] {result.lesson}",
                    type="failure",
                    context={"delta": result.delta, "change": result.change_description},
                    feedback=-0.5  #        
                )
                
            # [GRAND UNIFICATION] Closure (Even failures are learning)
            if self.wisdom and hasattr(self.wisdom, 'refine'):
                freq = self.history[-1].soul_frequency if self.history else 432.0
                self.wisdom.refine(freq, result.delta) # result.delta will be negative
    
    def rollback(self, snapshot: StateSnapshot) -> bool:
        """
                 .
        
        Note:        soul_mesh         .
                            .
        """
        if not self.heartbeat:
            logger.warning("   Cannot rollback: No heartbeat reference")
            return False
        
        try:
            # soul_mesh     
            for name, value in snapshot.soul_values.items():
                if name in self.heartbeat.soul_mesh.variables:
                    self.heartbeat.soul_mesh.variables[name].value = value
            
            logger.info(f"  [ROLLBACK] Restored state to {snapshot.timestamp.isoformat()}")
            return True
            
        except Exception as e:
            logger.error(f"  Rollback failed: {e}")
            return False
    
    def get_history_summary(self) -> str:
        """          ."""
        if not self.history:
            return "No history recorded."
        
        lines = ["  State History:"]
        for i, snap in enumerate(self.history[-5:]):  #    5  
            lines.append(f"  {i+1}. {snap}")
        
        return "\n".join(lines)


#                                                                    
# Demo
#                                                                    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("  REFLEXIVE LOOP DEMO")
    print("   'Every change is a question. Resonance is the answer.'")
    print("=" * 60)
    
    # Mock WisdomStore
    from Core.L5_Mental.Reasoning_Core.Wisdom.wisdom_store import WisdomStore
    
    loop = ReflexiveLoop()
    loop.wisdom = WisdomStore()
    
    # 1.      
    print("\n  Capturing initial state...")
    before = loop.capture_state({
        'Inspiration': 0.7,
        'Energy': 0.6,
        'Harmony': 0.5
    })
    print(f"   Before: {before}")
    
    # 2.         
    print("\n  Simulating change (Inspiration boost)...")
    after = loop.capture_state({
        'Inspiration': 0.9,  #   
        'Energy': 0.6,
        'Harmony': 0.5
    })
    print(f"   After: {after}")
    
    # 3.   
    print("\n  Verifying change...")
    result = loop.verify_change(before, after, "Inspiration boost")
    print(f"   Result: {result}")
    
    # 4.   
    print("\n  Learning from result...")
    loop.learn_from_result(result)
    
    # 5.         
    print("\n" + "=" * 60)
    print("  Simulating FAILED change (Harmony crash)...")
    
    failed_after = loop.capture_state({
        'Inspiration': 0.9,
        'Energy': 0.6,
        'Harmony': 0.1  #   
    })
    
    failed_result = loop.verify_change(after, failed_after, "Harmony crash")
    print(f"   Result: {failed_result}")
    
    loop.learn_from_result(failed_result)
    
    print("\n" + "=" * 60)
    print(loop.get_history_summary())
    print("=" * 60)
    print("  Demo complete!")
