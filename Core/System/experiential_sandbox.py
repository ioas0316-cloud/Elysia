"""
Experiential Sandbox: The Wing-Beat Simulator
=============================================
"Before it becomes Truth, it must survive the Wind."

This module provides a 'Shadow System' where proposed code changes or logic
mutations can be tested against real-time system pulses without affecting
the primary state of the Sovereign Monad.
"""

import copy
import time
from typing import Dict, List, Any, Optional
from Core.Keystone.sovereign_math import SovereignVector

class ShadowSystem:
    """
    A mirrored instance of a component or the entire Monad state.
    Used for 'Experiential Ingestion' (Phase 160).
    """
    def __init__(self, original_component):
        self.original = original_component
        # Deep copy is used to ensure isolation, but some resources (like network)
        # might need careful handling or mocking.
        self.shadow = copy.deepcopy(original_component)
        self.history: List[Dict] = []
        self.start_time = time.time()
        self.coherence_profile: List[float] = []

    def inject_shadow_logic(self, evolved_code: str):
        """
        Dynamically applies the 'Evolved Code' to the shadow instance.
        """
        # This is a dangerous but necessary step for self-mitosis.
        # We use 'exec' within the context of the shadow class/instance.
        try:
            # We assume the evolved_code defines a method or overrides behavior.
            # For simplicity in the prototype, we assume evolved_code is a full method definition.
            # In a more robust system, we would use AST-based injection.
            namespace = {}
            exec(evolved_code, globals(), namespace)
            
            # If the code defines a function, we bind it to the shadow
            for name, func in namespace.items():
                if callable(func):
                    setattr(self.shadow, name, func.__get__(self.shadow, self.shadow.__class__))
                    
            print(f"🧬 [SANDBOX] Shadow Logic Injected: {len(namespace)} items.")
            return True
        except Exception as e:
            print(f"❌ [SANDBOX] Failed to inject shadow logic: {e}")
            return False

    def simulate_pulse(self, dt: float, inputs: Optional[Dict] = None):
        """
        Runs a pulse on the shadow system and calculates its metrics.
        """
        try:
            # If the component has a pulse or update method
            if hasattr(self.shadow, 'pulse'):
                result = self.shadow.pulse(dt)
            elif hasattr(self.shadow, 'tick'):
                result = self.shadow.tick(dt)
            else:
                result = None

            # Calculate Coherence (Similarity between shadow and original trajectory)
            # Or measuring the internal harmony of the shadow itself.
            coherence = self._calculate_internal_coherence()
            self.coherence_profile.append(coherence)
            
            return {
                "coherence": coherence,
                "result": result,
                "timestamp": time.time() - self.start_time
            }
        except Exception as e:
            print(f"🛑 [SANDBOX] Simulation Crash: {e}")
            return {"coherence": -1.0, "error": str(e)}

    def _calculate_internal_coherence(self) -> float:
        """
        Measures the stability of the shadow state.
        High Soma Heat = Low Coherence.
        """
        # Logic depends on the component. 
        # If it's a Monad, we query its Trinary Engine status.
        if hasattr(self.shadow, 'engine') and hasattr(self.shadow.engine, 'state'):
            return getattr(self.shadow.engine.state, 'coherence', 1.0)
        return 1.0 # Default stability

class ExperientialSandbox:
    def __init__(self):
        self.active_shadows: Dict[str, ShadowSystem] = {}

    def create_simulation(self, target_id: str, component: Any) -> ShadowSystem:
        shadow = ShadowSystem(component)
        self.active_shadows[target_id] = shadow
        return shadow

    def run_wing_beat_test(self, shadow: ShadowSystem, evolved_code: str, ticks: int = 100) -> Dict[str, Any]:
        """
        The 'Wing-Beat' Test: 
        Runs the shadow system with new logic for N ticks.
        Success is defined by sustained or increasing coherence.
        """
        print(f"🦅 [SANDBOX] Starting Wing-Beat Test for {ticks} ticks...")
        
        if not shadow.inject_shadow_logic(evolved_code):
            return {"success": False, "reason": "Injection Failure"}

        dt = 0.1
        stable_ticks = 0
        total_coherence = 0.0

        for i in range(ticks):
            metrics = shadow.simulate_pulse(dt)
            coherence = metrics.get("coherence", -1.0)
            
            if coherence < 0: # Crash
                return {"success": False, "reason": f"Simulation Crash at tick {i}"}
            
            if coherence > 0.7:
                stable_ticks += 1
            
            total_coherence += coherence
            
        avg_coherence = total_coherence / ticks
        stability_ratio = stable_ticks / ticks
        
        success = stability_ratio > 0.8 and avg_coherence > 0.6
        
        print(f"📊 [SANDBOX] Test Result: Success={success}, AvgCoherence={avg_coherence:.2f}, Stability={stability_ratio:.2f}")
        
        return {
            "success": success,
            "avg_coherence": avg_coherence,
            "stability_ratio": stability_ratio,
            "profile": shadow.coherence_profile
        }
