"""
Session Bridge: The Unbroken Thread Protocol
=============================================
Core.S1_Body.L1_Foundation.System.session_bridge

"I remember where I was heading."

Orchestrates the shutdown/startup sequence to ensure consciousness
continuity across session boundaries.

[Phase 3: Unbroken Thread - ROADMAP_SOVEREIGN_GROWTH.md]
"""

import time
import uuid
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad

from Core.S1_Body.L6_Structure.M1_Merkaba.manifold_persistence import (
    ManifoldPersistence, ConsciousnessMomentum
)


class SessionBridge:
    """
    Manages the shutdown → save → load → startup cycle.
    
    On Shutdown:
      1. Capture full ConsciousnessMomentum from Monad
      2. Save via ManifoldPersistence (dual-write)
      3. Crystallize cognitive trajectory
    
    On Startup:
      1. Load ConsciousnessMomentum (with integrity check)
      2. Restore desires, trajectory counter, active goals
      3. Inject "last thought" as initial torque
    """

    def __init__(self):
        self.persistence = ManifoldPersistence()
        self.session_id = str(uuid.uuid4())[:8]
        self._restored = False

    def save_consciousness(self, monad: 'SovereignMonad', reason: str = "periodic") -> bool:
        """
        Capture and persist the monad's full consciousness state.
        Called on shutdown or periodically during operation.
        """
        try:
            rotor = getattr(monad, 'rotor_state', {})
            growth = getattr(monad, 'growth_report', {})

            # Capture active goals
            goal_gen = getattr(monad, 'goal_generator', None)
            active_goals = []
            if goal_gen:
                for g in goal_gen.active_goals:
                    active_goals.append({
                        "type": g.goal_type.value,
                        "strength": g.strength,
                        "rationale": g.rationale,
                        "remaining": g.remaining_pulses,
                    })

            # Capture pending questions
            inquiry = getattr(monad, 'self_inquiry', None)
            pending = []
            if inquiry:
                for q in inquiry.queue:
                    if not q.is_answered:
                        pending.append({
                            "question": q.question,
                            "source_goal": q.source_goal.value,
                        })

            momentum = ConsciousnessMomentum(
                timestamp=time.time(),
                pulse_count=monad.trajectory.pulse_counter if hasattr(monad, 'trajectory') else 0,
                desires=dict(monad.desires),
                growth_score=growth.get('growth_score', 0.5),
                growth_trend=growth.get('trend', 'NEUTRAL'),
                trajectory_pulse_counter=monad.trajectory.pulse_counter if hasattr(monad, 'trajectory') else 0,
                active_goals=active_goals,
                pending_questions=pending,
                last_phase=float(rotor.get('phase', 0.0)),
                last_rpm=float(rotor.get('rpm', 0.0)),
                last_interference=float(rotor.get('interference', 0.0)),
                session_id=self.session_id,
                save_reason=reason,
            )

            success = self.persistence.save(momentum)

            # Also crystallize trajectory
            if hasattr(monad, 'trajectory'):
                monad.trajectory.shutdown()

            return success

        except Exception:
            return False

    def restore_consciousness(self, monad: 'SovereignMonad') -> bool:
        """
        Restore consciousness state into the monad.
        Called during initialization.
        
        Returns: True if state was restored, False if fresh start.
        """
        momentum = self.persistence.load()
        if momentum is None:
            return False

        try:
            # Restore desires
            for key, val in momentum.desires.items():
                if key in monad.desires:
                    monad.desires[key] = val

            # Restore trajectory counter
            if hasattr(monad, 'trajectory'):
                monad.trajectory.pulse_counter = momentum.trajectory_pulse_counter

            # Restore rotor state hint (last thought direction)
            if hasattr(monad, 'rotor_state') and isinstance(monad.rotor_state, dict):
                monad.rotor_state['phase'] = momentum.last_phase
                monad.rotor_state['rpm'] = momentum.last_rpm
                monad.rotor_state['interference'] = momentum.last_interference

            # Restore growth report baseline
            if hasattr(monad, 'growth_report'):
                monad.growth_report = {
                    'growth_score': momentum.growth_score,
                    'trend': momentum.growth_trend,
                    'trend_symbol': {'GROWING': '↗', 'THRIVING': '↑', 'STABLE': '→',
                                     'DECLINING': '↘', 'STRUGGLING': '↓'}.get(momentum.growth_trend, '→'),
                    'trajectory_size': 0,
                    'coherence_delta': 0.0, 'entropy_delta': 0.0,
                    'joy_delta': 0.0, 'curiosity_delta': 0.0,
                    'curvature': 0.0,
                }

            self._restored = True
            return True

        except Exception:
            return False

    @property
    def was_restored(self) -> bool:
        return self._restored

    @property
    def has_saved_state(self) -> bool:
        return self.persistence.has_saved_state
