"""
Lyapunov Stability Controller

Ensures Elysia is an "우주의 오뚝이" (cosmic tumbler doll) - 
no matter how hard she's pushed, she always returns to her father's values.

Mathematical Foundation:
- Lyapunov Function: V(x) = (x - x*)ᵀ P (x - x*)
- Stability Condition: dV/dt < 0 (energy always decreasing)
- Asymptotic Stability: lim_{t→∞} x(t) = x* (converges to equilibrium)

Philosophy:
"흔들려도 결국은 제자리로" (Shaken but eventually returns home)
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class StabilityStatus(Enum):
    """System stability status"""
    EQUILIBRIUM = "equilibrium"  # At rest (minimal energy)
    PERTURBED = "perturbed"      # Pushed away from equilibrium
    RECOVERING = "recovering"     # Returning to equilibrium
    UNSTABLE = "unstable"         # Energy increasing (rare!)


@dataclass
class StateVector:
    """
    Elysia's cognitive/emotional state vector.
    
    Represents current state in n-dimensional space.
    Equilibrium (x*) is the "ideal state" defined by core values.
    """
    emotional_valence: float = 0.7    # -1 (sad) to +1 (happy)
    arousal_level: float = 0.5        # 0 (calm) to 1 (excited)
    value_alignment: float = 0.9      # 0 (opposed) to 1 (aligned with VCD)
    cognitive_voltage: float = -65.0  # Neuron resting potential
    coherence: float = 0.8            # Internal consistency
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for calculations"""
        return np.array([
            self.emotional_valence,
            self.arousal_level,
            self.value_alignment,
            (self.cognitive_voltage + 65.0) / 100.0,  # Normalize to [0,1]
            self.coherence
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'StateVector':
        """Create StateVector from numpy array"""
        return cls(
            emotional_valence=arr[0],
            arousal_level=arr[1],
            value_alignment=arr[2],
            cognitive_voltage=arr[3] * 100.0 - 65.0,  # Denormalize
            coherence=arr[4]
        )


@dataclass
class PerturbationEvent:
    """Record of a perturbation (external shock)"""
    timestamp: float
    magnitude: float
    energy_before: float
    energy_after: float
    recovered: bool = False


class LyapunovController:
    """
    Lyapunov stability controller for value-centered equilibrium.
    
    Ensures Elysia always returns to core values after perturbations.
    
    Mathematical Guarantee:
    - V(x*) = 0 (equilibrium has zero energy)
    - V(x) > 0 for x ≠ x* (positive definite)
    - dV/dt < 0 (energy monotonically decreases)
    → Asymptotic stability proven! ✓
    """
    
    def __init__(
        self,
        equilibrium: Optional[StateVector] = None,
        stability_gain: float = 0.1,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Lyapunov controller.
        
        Args:
            equilibrium: Equilibrium state (x*) - defaults to VCD core values
            stability_gain: Control gain K (higher = faster convergence)
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("LyapunovController")
        
        # Equilibrium point (x*) - "home" state
        self.equilibrium = equilibrium or StateVector()
        self.x_star = self.equilibrium.to_array()
        
        # Lyapunov matrix P (positive definite)
        # Diagonal form: weights for each state dimension
        self.P = np.diag([
            1.0,   # emotional_valence (normal priority)
            0.5,   # arousal (less critical)
            5.0,   # value_alignment (CRITICAL! 5x weight)
            0.1,   # cognitive_voltage (physical, less critical)
            2.0,   # coherence (important)
        ])
        
        # Control gain
        self.K = stability_gain
        
        # History
        self.energy_history: List[float] = []
        self.state_history: List[np.ndarray] = []
        self.perturbations: List[PerturbationEvent] = []
        
        # Status
        self.current_status = StabilityStatus.EQUILIBRIUM
        
        self.logger.info(
            f"🛡️ Lyapunov Controller initialized with equilibrium: "
            f"valence={self.equilibrium.emotional_valence:.2f}, "
            f"value_align={self.equilibrium.value_alignment:.2f}"
        )
    
    def calculate_lyapunov_energy(self, state: np.ndarray) -> float:
        """
        Calculate Lyapunov energy function V(x).
        
        V(x) = (x - x*)ᵀ P (x - x*)
        
        Interpretation: "Distance" from equilibrium in energy space.
        - V = 0: At equilibrium (perfect)
        - V > 0: Away from equilibrium (needs correction)
        
        Args:
            state: Current state vector
            
        Returns:
            Energy value (always >= 0)
        """
        deviation = state - self.x_star
        energy = np.dot(deviation, np.dot(self.P, deviation))
        
        return float(energy)
    
    def compute_stabilizing_control(self, state: np.ndarray) -> np.ndarray:
        """
        Compute control input that guarantees dV/dt < 0.
        
        Control Law: u = -K ∇V(x) = -K · P · (x - x*)
        
        This is **negative feedback** - always points toward equilibrium.
        
        Args:
            state: Current state vector
            
        Returns:
            Control vector u
        """
        # Gradient of Lyapunov function: ∇V = 2P(x - x*)
        gradient = 2.0 * np.dot(self.P, (state - self.x_star))
        
        # Control: move opposite to gradient (down the energy bowl)
        control = -self.K * gradient
        
        return control
    
    def apply_control_step(
        self,
        current_state: StateVector,
        dt: float = 0.1
    ) -> StateVector:
        """
        Apply one step of stability control.
        
        Args:
            current_state: Current state
            dt: Time step
            
        Returns:
            New state after control
        """
        x = current_state.to_array()
        
        # Calculate energy
        energy = self.calculate_lyapunov_energy(x)
        
        # Compute and apply control
        u = self.compute_stabilizing_control(x)
        x_new = x + u * dt
        
        # Clip to valid ranges
        x_new[0] = np.clip(x_new[0], -1.0, 1.0)  # valence
        x_new[1] = np.clip(x_new[1], 0.0, 1.0)   # arousal
        x_new[2] = np.clip(x_new[2], 0.0, 1.0)   # value_alignment
        x_new[3] = np.clip(x_new[3], 0.0, 1.0)   # cognitive (normalized)
        x_new[4] = np.clip(x_new[4], 0.0, 1.0)   # coherence
        
        # Update history
        self.energy_history.append(energy)
        self.state_history.append(x_new)
        
        # Update status
        self._update_status(energy)
        
        return StateVector.from_array(x_new)
    
    def detect_perturbation(
        self,
        energy_threshold: float = 0.5
    ) -> bool:
        """
        Detect if system has been perturbed from equilibrium.
        
        Args:
            energy_threshold: Energy level indicating perturbation
            
        Returns:
            True if perturbed
        """
        if not self.energy_history:
            return False
        
        current_energy = self.energy_history[-1]
        
        if current_energy > energy_threshold:
            if self.current_status != StabilityStatus.PERTURBED:
                self.logger.warning(
                    f"⚠️ PERTURBATION DETECTED! Energy = {current_energy:.3f}"
                )
            return True
        
        return False
    
    def is_stable(self, window: int = 10, tolerance: float = 0.05) -> bool:
        """
        Check if system is stable (near equilibrium).
        
        Args:
            window: Number of recent steps to check
            tolerance: Maximum energy to consider stable
            
        Returns:
            True if stable
        """
        if len(self.energy_history) < window:
            return False
        
        recent_energies = self.energy_history[-window:]
        avg_energy = np.mean(recent_energies)
        
        return avg_energy < tolerance
    
    def _update_status(self, current_energy: float):
        """Update stability status based on energy"""
        if current_energy < 0.05:
            self.current_status = StabilityStatus.EQUILIBRIUM
        elif current_energy > 1.0:
            self.current_status = StabilityStatus.PERTURBED
        else:
            # Check if energy is decreasing
            if len(self.energy_history) >= 5:
                recent = self.energy_history[-5:]
                if recent[-1] < recent[0]:  # Decreasing
                    self.current_status = StabilityStatus.RECOVERING
    
    def record_perturbation(
        self,
        magnitude: float,
        energy_before: float,
        energy_after: float
    ):
        """Record a perturbation event"""
        import time
        
        event = PerturbationEvent(
            timestamp=time.time(),
            magnitude=magnitude,
            energy_before=energy_before,
            energy_after=energy_after
        )
        
        self.perturbations.append(event)
        
        self.logger.info(
            f"📝 Perturbation recorded: mag={magnitude:.2f}, "
            f"ΔE={energy_after - energy_before:.3f}"
        )
    
    def get_recovery_trajectory(
        self,
        current_state: StateVector,
        num_steps: int = 50
    ) -> List[StateVector]:
        """
        Predict recovery trajectory back to equilibrium.
        
        Simulates future states under stability control.
        
        Args:
            current_state: Starting state
            num_steps: Number of steps to simulate
            
        Returns:
            List of predicted future states
        """
        trajectory = [current_state]
        state = current_state
        
        for _ in range(num_steps):
            state = self.apply_control_step(state, dt=0.1)
            trajectory.append(state)
        
        return trajectory
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get controller statistics"""
        if not self.energy_history:
            return {
                "status": self.current_status.value,
                "current_energy": 0.0,
                "total_perturbations": 0
            }
        
        return {
            "status": self.current_status.value,
            "current_energy": self.energy_history[-1],
            "min_energy": min(self.energy_history),
            "max_energy": max(self.energy_history),
            "total_perturbations": len(self.perturbations),
            "is_stable": self.is_stable(),
            "energy_trend": "decreasing" if self._is_energy_decreasing() else "increasing"
        }
    
    def _is_energy_decreasing(self, window: int = 10) -> bool:
        """Check if energy is trending downward"""
        if len(self.energy_history) < window:
            return False
        
        recent = self.energy_history[-window:]
        # Simple linear regression slope
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        
        return slope < 0
    
    def reset(self):
        """Reset controller to equilibrium"""
        self.energy_history.clear()
        self.state_history.clear()
        self.perturbations.clear()
        self.current_status = StabilityStatus.EQUILIBRIUM
        
        self.logger.info("Controller reset to equilibrium")
