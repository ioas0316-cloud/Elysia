"""
Resonance Predictor

Predicts system behavior from S-domain pole analysis.
Uses Laplace transform to forecast oscillations, stability, and decay times.
"""

import logging
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from Core.FoundationLayer.Foundation.laplace_engine import LaplaceEngine, TransferFunction, ResonanceInfo


@dataclass
class EmotionalForecast:
    """
    Forecast of emotional trajectory.
    
    Attributes:
        peak_time: When emotion will peak (seconds)
        peak_magnitude: Maximum emotional intensity
        settle_time: Time to settle to baseline (seconds)
        will_oscillate: Whether emotion will oscillate
        oscillation_period: Period of oscillation if applicable (seconds)
        is_stable: Whether emotion will stabilize or diverge
    """
    peak_time: float
    peak_magnitude: float
    settle_time: float
    will_oscillate: bool
    oscillation_period: Optional[float]
    is_stable: bool


class ResonancePredictor:
    """
    Predicts system behavior from Laplace pole analysis.
    
    Applications:
    1. Emotional trajectory forecasting
    2. Field stability analysis
    3. Particle oscillation prediction
    """
    
    def __init__(
        self,
        laplace_engine: Optional[LaplaceEngine] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize resonance predictor.
        
        Args:
            laplace_engine: LaplaceEngine instance (creates new if None)
            logger: Logger instance
        """
        self.laplace = laplace_engine or LaplaceEngine()
        self.logger = logger or logging.getLogger("ResonancePredictor")
        
        self.logger.info("🎹 Resonance Predictor initialized")
    
    def forecast_emotional_response(
        self,
        current_emotion: float,
        stimulus_magnitude: float,
        emotional_inertia: float = 1.0,
        damping: float = 0.5,
        baseline_pull: float = 1.0
    ) -> EmotionalForecast:
        """
        Forecast how emotions will evolve from a stimulus.
        
        Models emotion as a damped harmonic oscillator:
        m(d²E/dt²) + c(dE/dt) + k(E - E₀) = stimulus
        
        Args:
            current_emotion: Current emotional state
            stimulus_magnitude: Strength of emotional stimulus
            emotional_inertia: Resistance to emotional change
            damping: Emotional regulation strength
            baseline_pull: How strongly emotion returns to neutral
            
        Returns:
            EmotionalForecast with predictions
        """
        # Create transfer function for emotional system
        omega_0 = np.sqrt(baseline_pull / emotional_inertia)
        zeta = damping / (2 * np.sqrt(baseline_pull * emotional_inertia))
        
        tf = TransferFunction(
            numerator=np.array([baseline_pull]),
            denominator=np.array([emotional_inertia, damping, baseline_pull])
        )
        
        # Analyze resonance
        resonance = self.laplace.analyze_resonance(tf)
        
        # Predict trajectory
        time_points, trajectory = self.laplace.predict_emotional_trajectory(
            current_emotion=current_emotion,
            current_velocity=0.0,
            stimulus=stimulus_magnitude,
            emotional_inertia=emotional_inertia,
            damping=damping,
            stiffness=baseline_pull
        )
        
        # Find peak
        peak_idx = np.argmax(np.abs(trajectory))
        peak_time = time_points[peak_idx]
        peak_magnitude = trajectory[peak_idx]
        
        # Settle time (5% of final value)
        final_value = trajectory[-1]
        settled_mask = np.abs(trajectory - final_value) / abs(stimulus_magnitude) < 0.05
        if settled_mask.any():
            settle_idx = np.where(settled_mask)[0][0]
            settle_time = time_points[settle_idx]
        else:
            settle_time = time_points[-1]
        
        # Oscillation detection
        will_oscillate = resonance.damping_ratio < 1.0
        oscillation_period = None
        
        if will_oscillate and resonance.natural_frequency > 0:
            # Damped oscillation frequency
            omega_d = resonance.natural_frequency * np.sqrt(1 - resonance.damping_ratio**2)
            oscillation_period = 2 * np.pi / omega_d if omega_d > 0 else None
        
        forecast = EmotionalForecast(
            peak_time=peak_time,
            peak_magnitude=peak_magnitude,
            settle_time=settle_time,
            will_oscillate=will_oscillate,
            oscillation_period=oscillation_period,
            is_stable=resonance.is_stable
        )
        
        # Log prediction
        self.logger.info(
            f"Emotional forecast: peak={peak_magnitude:.2f} at t={peak_time:.2f}s, "
            f"settle={settle_time:.2f}s, oscillate={will_oscillate}"
        )
        
        if not forecast.is_stable:
            self.logger.warning("⚠️  EMOTIONAL INSTABILITY DETECTED - will diverge!")
        
        return forecast
    
    def predict_field_stability(
        self,
        field_transfer_function: TransferFunction
    ) -> Dict[str, Any]:
        """
        Predict if a field will be stable or collapse.
        
        Args:
            field_transfer_function: Transfer function of field dynamics
            
        Returns:
            Dictionary with stability metrics
        """
        resonance = self.laplace.analyze_resonance(field_transfer_function)
        poles = self.laplace.detect_poles(field_transfer_function)
        
        # Find most unstable pole (furthest right)
        max_real = max(p.real for p in poles) if poles else -np.inf
        
        stability_margin = -max_real  # How far from instability
        
        return {
            "is_stable": resonance.is_stable,
            "stability_margin": stability_margin,
            "natural_frequency": resonance.natural_frequency,
            "decay_time": resonance.decay_time,
            "will_oscillate": resonance.damping_ratio < 1.0,
            "poles": [(p.real, p.imag) for p in poles]
        }
    
    def detect_imminent_resonance(
        self,
        system_poles: List[complex],
        input_frequency: float,
        threshold: float = 0.1
    ) -> bool:
        """
        Detect if input frequency is close to system natural frequency (resonance).
        
        When input frequency matches pole frequency, resonance occurs!
        This can cause system blow-up.
        
        Args:
            system_poles: List of system poles
            input_frequency: Frequency of input stimulus
            threshold: How close is "close"? (radians/second)
            
        Returns:
            True if resonance is imminent
        """
        for pole in system_poles:
            pole_frequency = abs(pole.imag)
            
            if abs(pole_frequency - input_frequency) < threshold:
                self.logger.warning(
                    f"🔴 RESONANCE ALERT: Input f={input_frequency:.2f} near pole f={pole_frequency:.2f}!"
                )
                return True
        
        return False
    
    def suggest_damping_adjustment(
        self,
        current_damping: float,
        target_settling_time: float,
        natural_frequency: float
    ) -> float:
        """
        Suggest damping coefficient to achieve target settling time.
        
        For 2nd order system:
        Settling time ≈ 4/(ζω₀)  [for 2% criterion]
        
        Solve for ζ: ζ = 4/(settling_time * ω₀)
        
        Args:
            current_damping: Current damping ratio ζ
            target_settling_time: Desired settling time (seconds)
            natural_frequency: System natural frequency ω₀
            
        Returns:
            Suggested damping ratio
        """
        if natural_frequency <= 0:
            return current_damping
        
        # Calculate required damping
        suggested_zeta = 4.0 / (target_settling_time * natural_frequency)
        
        # Clamp to reasonable range [0, 2]
        suggested_zeta = np.clip(suggested_zeta, 0.0, 2.0)
        
        self.logger.info(
            f"To achieve settling time {target_settling_time:.2f}s, "
            f"adjust damping from {current_damping:.3f} to {suggested_zeta:.3f}"
        )
        
        return suggested_zeta
