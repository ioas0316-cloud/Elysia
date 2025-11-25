"""
Phase Portrait Neurons - FitzHugh-Nagumo Model

2D geometric neural dynamics for 10x efficiency vs Hodgkin-Huxley.
Perfect for 1060 3GB GPU! 🎮⚡

Implements:
- IntegratorNeuron (Mind/Logos) - 물통형
- ResonatorNeuron (Heart/Pathos) - 그네형  
- LimitCycleGenerator (Soul) - 심장 박동

Based on: "Elegant Geometry of Neural Computations" - Artem Kirsanov
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class NeuronType(Enum):
    """Types of phase portrait neurons"""
    INTEGRATOR = "integrator"  # 물통형 - Mind/Logos
    RESONATOR = "resonator"     # 그네형 - Heart/Pathos
    LIMIT_CYCLE = "limit_cycle" # 심장 박동 - Soul


@dataclass
class PhaseState:
    """State in 2D phase space"""
    v: float  # Voltage-like variable
    w: float  # Recovery variable
    
    def to_array(self) -> np.ndarray:
        return np.array([self.v, self.w])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'PhaseState':
        return cls(v=arr[0], w=arr[1])


class PhasePortraitNeuron:
    """
    Base class for 2D FitzHugh-Nagumo neurons.
    
    Equations:
        dv/dt = v - v³/3 - w + I
        dw/dt = ε(v - γw + β)
    
    Where:
        v: membrane potential (voltage-like)
        w: recovery variable (lumps Na inactivation + K activation)
        I: external stimulus
        ε, γ, β: shape parameters
    
    Philosophy:
        "4차원을 2차원으로, 복잡함을 우아함으로"
        (4D to 2D, complexity to elegance)
    """
    
    def __init__(
        self,
        epsilon: float = 0.08,
        gamma: float = 0.8,
        beta: float = 0.7,
        neuron_type: NeuronType = NeuronType.INTEGRATOR,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize phase portrait neuron.
        
        Args:
            epsilon: Recovery time scale (smaller = slower)
            gamma: Recovery strength
            beta: Recovery baseline
            neuron_type: Integrator, Resonator, or Limit Cycle
            logger: Logger instance
        """
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta
        self.neuron_type = neuron_type
        self.logger = logger or logging.getLogger(f"PhaseNeuron_{neuron_type.value}")
        
        # State variables (2D!)
        self.v = 0.0  # Voltage
        self.w = 0.0  # Recovery
        
        # History
        self.trajectory: List[Tuple[float, float]] = []
        self.spike_times: List[float] = []
        
        # Threshold for spike detection
        self.spike_threshold = 1.0
        self.last_v = 0.0
        
        self.logger.info(
            f"📐 {neuron_type.value.upper()} neuron initialized: "
            f"ε={epsilon}, γ={gamma}, β={beta}"
        )
    
    def dynamics(self, v: float, w: float, I: float) -> Tuple[float, float]:
        """
        FitzHugh-Nagumo dynamics.
        
        Args:
            v: Current voltage
            w: Current recovery
            I: External input
            
        Returns:
            (dv/dt, dw/dt)
        """
        # Cubic nonlinearity for voltage
        dv_dt = v - (v**3) / 3.0 - w + I
        
        # Linear recovery
        dw_dt = self.epsilon * (v - self.gamma * w + self.beta)
        
        return dv_dt, dw_dt
    
    def step(self, I_external: float = 0.0, dt: float = 0.01) -> bool:
        """
        Advance neuron by one time step (Euler integration).
        
        Args:
            I_external: External stimulus current
            dt: Time step
            
        Returns:
            True if spiked, False otherwise
        """
        # Compute derivatives
        dv, dw = self.dynamics(self.v, self.w, I_external)
        
        # Euler update
        self.v += dv * dt
        self.w += dw * dt
        
        # Store trajectory
        self.trajectory.append((self.v, self.w))
        if len(self.trajectory) > 1000:
            self.trajectory.pop(0)
        
        # Detect spike (crossing threshold with positive slope)
        spiked = False
        if self.v > self.spike_threshold and self.last_v <= self.spike_threshold:
            import time
            self.spike_times.append(time.time())
            self.logger.debug(f"🔥 Spike! v={self.v:.2f}")
            spiked = True
        
        self.last_v = self.v
        
        return spiked
    
    def get_nullclines(self, v_range: Tuple[float, float] = (-3, 3)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate nullclines for phase portrait.
        
        Returns:
            (v_nullcline, w_nullcline) as (v_array, w_array) pairs
        """
        v_vals = np.linspace(v_range[0], v_range[1], 200)
        
        # v-nullcline: dv/dt = 0 → w = v - v³/3 + I
        # (assuming I = 0 for nullcline plot)
        w_v_nullcline = v_vals - (v_vals**3) / 3.0
        
        # w-nullcline: dw/dt = 0 → w = (v + β) / γ
        w_w_nullcline = (v_vals + self.beta) / self.gamma
        
        return (v_vals, w_v_nullcline), (v_vals, w_w_nullcline)
    
    def reset(self):
        """Reset to resting state"""
        self.v = 0.0
        self.w = 0.0
        self.trajectory.clear()
        self.spike_times.clear()
        self.last_v = 0.0
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return {
            "type": self.neuron_type.value,
            "v": self.v,
            "w": self.w,
            "fired_recently": len(self.spike_times) > 0 and (
                self.spike_times[-1] > (len(self.trajectory) * 0.01 - 1.0)
            ) if self.spike_times else False
        }


class IntegratorNeuron(PhasePortraitNeuron):
    """
    물통형 - Mind/Logos
    
    Accumulates inputs like filling a bucket.
    Fires when threshold reached.
    
    Use cases:
    - Logical reasoning
    - Decision making
    - Evidence accumulation
    
    Parameters tuned for slow recovery (high integration).
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(
            epsilon=0.01,   # Very slow recovery → integrates
            gamma=0.5,
            beta=0.0,
            neuron_type=NeuronType.INTEGRATOR,
            logger=logger
        )
        
        self.accumulated_input = 0.0
        self.integration_window = []
    
    def accumulate(self, stimulus: float, duration: int = 10):
        """
        Accumulate stimulus over time.
        
        Args:
            stimulus: Input magnitude
            duration: Number of steps to integrate
            
        Returns:
            True if fired during accumulation
        """
        self.logger.debug(f"Accumulating stimulus={stimulus:.2f} for {duration} steps")
        
        for _ in range(duration):
            fired = self.step(I_external=stimulus)
            self.accumulated_input += stimulus
            
            if fired:
                self.logger.info(f"🔥 Integrator fired after accumulating {self.accumulated_input:.2f}")
                self.accumulated_input = 0.0
                return True
        
        return False


class ResonatorNeuron(PhasePortraitNeuron):
    """
    그네형 - Heart/Pathos
    
    Responds selectively to specific frequencies.
    Like pushing a swing - timing matters!
    
    Use cases:
    - Emotional resonance
    - Empathy ("feeling" the rhythm)
    - Pattern recognition
    
    Parameters tuned for oscillatory behavior.
    """
    
    def __init__(
        self,
        natural_frequency: float = 2.0,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(
            epsilon=0.1,    # Moderate recovery → oscillatory
            gamma=0.8,
            beta=0.7,
            neuron_type=NeuronType.RESONATOR,
            logger=logger
        )
        
        self.natural_frequency = natural_frequency
        self.resonance_amplitude = 0.0
    
    def resonate_to(
        self,
        input_signal: np.ndarray,
        time_array: np.ndarray,
        dt: float = 0.01
    ) -> float:
        """
        Measure resonance to input signal.
        
        Args:
            input_signal: Time-varying input
            time_array: Corresponding time points
            dt: Time step
            
        Returns:
            Resonance amplitude (high if frequency matches!)
        """
        v_trajectory = []
        
        for I in input_signal:
            self.step(I_external=I, dt=dt)
            v_trajectory.append(self.v)
        
        # Measure amplitude of oscillation
        v_array = np.array(v_trajectory)
        self.resonance_amplitude = np.max(v_array) - np.min(v_array)
        
        self.logger.info(
            f"Resonance amplitude: {self.resonance_amplitude:.3f} "
            f"(natural f={self.natural_frequency:.1f} Hz)"
        )
        
        return self.resonance_amplitude


class LimitCycleGenerator(PhasePortraitNeuron):
    """
    심장 박동 - Soul (자아의 순환)
    
    Autonomous oscillation - exists even without input!
    "I oscillate, therefore I am."
    
    Use cases:
    - Self-awareness
    - Continuous existence
    - Intrinsic motivation
    
    Parameters tuned for stable limit cycle.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(
            epsilon=0.08,   # Goldilocks zone for limit cycle!
            gamma=0.8,
            beta=0.7,
            neuron_type=NeuronType.LIMIT_CYCLE,
            logger=logger
        )
        
        self.I_tonic = 0.5  # Constant background drive
        self.cycle_period = 0.0
        self.cycle_stable = False
    
    def heartbeat(self, duration: float = 10.0, dt: float = 0.01) -> List[Tuple[float, float]]:
        """
        Generate autonomous oscillation (heartbeat).
        
        Args:
            duration: Duration to oscillate (seconds)
            dt: Time step
            
        Returns:
            Trajectory in phase space
        """
        num_steps = int(duration / dt)
        trajectory = []
        
        self.logger.info(f"💓 Heartbeat starting for {duration}s...")
        
        for step in range(num_steps):
            # No external input - only tonic drive!
            self.step(I_external=self.I_tonic, dt=dt)
            trajectory.append((self.v, self.w))
        
        # Check if limit cycle established
        self._analyze_limit_cycle(trajectory)
        
        return trajectory
    
    def _analyze_limit_cycle(self, trajectory: List[Tuple[float, float]]):
        """Analyze if trajectory forms a stable limit cycle"""
        if len(trajectory) < 100:
            return
        
        # Extract v values from second half (should be settled)
        v_vals = np.array([v for v, w in trajectory[len(trajectory)//2:]])
        
        # Detect peaks (crossing detection)
        peaks = []
        for i in range(1, len(v_vals)-1):
            if v_vals[i] > v_vals[i-1] and v_vals[i] > v_vals[i+1]:
                if v_vals[i] > 0.5:  # Significant peak
                    peaks.append(i)
        
        if len(peaks) >= 2:
            # Estimate period from peak-to-peak intervals
            intervals = np.diff(peaks)
            if len(intervals) > 0:
                avg_interval = np.mean(intervals) * 0.01  # Convert to seconds
                self.cycle_period = avg_interval
                
                # Check stability (period variance)
                if len(intervals) > 1:
                    period_variance = np.std(intervals) / np.mean(intervals)
                    self.cycle_stable = period_variance < 0.1  # 10% tolerance
                
                self.logger.info(
                    f"💓 Limit cycle: period={self.cycle_period:.2f}s, "
                    f"stable={self.cycle_stable}"
                )
