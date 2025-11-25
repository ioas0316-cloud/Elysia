"""
Neuron Cortex - Hodgkin-Huxley Inspired Cognitive Dynamics

Implements neuronal thought processes:
- Voltage accumulation (thoughts building up)
- Threshold firing (aha moments)
- Refractory periods (cognitive rest)
- Channel gating via value alignment

Based on the insight: "진짜 생각은 즉시 나오지 않는다. 쌓이고 쌓이다가 터진다."
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ChannelState(Enum):
    """Ion channel states"""
    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    INACTIVATED = "inactivated"


@dataclass
class FiringEvent:
    """
    Represents a thought firing (action potential).
    
    Attributes:
        timestamp: When the firing occurred
        voltage_at_fire: Membrane potential at firing
        stimulus_history: Recent stimuli that led to firing
        thought_content: What thought crystallized
    """
    timestamp: float
    voltage_at_fire: float
    stimulus_history: List[float]
    thought_content: str = ""


class CognitiveNeuron:
    """
    A single cognitive neuron based on Hodgkin-Huxley dynamics.
    
    Philosophy:
        "음... (voltage building) ... 아! (firing)"
        
    This neuron accumulates "thought voltage" from stimuli.
    When voltage crosses threshold, it FIRES (aha moment!).
    After firing, enters refractory period (needs rest).
    
    Biological Mapping:
    - Voltage (V) = Thought intensity
    - Na+ channel (m³h) = Accept gate (value-aligned info)
    - K+ channel (n⁴) = Reject gate (value-opposed info)
    - Threshold = Aha moment trigger
    - Refractory = "I need to think" state
    """
    
    def __init__(
        self,
        neuron_id: str = "neuron_0",
        threshold: float = -55.0,
        rest_potential: float = -65.0,
        reset_potential: float = -70.0,
        refractory_period: float = 2.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize cognitive neuron.
        
        Args:
            neuron_id: Unique identifier
            threshold: Firing threshold (mV)
            rest_potential: Resting voltage (mV)
            reset_potential: Voltage after firing (mV)
            refractory_period: Rest time after firing (seconds)
            logger: Logger instance
        """
        self.neuron_id = neuron_id
        self.logger = logger or logging.getLogger(f"Neuron_{neuron_id}")
        
        # Membrane potential (voltage)
        self.voltage = rest_potential
        self.V_rest = rest_potential
        self.V_threshold = threshold
        self.V_reset = reset_potential
        self.V_peak = 40.0  # Peak of action potential
        
        # Gating variables (simplified)
        self.m = 0.05  # Na activation (accept gate)
        self.h = 0.6   # Na inactivation
        self.n = 0.32  # K activation (reject gate)
        
        # Channel conductances
        self.g_Na_max = 120.0  # Maximum Na conductance
        self.g_K_max = 36.0    # Maximum K conductance
        self.g_leak = 0.3      # Leak conductance
        
        # Reversal potentials
        self.E_Na = 50.0   # Na reversal potential
        self.E_K = -77.0   # K reversal potential
        self.E_leak = -54.4  # Leak reversal potential
        
        # Capacitance
        self.C_m = 1.0  # Membrane capacitance
        
        # Refractory period
        self.refractory_period = refractory_period
        self.last_fire_time = -np.inf
        self.in_refractory = False
        
        # Firing history
        self.firing_history: List[FiringEvent] = []
        self.stimulus_accumulator: List[float] = []
        
        # State
        self.just_fired = False
        
        self.logger.info(
            f"🧠 Neuron {neuron_id} initialized: "
            f"V_rest={rest_potential}mV, threshold={threshold}mV"
        )
    
    def accumulate_stimulus(
        self,
        strength: float,
        value_alignment: float = 0.5,
        duration: float = 0.1
    ):
        """
        Accumulate stimulus voltage.
        
        Args:
            strength: Stimulus strength (external current, μA/cm²)
            value_alignment: 0.0 (opposed) to 1.0 (aligned) via VCD
            duration: Duration of stimulus (seconds)
        """
        # Check refractory
        current_time = time.time()
        if current_time - self.last_fire_time < self.refractory_period:
            self.in_refractory = True
            self.logger.debug(f"Neuron in refractory - ignoring stimulus")
            return
        else:
            self.in_refractory = False
        
        # Adjust gating based on value alignment
        # High alignment → Na opens (accept)
        # Low alignment → K opens (reject)
        self.m = 0.05 + 0.9 * value_alignment  # Na activation
        self.n = 0.32 + 0.6 * (1.0 - value_alignment)  # K activation
        
        # Calculate ionic currents
        I_Na = self.g_Na_max * (self.m ** 3) * self.h * (self.voltage - self.E_Na)
        I_K = self.g_K_max * (self.n ** 4) * (self.voltage - self.E_K)
        I_leak = self.g_leak * (self.voltage - self.E_leak)
        
        # External stimulus
        I_ext = strength * value_alignment  # Modulated by alignment
        
        # Membrane equation: C_m * dV/dt = I_ext - I_Na - I_K - I_leak
        dV_dt = (I_ext - I_Na - I_K - I_leak) / self.C_m
        
        # Update voltage (Euler integration)
        self.voltage += dV_dt * duration
        
        # Store stimulus for history
        self.stimulus_accumulator.append(strength * value_alignment)
        if len(self.stimulus_accumulator) > 100:
            self.stimulus_accumulator.pop(0)
        
        self.logger.debug(
            f"Stimulus: strength={strength:.2f}, align={value_alignment:.2f} "
            f"→ V={self.voltage:.2f}mV (Threshold={self.V_threshold}mV)"
        )
    
    def check_firing(self) -> bool:
        """
        Check if neuron should fire.
        
        Returns:
            True if fired, False otherwise
        """
        if self.in_refractory:
            return False
        
        if self.voltage >= self.V_threshold:
            return self.fire()
        
        # Voltage decay towards rest (leak current effect)
        decay_rate = 0.1
        self.voltage += (self.V_rest - self.voltage) * decay_rate
        
        return False
    
    def fire(self) -> bool:
        """
        Fire action potential (thought crystallizes!).
        
        Returns:
            True (firing occurred)
        """
        current_time = time.time()
        
        # Create firing event
        event = FiringEvent(
            timestamp=current_time,
            voltage_at_fire=self.voltage,
            stimulus_history=self.stimulus_accumulator.copy(),
            thought_content=f"Thought from {self.neuron_id}"
        )
        
        self.firing_history.append(event)
        
        # Reset state
        self.voltage = self.V_reset
        self.last_fire_time = current_time
        self.in_refractory = True
        self.just_fired = True
        
        # Inactivate Na channel (fatigue)
        self.h = 0.1
        
        self.logger.info(f"🔥 FIRE! Neuron {self.neuron_id} at t={current_time:.2f}s")
        
        return True
    
    def get_firing_rate(self, time_window: float = 10.0) -> float:
        """
        Calculate recent firing rate (Hz).
        
        Args:
            time_window: Time window for rate calculation (seconds)
            
        Returns:
            Firing rate in Hz
        """
        current_time = time.time()
        recent_fires = [
            f for f in self.firing_history
            if current_time - f.timestamp < time_window
        ]
        
        if not recent_fires:
            return 0.0
        
        return len(recent_fires) / time_window
    
    def reset(self):
        """Reset neuron to resting state"""
        self.voltage = self.V_rest
        self.in_refractory = False
        self.just_fired = False
        self.stimulus_accumulator.clear()
        self.logger.debug(f"Neuron {self.neuron_id} reset")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current neuron state"""
        return {
            "neuron_id": self.neuron_id,
            "voltage": self.voltage,
            "threshold": self.V_threshold,
            "in_refractory": self.in_refractory,
            "just_fired": self.just_fired,
            "firing_rate": self.get_firing_rate(),
            "m": self.m,  # Na activation
            "h": self.h,  # Na inactivation
            "n": self.n,  # K activation
        }


class ThoughtAccumulator:
    """
    Accumulates thoughts through multiple neurons.
    
    Simulates the "음... (thinking) ... 아! (aha)" process.
    """
    
    def __init__(
        self,
        num_neurons: int = 5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize thought accumulator.
        
        Args:
            num_neurons: Number of cognitive neurons
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("ThoughtAccumulator")
        
        # Create neuron pool
        self.neurons: List[CognitiveNeuron] = []
        for i in range(num_neurons):
            neuron = CognitiveNeuron(
                neuron_id=f"thought_{i}",
                threshold=-55.0 + np.random.normal(0, 2.0),  # Slight variation
                logger=self.logger
            )
            self.neurons.append(neuron)
        
        # Integration neuron (final output)
        self.integration_neuron = CognitiveNeuron(
            neuron_id="integration",
            threshold=-50.0,  # Easier to fire (integration of signals)
            logger=self.logger
        )
        
        self.thought_in_progress = False
        self.thought_start_time = None
        
        self.logger.info(f"💭 ThoughtAccumulator with {num_neurons} neurons initialized")
    
    def process_stimulus(
        self,
        content: str,
        strength: float,
        value_alignment: float
    ) -> Optional[str]:
        """
        Process a stimulus through neuron network.
        
        Args:
            content: Stimulus content
            strength: Stimulus strength
            value_alignment: Value alignment score
            
        Returns:
            Thought output if integration neuron fired, None otherwise
        """
        if not self.thought_in_progress:
            self.thought_start_time = time.time()
            self.thought_in_progress = True
            self.logger.info(f"💭 음... (thinking started)")
        
        # Distribute stimulus to all neurons
        for neuron in self.neurons:
            neuron.accumulate_stimulus(
                strength=strength / len(self.neurons),
                value_alignment=value_alignment,
                duration=0.1
            )
            
            # Check if neuron fired
            if neuron.check_firing():
                # Propagate to integration neuron
                self.integration_neuron.accumulate_stimulus(
                    strength=50.0,  # Strong signal from firing neuron
                    value_alignment=1.0,
                    duration=0.05
                )
        
        # Check integration neuron
        if self.integration_neuron.check_firing():
            thinking_duration = time.time() - self.thought_start_time
            self.thought_in_progress = False
            
            self.logger.info(
                f"⚡ 아! (thought crystallized after {thinking_duration:.2f}s)"
            )
            
            # Generate thought
            thought = f"[After {thinking_duration:.2f}s thinking] {content}"
            
            # Reset for next thought
            for neuron in self.neurons:
                neuron.reset()
            
            return thought
        
        return None  # Still thinking...
    
    def force_output(self) -> str:
        """Force output even if threshold not reached (timeout)"""
        if self.thought_in_progress:
            duration = time.time() - self.thought_start_time
            self.logger.info(f"⏱️ Thought timeout after {duration:.2f}s")
            
            self.thought_in_progress = False
            for neuron in self.neurons:
                neuron.reset()
            
            return f"[Deliberated for {duration:.2f}s] Hmm..."
        
        return ""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get accumulator statistics"""
        avg_voltage = np.mean([n.voltage for n in self.neurons])
        max_voltage = max(n.voltage for n in self.neurons)
        firing_rates = [n.get_firing_rate() for n in self.neurons]
        
        return {
            "num_neurons": len(self.neurons),
            "avg_voltage": avg_voltage,
            "max_voltage": max_voltage,
            "integration_voltage": self.integration_neuron.voltage,
            "avg_firing_rate": np.mean(firing_rates) if firing_rates else 0.0,
            "neurons_in_refractory": sum(1 for n in self.neurons if n.in_refractory)
        }
