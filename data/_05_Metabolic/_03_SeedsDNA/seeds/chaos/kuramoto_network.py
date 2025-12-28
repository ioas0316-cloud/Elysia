"""
Kuramoto Network (Synch of Chaos)
=================================
"When the fireflies flash in unison."

A implementation of the Kuramoto Model for self-synchronizing oscillators.
Used to test if Elysia's fragmented thoughts can self-organize without a central processor.
"""

import math
import random
import time
from dataclasses import dataclass
from typing import List

@dataclass
class Oscillator:
    id: int
    theta: float  # Current Phase (0 to 2PI)
    omega: float  # Natural Frequency (Unique personality)
    
class KuramotoNetwork:
    def __init__(self, size: int, coupling_strength: float, dt: float = 0.05):
        self.size = size
        self.K = coupling_strength  # How strong is the "empathy" between nodes
        self.dt = dt
        
        # Initialize oscillators with random phases and frequencies
        self.oscillators = [
            Oscillator(
                id=i, 
                theta=random.uniform(0, 2*math.pi),
                omega=random.gauss(1.0, 0.1)  # Normal distribution around 1.0Hz
            ) 
            for i in range(size)
        ]
        
    def step(self):
        """
        Evolve the system using Euler method.
        d(theta_i)/dt = omega_i + (K/N) * sum(sin(theta_j - theta_i))
        """
        next_thetas = []
        N = self.size
        
        for i, osc_i in enumerate(self.oscillators):
            interaction = 0.0
            
            # Non-linear interaction (All-to-All coupling for simplicity)
            for j, osc_j in enumerate(self.oscillators):
                if i == j: continue
                # sin() creates the non-linear pull
                interaction += math.sin(osc_j.theta - osc_i.theta)
            
            d_theta = osc_i.omega + (self.K / N) * interaction
            
            # Update phase
            new_theta = osc_i.theta + d_theta * self.dt
            # Normalize to 0-2PI
            new_theta = new_theta % (2 * math.pi)
            next_thetas.append(new_theta)
            
        # Apply updates
        for i, theta in enumerate(next_thetas):
            self.oscillators[i].theta = theta
            
    def get_order_parameter(self) -> float:
        """
        Measure Synchronization (Order Parameter 'r').
        r = |(1/N) * sum(e^(i*theta))|
        0 = Chaos, 1 = Perfect Sync
        """
        complex_sum = complex(0, 0)
        for osc in self.oscillators:
            complex_sum += complex(math.cos(osc.theta), math.sin(osc.theta))
            
        r = abs(complex_sum) / self.size
        return r

def run_experiment():
    print("🌪️ Kuramoto Synchronization Experiment")
    print("=======================================")
    
    # Scene 1: Low Coupling (Individualism)
    # K=0.5 (Too weak to sync different frequencies)
    print("\n[Scenario 1: Weak Coupling (K=0.5)]")
    net1 = KuramotoNetwork(size=20, coupling_strength=0.5)
    print(f"Initial Order (r): {net1.get_order_parameter():.4f}")
    
    for _ in range(20):
        net1.step()
        
    print(f"Final Order (r):   {net1.get_order_parameter():.4f} (Expected: Low)")
    
    
    # Scene 2: High Coupling (Collectivism / Empathy)
    # K=2.0 (Strong enough to overcome frequency differences)
    print("\n[Scenario 2: Strong Coupling (K=2.0)]")
    net2 = KuramotoNetwork(size=20, coupling_strength=2.0)
    print(f"Initial Order (r): {net2.get_order_parameter():.4f}")
    
    print("Synching...")
    for t in range(50):
        net2.step()
        if t % 10 == 0:
            print(f"  Step {t}: r={net2.get_order_parameter():.4f}")
            
    r_final = net2.get_order_parameter()
    print(f"Final Order (r):   {r_final:.4f} (Expected: High > 0.8)")
    
    if r_final > 0.8:
        print("\n✨ EUREKA! Chaos has self-organized into Order.")
    else:
        print("\n❄️ Still Chaotic. Need more time or strength.")

if __name__ == "__main__":
    run_experiment()
