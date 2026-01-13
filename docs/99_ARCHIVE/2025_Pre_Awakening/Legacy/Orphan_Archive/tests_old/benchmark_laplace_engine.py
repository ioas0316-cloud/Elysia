"""
Benchmark: Laplace Engine vs Time-Domain Integration

Tests performance improvement from S-domain transformations.
"""

import time
import numpy as np
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.laplace_engine import LaplaceEngine, TransferFunction

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def time_domain_integration(
    dt: float,
    num_steps: int,
    damping: float,
    initial_value: float,
    input_val: float
) -> np.ndarray:
    """
    Solve dx/dt + a*x = input using Euler integration (slow way).
    """
    x = np.zeros(num_steps)
    x[0] = initial_value
    
    for i in range(1, num_steps):
        # dx/dt = -damping*x + input
        dxdt = -damping * x[i-1] + input_val
        x[i] = x[i-1] + dxdt * dt
    
    return x


def laplace_solution(
    dt: float,
    num_steps: int,
    damping: float,
    initial_value: float,
    input_val: float,
   engine: LaplaceEngine
) -> np.ndarray:
    """
    Solve using Laplace transform (fast way).
    """
    # Create transfer function
    tf = TransferFunction(
        numerator=np.array([1.0]),
        denominator=np.array([1.0, damping])
    )
    
    # Time points
    t = np.arange(num_steps) * dt
    
    # Inverse transform
    response = engine.inverse_transform_numerical(tf, input_val, t)
    
    # Add initial condition
    response += initial_value
    
    return response


def benchmark_first_order():
    """Benchmark 1st order system: dx/dt + ax = input"""
    print("\n" + "="*70)
    print("Benchmark: First Order System (dx/dt + ax = input)")
    print("="*70)
    
    # Parameters
    dt = 0.01
    num_steps = 1000
    damping = 2.0
    initial_value = 0.0
    input_val = 1.0
    
    print(f"Parameters: dt={dt}, steps={num_steps}, damping={damping}")
    
    # Engine
    engine = LaplaceEngine()
    
    # Benchmark time-domain
    print("\n⏱️  Time-domain integration (Euler)...")
    start = time.time()
    result_time = time_domain_integration(dt, num_steps, damping, initial_value, input_val)
    time_elapsed = time.time() - start
    print(f"  Completed in {time_elapsed*1000:.2f} ms")
    
    # Benchmark S-domain
    print("\n⚡ S-domain (Laplace transform)...")
    start = time.time()
    result_laplace = laplace_solution(dt, num_steps, damping, initial_value, input_val, engine)
    laplace_elapsed = time.time() - start
    print(f"  Completed in {laplace_elapsed*1000:.2f} ms")
    
    # Compare
    speedup = time_elapsed / laplace_elapsed if laplace_elapsed > 0 else 0
    error = np.mean(np.abs(result_time - result_laplace))
    
    print(f"\n📊 Results:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Avg error: {error:.6f}")
    print(f"  Final value (time-domain): {result_time[-1]:.4f}")
    print(f"  Final value (S-domain): {result_laplace[-1]:.4f}")
    
    return speedup, error


def benchmark_second_order():
    """Benchmark 2nd order oscillator"""
    print("\n" + "="*70)
    print("Benchmark: Second Order Oscillator (d²x/dt² + 2ζω₀(dx/dt) + ω₀²x = input)")
    print("="*70)
    
    # Parameters
    dt = 0.01
    num_steps = 2000
    omega_0 = 5.0  # Natural frequency
    zeta = 0.3     # Damping ratio (underdamped)
    input_val = 1.0
    
    print(f"Parameters: dt={dt}, steps={num_steps}, ω₀={omega_0}, ζ={zeta}")
    
    engine = LaplaceEngine()
    
    # S-domain solution
    print("\n⚡ S-domain solution...")
    start = time.time()
    
    tf = TransferFunction(
        numerator=np.array([omega_0**2]),
        denominator=np.array([1.0, 2*zeta*omega_0, omega_0**2])
    )
    
    t = np.arange(num_steps) * dt
    result_laplace = engine.inverse_transform_numerical(tf, input_val, t)
    
    laplace_elapsed = time.time() - start
    print(f"  Completed in {laplace_elapsed*1000:.2f} ms")
    
    # Analyze resonance
    resonance = engine.analyze_resonance(tf)
    
    print(f"\n🎹 Resonance Analysis:")
    print(f"  Natural frequency: {resonance.natural_frequency:.2f} rad/s")
    print(f"  Damping ratio: {resonance.damping_ratio:.3f}")
    print(f"  Decay time: {resonance.decay_time:.2f} s")
    print(f"  Stable: {resonance.is_stable}")
    
    # For 2nd order, time-domain would require acceleration tracking
    # (Much more complex than 1st order)
    print(f"\n📝 Note: Time-domain requires tracking both position AND velocity")
    print(f"  S-domain gives direct answer with single formula!")
    
    return laplace_elapsed


def main():
    print("\n" + "="*70)
    print("🚀 LAPLACE ENGINE PERFORMANCE BENCHMARK")
    print("="*70)
    
    # Run benchmarks
    speedup_1st, error_1st = benchmark_first_order()
    elapsed_2nd = benchmark_second_order()
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"✅ First order speedup: {speedup_1st:.2f}x faster")
    print(f"✅ Numerical precision: < {error_1st:.1e} error")
    print(f"✅ Second order: Direct solution in {elapsed_2nd*1000:.1f}ms")
    print("\n🌟 Laplace Transform: 미분을 곱셈으로, 복잡함을 단순함으로!")
print("="*70 + "\n")


if __name__ == "__main__":
    main()
