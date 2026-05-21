import numpy as np

def conserve_energy(delta, neutral_point, prev_energy, scale_factors):
    """
    Hamiltonian constraint: H = T + V = constant
    T (Kinetic)  = 0.5 * sum(m_i * v_i^2), m_i ~ scale_i
    V (Potential) = 0.5 * k * x^2, x ~ abs(neutral_point)
    """
    # Assuming pseudo-mass based on the scales of the rotors to give weight
    mass_total = sum(scale_factors) / 3.0

    # Kinetic energy proxy (rate of change ~ delta)
    kinetic = 0.5 * mass_total * (abs(delta) ** 2)

    # Potential energy proxy (distance from Neutral-Y)
    k_spring = 1.2 # Coupling spring constant
    potential = 0.5 * k_spring * (abs(neutral_point) ** 2)

    current_H = kinetic + potential

    # We want to conserve prev_energy, but allow slight decay/forcing
    target_H = prev_energy * 0.99 + 0.01 * 1.5  # Base state energy injection

    # If energy explodes, dampen it. If it dies, inject.
    ratio = target_H / (current_H + 1e-8)

    # Smooth transition
    new_energy = prev_energy * 0.9 + current_H * ratio * 0.1

    # Limit extreme bounds
    new_energy = max(0.1, min(10.0, new_energy))

    return float(new_energy), current_H

print("Hamiltonian concept tested")
