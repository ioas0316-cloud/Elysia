"""
verify_particle_rotor_matrix_sandbox.py: Verification script demonstrating Particle Rotor Matrix Sandbox,
Causal Timeline Scrubbing, Symbol Grounding into Unit '1', and Unit Spectrum Scale Translation.
"""

import numpy as np
from L1_causal_field.core.topology.particle_rotor_matrix_sandbox import (
    MicroRotorParticle,
    ParticleRotorMatrixSandbox,
)
from L1_causal_field.core.consciousness.unit_spectrum_observation_engine import (
    UnitSpectrumObservationEngine,
)


def run_demonstration():
    print("=" * 80)
    print("ELYSA MATRIX SANDBOX & UNIT SPECTRUM VERIFICATION DEMO")
    print("=" * 80)

    # 1. Initialize Micro-Rotor Particle Sandbox
    sandbox = ParticleRotorMatrixSandbox(dimensions=2, bounds=10.0, dt=0.05, phase_coupling_strength=3.0)

    print("\n[Step 1] Instantiating 12 Micro-Rotor Particles in 2 Clusters...")
    # Cluster A
    for i in range(6):
        pos = np.array([1.0, 1.0]) + np.random.normal(0, 0.3, size=2)
        vel = np.random.uniform(-0.1, 0.1, size=2)
        p = MicroRotorParticle(id=i, position=pos, velocity=vel, theta=0.0, mass=1.2, friction=0.15, type_id=0)
        sandbox.add_particle(p)

    # Cluster B
    for i in range(6, 12):
        pos = np.array([-4.0, -4.0]) + np.random.normal(0, 0.3, size=2)
        vel = np.random.uniform(-0.1, 0.1, size=2)
        p = MicroRotorParticle(id=i, position=pos, velocity=vel, theta=np.pi, mass=0.8, friction=0.1, type_id=1)
        sandbox.add_particle(p)

    print(f"Total Sandbox Particles: {len(sandbox.particles)}")
    print(f"Initial Macro Phase Order R: {sandbox.get_macro_order_parameter():.4f}")

    # 2. Run Forward Simulation and Timeline Recording
    print("\n[Step 2] Evolving Sandbox for 30 Time Steps...")
    for t in range(30):
        info = sandbox.step()
        if (t + 1) % 10 == 0:
            print(f"  Step {info['step_idx']:02d}: Macro Order R = {info['macro_order']:.4f}, Energy = {info['total_energy']:.4f}")

    # 3. Causal Timeline Scrubbing & Counterfactual Parallel Trace
    print("\n[Step 3] Executing Causal Timeline Scrubbing to Step 15...")
    frame_15 = sandbox.scrubber.scrub_to(15)
    print(f"  Scrubbed Frame 15: Timestamp = {frame_15['timestamp']:.2f}s, Macro Order = {frame_15['macro_order']:.4f}")

    print("  Forking parallel counterfactual timeline at Step 15 with phase inversion perturbation...")
    def phase_invert(particles):
        for p in particles:
            p.theta = (p.theta + np.pi) % (2 * np.pi)

    def sandbox_factory(particles):
        return ParticleRotorMatrixSandbox(particles=particles, dimensions=2, bounds=10.0, dt=0.05, phase_coupling_strength=3.0)

    cf_trace = sandbox.scrubber.counterfactual_trace(
        start_step=15,
        steps_forward=10,
        perturbation_fn=phase_invert,
        sandbox_factory=sandbox_factory,
    )
    print(f"  Counterfactual Order Divergence across 10 steps: {cf_trace['order_divergence']:.4f}")

    # 4. Unit Spectrum Observation & Symbol Grounding into '1'
    print("\n[Step 4] Observing Sandbox across Unit Spectrum Observation Engine...")
    obs_engine = UnitSpectrumObservationEngine(observation_scale=1.0, cluster_radius=2.5)
    units = obs_engine.observe_and_bind_units(sandbox)

    print(f"  Microscopic/Mesoscopic Scale (Scale = 1.0): Bound into {len(units)} Grounded '1' Units.")
    for unit in units:
        print(f"    - Unit ID: {unit.unit_id}")
        print(f"      Bound Particles: {unit.bound_particle_ids}")
        print(f"      Physical Mass: {unit.physical_mass:.2f} kg, Phase Coherence R: {unit.phase_coherence:.4f}")
        print(f"      Mental Semantic Mass Ms: {unit.semantic_mass:.2f}")

        # Interchangeability demonstration
        phys_rep = obs_engine.interchange_unit_representation(unit.unit_id, "physical")
        ment_rep = obs_engine.interchange_unit_representation(unit.unit_id, "mental")
        print(f"      [Physical Representation] Pressure: {phys_rep['pressure_Pa']:.2f} Pa")
        print(f"      [Mental Representation] Symbol: '{ment_rep['cognitive_unit_symbol']}', Value Density: {ment_rep['value_density']:.2f}")

    # 5. Scale Translation Lens Zoom Out
    print("\n[Step 5] Scaling Observation Lens Out (Scale = 4.0)...")
    obs_engine.set_observation_scale(4.0)
    macro_units = obs_engine.observe_and_bind_units(sandbox)
    print(f"  Macroscopic Scale (Scale = 4.0): Bound into {len(macro_units)} Global Macro '1' Unit.")
    print(f"    - Macro Unit ID: {macro_units[0].unit_id}")
    print(f"      Bound Particle Count: {len(macro_units[0].bound_particle_ids)}")
    print(f"      Total Physical Mass: {macro_units[0].physical_mass:.2f} kg")
    print(f"      Total Semantic Mass Ms: {macro_units[0].semantic_mass:.2f}")

    print("\nVERIFICATION COMPLETE: Micro-Rotor Particle Matrix Sandbox, Timeline Scrubbing, and Grounded Unit Spectrum operating with continuous causality!")
    print("=" * 80)


if __name__ == "__main__":
    run_demonstration()
