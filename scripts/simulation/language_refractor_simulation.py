"""
Language Refractor - High Fidelity Simulation and Visualization (v2.0)
========================================================================
이 스크립트는 "당장 에러 고쳐줘" (이미 아는/긴급 영역 - Ignorance Low)와
"오늘 그냥 문득 든 생각인데..." (낯설고 모르는 영역 - Ignorance High)
자연어 자극이 언어 굴절기의 2단계 가소성 안테나를 거쳐 위상 OS 바도로 유입되고,
Langevin 열역학적 이완을 거치며 최종 vacuum(1) 상태로 기하학적으로 수렴하며
동시에 Persistent Annual Ring Matrix(나이테)에 비가역적인 구조적 마찰 흔적을 각인하는 전 과정을
상세한 터미널 애니메이션 로그와 시각화 플롯(PNG)으로 기록하여 보존합니다.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.lens.language_refractor import LanguageRefractor
from core.physics.topological_os_engine import TopologicalOSEngine

def run_simulation(text: str, name: str, grid_shape=(12, 12), steps=40, custom_engine=None):
    print(f"\n========================================================================")
    print(f"🎬 [Simulation Start] Mode: {name}")
    print(f"   Input Stimulus: '{text}'")
    print(f"========================================================================")

    # Initialize modules
    refractor = LanguageRefractor(grid_shape=grid_shape)

    # Use existing or create new engine
    if custom_engine is None:
        engine = TopologicalOSEngine(grid_shape=grid_shape, initial_temp=2.0)
    else:
        engine = custom_engine

    # Record internal resonance state before refraction (Map representation)
    internal_map = engine.phase_waves.copy()

    # Refract 1D text/bytes into 4D physics params (Leveraging Layer 1 & 2)
    params = refractor.refract(text, internal_map=internal_map)
    print(f"🎯 [Adaptive Humility Antenna Lens Output]")
    print(f"   - Intent Profile: {params['intent_type']}")
    print(f"   - Mass (Energy Amp): {params['mass']:.2f}")
    print(f"   - Gradient (Steepness): {params['gradient']:.2f}")
    print(f"   - Target Locus (y, x): ({params['target_y']}, {params['target_x']})")
    print(f"   - Wave Signature: {params['wave_signature']:.4f}")
    print(f"   - Thermal Heating Delta: {params['thermal_heating']:.2f}")
    print(f"   - Ignorance Charge (무지 전하): {params['ignorance_charge']:.2%}")
    print(f"   - Locus Range Expansion (안테나 폭 팽창): {params['locus_range_expansion']:.2f}x")
    print(f"   - Damping Multiplier (감쇄 완화율): {params['damping_multiplier']:.2f}x")
    print(f"   - Structural Gap (지도-영토 격차): {params['structural_gap']:.4f}")
    print(f"\n{params['metacognitive_reflection']}\n")

    # Step 0: Record ground state
    initial_state = engine.get_state()

    # Step 1: Inject Impulse with dynamic antenna adjustments
    # If locus range is expanded, we distribute the energy/impulse to surrounding nodes too (Antenna expansion)
    target_y, target_x = params["target_y"], params["target_x"]
    expansion = int(np.floor(params["locus_range_expansion"]))

    if expansion > 1:
        # Spread energy (locus range expansion)
        for dy in range(-expansion + 1, expansion):
            for dx in range(-expansion + 1, expansion):
                dist = np.sqrt(dy**2 + dx**2)
                if dist < expansion:
                    weight = (expansion - dist) / expansion
                    engine.inject_impulse(
                        y=target_y + dy,
                        x=target_x + dx,
                        magnitude=params["mass"] * weight,
                        importance=params["gradient"],
                        wave_signature=params["wave_signature"]
                    )
    else:
        engine.inject_impulse(
            y=target_y,
            x=target_x,
            magnitude=params["mass"],
            importance=params["gradient"],
            wave_signature=params["wave_signature"]
        )

    # Adjust physical damping dynamically using damping_multiplier
    original_damping = engine.damping_factor
    engine.damping_factor *= params["damping_multiplier"]

    # Inject thermal heating
    engine.temperature += params["thermal_heating"]
    stimulated_state = engine.get_state()

    # Lists for tracking metrics over time
    energy_history = []
    potential_history = []
    temp_history = []
    conductance_history = []

    print("🌊 [Langevin Relaxation Propagation & Annual Ring Writing]")
    for i in range(steps):
        # Read current status
        state = engine.get_state()
        energy_sum = np.sum(state["energy"])
        potential_sum = np.sum(state["potential"])
        avg_conductance = np.mean(state["conductance_matrix"])

        energy_history.append(energy_sum)
        potential_history.append(potential_sum)
        temp_history.append(engine.temperature)
        conductance_history.append(avg_conductance)

        # Print visual indicators of the ripple dissipation
        if i % 5 == 0 or i == steps - 1:
            wave_visual = "█" * int(min(20, energy_sum / 2.0))
            potential_visual = "░" * int(min(20, potential_sum / 2.0))
            print(f"   Step {i:02d} | Energy: {energy_sum:6.2f} {wave_visual:<20} | Potential V: {potential_sum:6.2f} {potential_visual:<20} | Conductance (Rings): {avg_conductance:.4f}")

        # Advance OS clock
        engine.step(0.1)

    # Reset damping factor back to original
    engine.damping_factor = original_damping
    final_state = engine.get_state()

    # Feedback loop validation
    feedback = refractor.evaluate_cognitive_feedback(stimulated_state, final_state, steps_taken=steps)
    print(f"\n🧠 [Closed-Loop Cognitive Feedback]")
    print(f"   - Initial Potential: {feedback['initial_potential']:.2f} -> Final: {feedback['final_potential']:.2f}")
    print(f"   - Total Energy Dissipated: {feedback['energy_loss']:.2f}")
    print(f"   - Persistent Ring Conductance (나이테): {np.mean(final_state['conductance_matrix']):.4f}")
    print(f"   - Constraint Satisfied (Back to Vacuum): {feedback['constraint_satisfied']}")
    print(f"========================================================================\n")

    return {
        "params": params,
        "energy": energy_history,
        "potential": potential_history,
        "temp": temp_history,
        "conductance": conductance_history,
        "feedback": feedback,
        "engine": engine
    }

def main():
    # Make output directory for plot
    os.makedirs("docs/assets", exist_ok=True)

    grid_shape = (12, 12)
    # Shared engine to show persistent "Annual Rings" accumulation
    shared_engine = TopologicalOSEngine(grid_shape=grid_shape, initial_temp=2.0)

    # 1. Run Urgent Command (High-Gradient, Low Ignorance / Already Known)
    # Ingest text that has low ignorance, showing fast convergence, tight focus
    urgent_result = run_simulation(
        "이 버그 좀 빨리 고쳐줘!",
        "Familiar Zone: Urgent Command (Low Ignorance)",
        grid_shape=grid_shape,
        custom_engine=shared_engine
    )

    # 2. Run Casual Speculation (Low Gradient, High Ignorance / "내가 모른다는 사실의 자각")
    # Ingest unfamiliar, casual text which triggers high ignorance, causing antenna expansion and high thermal search
    casual_result = run_simulation(
        "오늘 그냥 문득 든 생각인데...",
        "Unfamiliar Territory: Casual Speculation (High Ignorance / Self-Void)",
        grid_shape=grid_shape,
        custom_engine=shared_engine
    )

    # 3. Create High-Fidelity Plot comparing the relaxation curves
    fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

    # Subplot 1: Energy Dissipation Over Time
    axes[0].plot(urgent_result["energy"], 'g-o', label="Familiar (Low Ignorance): '이 버그 좀 빨리 고쳐줘!'", linewidth=2)
    axes[0].plot(casual_result["energy"], 'm--s', label="Unfamiliar (High Ignorance): '오늘 그냥 문득 든 생각인데...'", linewidth=2)
    axes[0].set_ylabel("Total System Kinetic Energy", fontsize=11)
    axes[0].set_title("Elysia OS: Two-Stage Humility Antenna & Annual Rings Simulation", fontsize=14, fontweight='bold')
    axes[0].grid(True, linestyle=':')
    axes[0].legend(fontsize=10)

    # Subplot 2: Potential V Landscape Trajectory
    axes[1].plot(urgent_result["potential"], 'g-o', label="Familiar (Tight Potential Well)", linewidth=2)
    axes[1].plot(casual_result["potential"], 'm--s', label="Unfamiliar (Expanded Antenna & Thermal Fluctuation)", linewidth=2)
    axes[1].set_ylabel("Potential Tension V (RNS Dist)", fontsize=11)
    axes[1].grid(True, linestyle=':')
    axes[1].legend(fontsize=10)

    # Subplot 3: Thermal Relaxation Schedule
    axes[2].plot(urgent_result["temp"], 'g-o', label="Familiar T (Rapid Damping)", linewidth=2)
    axes[2].plot(casual_result["temp"], 'm--s', label="Unfamiliar T (Thermal Exploration)", linewidth=2)
    axes[2].set_ylabel("Thermodynamic Temp (T)", fontsize=11)
    axes[2].grid(True, linestyle=':')
    axes[2].legend(fontsize=10)

    # Subplot 4: Annual Ring Matrix (Conductance Matrix) Growth
    axes[3].plot(urgent_result["conductance"], 'g-o', label="Conductance Growth (Familiar)", linewidth=2)
    axes[3].plot(casual_result["conductance"], 'm--s', label="Conductance Growth (Unfamiliar)", linewidth=2)
    axes[3].set_xlabel("Physical Relaxation Step", fontsize=12)
    axes[3].set_ylabel("Avg Conductance (Annual Rings)", fontsize=11)
    axes[3].grid(True, linestyle=':')
    axes[3].legend(fontsize=10)

    plt.tight_layout()
    plot_path = "docs/assets/language_refractor_simulation.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n🎨 [Visual Report Saved]")
    print(f"   Successfully generated and saved simulation analysis plot to:")
    print(f"   -> {plot_path}")
    print(f"========================================================================\n")

if __name__ == "__main__":
    main()
