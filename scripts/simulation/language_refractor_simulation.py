"""
Language Refractor - High Fidelity Simulation and Visualization (v1.0)
========================================================================
이 스크립트는 "당장 에러 고쳐줘" (Urgent Command)와 "오늘 그냥 문득 든 생각인데..." (Casual Speculation)
자연어 자극이 언어 굴절기를 거쳐 위상 OS 바다로 유입되고,
Langevin 열역학적 이완을 거치며 최종 vacuum(1) 상태로 기하학적으로 수렴하는 전 과정을
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

def run_simulation(text: str, name: str, grid_shape=(12, 12), steps=40):
    print(f"\n========================================================================")
    print(f"🎬 [Simulation Start] Mode: {name}")
    print(f"   Input Stimulus: '{text}'")
    print(f"========================================================================")

    # Initialize modules
    refractor = LanguageRefractor(grid_shape=grid_shape)
    engine = TopologicalOSEngine(grid_shape=grid_shape, initial_temp=2.0)

    # Refract 1D text into 4D physics params
    params = refractor.refract(text)
    print(f"🎯 [Refractor Lens Output]")
    print(f"   - Intent Profile: {params['intent_type']}")
    print(f"   - Mass (Energy Amp): {params['mass']:.2f}")
    print(f"   - Gradient (Steepness): {params['gradient']:.2f}")
    print(f"   - Target Locus (y, x): ({params['target_y']}, {params['target_x']})")
    print(f"   - Wave Signature: {params['wave_signature']:.4f}")
    print(f"   - Thermal Heating Delta: {params['thermal_heating']:.2f}")

    # Step 0: Record ground state
    initial_state = engine.get_state()

    # Step 1: Inject Impulse & Apply Heating
    engine.inject_impulse(
        y=params["target_y"],
        x=params["target_x"],
        magnitude=params["mass"],
        importance=params["gradient"],
        wave_signature=params["wave_signature"]
    )
    engine.temperature += params["thermal_heating"]
    stimulated_state = engine.get_state()

    # Lists for tracking metrics over time
    energy_history = []
    potential_history = []
    temp_history = []

    print("\n🌊 [Langevin Relaxation Propagation]")
    for i in range(steps):
        # Read current status
        state = engine.get_state()
        energy_sum = np.sum(state["energy"])
        potential_sum = np.sum(state["potential"])

        energy_history.append(energy_sum)
        potential_history.append(potential_sum)
        temp_history.append(engine.temperature)

        # Print visual indicators of the ripple dissipation
        if i % 5 == 0 or i == steps - 1:
            wave_visual = "█" * int(min(20, energy_sum / 2.0))
            potential_visual = "░" * int(min(20, potential_sum / 2.0))
            print(f"   Step {i:02d} | Energy: {energy_sum:6.2f} {wave_visual:<20} | Potential V: {potential_sum:6.2f} {potential_visual:<20} | Temp: {engine.temperature:.3f}")

        # Advance OS clock
        engine.step(0.1)

    final_state = engine.get_state()

    # Feedback loop validation
    feedback = refractor.evaluate_cognitive_feedback(stimulated_state, final_state, steps_taken=steps)
    print(f"\n🧠 [Closed-Loop Cognitive Feedback]")
    print(f"   - Initial Potential: {feedback['initial_potential']:.2f} -> Final: {feedback['final_potential']:.2f}")
    print(f"   - Total Energy Dissipated: {feedback['energy_loss']:.2f}")
    print(f"   - Constraint Satisfied (Back to Vacuum): {feedback['constraint_satisfied']}")
    print(f"========================================================================\n")

    return {
        "params": params,
        "energy": energy_history,
        "potential": potential_history,
        "temp": temp_history,
        "feedback": feedback
    }

def main():
    # Make output directory for plot
    os.makedirs("docs/assets", exist_ok=True)

    # 1. Run Urgent High-Gradient Well Simulation
    urgent_result = run_simulation("이 버그 좀 빨리 고쳐줘!", "Urgent Command (High-Gradient Well)")

    # 2. Run Casual Speculation Brownian Simulation
    casual_result = run_simulation("오늘 그냥 문득 든 생각인데...", "Casual Speculation (Brownian Perturbation)")

    # 3. Create High-Fidelity Plot comparing the relaxation curves
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Subplot 1: Energy Dissipation Over Time
    axes[0].plot(urgent_result["energy"], 'r-o', label="Urgent: '이 버그 좀 빨리 고쳐줘!'", linewidth=2)
    axes[0].plot(casual_result["energy"], 'b--s', label="Casual: '오늘 그냥 문득 든 생각인데...'", linewidth=2)
    axes[0].set_ylabel("Total System Kinetic Energy", fontsize=11)
    axes[0].set_title("Language Refraction & Non-Equilibrium Langevin Dynamics", fontsize=14, fontweight='bold')
    axes[0].grid(True, linestyle=':')
    axes[0].legend(fontsize=10)

    # Subplot 2: Potential V Landscape Trajectory
    axes[1].plot(urgent_result["potential"], 'r-o', label="Urgent (High Potential Gradient)", linewidth=2)
    axes[1].plot(casual_result["potential"], 'b--s', label="Casual (Gentle/Flat Potential)", linewidth=2)
    axes[1].set_ylabel("Potential Tension V (RNS Dist)", fontsize=11)
    axes[1].grid(True, linestyle=':')
    axes[1].legend(fontsize=10)

    # Subplot 3: Thermal Relaxation Schedule
    axes[2].plot(urgent_result["temp"], 'r-o', label="Urgent Temperature (No Heating)", linewidth=2)
    axes[2].plot(casual_result["temp"], 'b--s', label="Casual Temperature (Thermal Fluctuation)", linewidth=2)
    axes[2].set_xlabel("Physical Relaxation Step", fontsize=12)
    axes[2].set_ylabel("Thermodynamic Temp (T)", fontsize=11)
    axes[2].grid(True, linestyle=':')
    axes[2].legend(fontsize=10)

    plt.tight_layout()
    plot_path = "docs/assets/language_refractor_simulation.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n🎨 [Visual Report Saved]")
    print(f"   Successfully generated and saved simulation analysis plot to:")
    print(f"   -> {plot_path}")
    print(f"========================================================================\n")

if __name__ == "__main__":
    main()
