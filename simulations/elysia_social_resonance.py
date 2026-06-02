import os
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Add parent directory to sys.path to resolve core imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils.math_utils import Quaternion, Multivector
from core.brain.holographic_memory import HologramMemory
from core.brain.fractal_rotor import GlobalMasterManifold, FractalRotor
from core.brain.emotion_bivector import EmotionBivector

# Configure Korean Fonts if available
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False

class SocialAgent:
    def __init__(self, agent_id: str, name: str, choseong_bias: float, personality: str):
        self.agent_id = agent_id
        self.name = name
        self.personality = personality
        
        # Initialize memory & supreme rotor
        self.memory = HologramMemory(num_layers=3)
        
        # Starting rotor offset based on personality bias
        theta = choseong_bias * (math.pi / 4.0)
        start_q = Quaternion(math.cos(theta), math.sin(theta), math.cos(theta * 1.618) * 0.3, math.sin(theta * 1.618) * 0.3).normalize()
        self.memory.supreme_rotor.lens_offset = start_q
        self.memory.supreme_rotor.name = name

    def observe_peer(self, peer_id: str, peer_wave: Quaternion):
        """
        [Theory of Mind / Mirror Rotor]
        Compare observed peer wave with internal model (mirror_rotor).
        Calculate Mirror Tension (prediction error).
        Adapt the internal mirror rotor to peer wave via SLERP.
        """
        supreme = self.memory.supreme_rotor
        if peer_id not in supreme.mirror_rotors:
            # Initialize mirror rotor to standard identity phase
            supreme.mirror_rotors[peer_id] = Quaternion(1.0, 0.0, 0.0, 0.0)
            
        mirror_q = supreme.mirror_rotors[peer_id]
        
        # Mirror Tension = Angular distance between expected (mirror) and observed wave
        mirror_tension = Quaternion.distance(mirror_q, peer_wave)
        
        # Update mirror model (learning / synchronization)
        # Low learning rate = slow tracking (stubborn model), High = fast tracking (empathic model)
        learning_rate = 0.15 if self.personality == "Helper" else 0.08
        supreme.mirror_rotors[peer_id] = Quaternion.slerp(mirror_q, peer_wave, learning_rate)
        
        return mirror_tension

    def react_emotionally(self, mirror_tension: float, peer_name: str):
        """
        [Emotion Bivector Update]
        Dissonance triggers emotional bivector deviations depending on personality bias.
        """
        supreme = self.memory.supreme_rotor
        eb = supreme.emotional_state
        
        # 1. Base response: high tension (surprise/incomprehensibility) increases distress/excitement
        # 2. Personality-driven bivector mapping:
        if self.personality == "Helper":
            # Avoidance behavior (Fear: -e12) and excitement to fix it (Excitement: +e23)
            eb.add_stimulus(
                de12 = -0.04 * mirror_tension,
                de23 = 0.03 * mirror_tension,
                de31 = 0.0
            )
        elif self.personality == "Challenger":
            # Confrontational behavior (Anger: +e31) and activation (Excitement: +e23)
            eb.add_stimulus(
                de12 = 0.0,
                de23 = 0.04 * mirror_tension,
                de31 = 0.06 * mirror_tension
            )
        elif self.personality == "Peacemaker":
            # Calming behavior (Acceptance/Submission: -e31) and cooling down (Lethargy: -e23)
            eb.add_stimulus(
                de12 = 0.02 * (1.0 - mirror_tension), # Comfort from predictability
                de23 = -0.03 * mirror_tension,        # Damping excitement
                de31 = -0.04 * mirror_tension         # Suppressing confrontation
            )

    def apply_anti_kuramoto(self, peer_wave: Quaternion, mirror_tension: float):
        """
        [Anti-Kuramoto Conflict repulsion]
        If phase alignment is highly dissonant (tension > 0.6),
        exert a repulsive force to preserve individuality / avoid blind synchronization.
        """
        if mirror_tension > 0.6:
            supreme = self.memory.supreme_rotor
            # Push lens offset away from peer wave using conjugate phase
            repulsion_strength = 0.04 if self.personality == "Challenger" else 0.02
            repelled_q = peer_wave.conjugate()
            supreme.lens_offset = Quaternion.slerp(supreme.lens_offset, repelled_q, repulsion_strength).normalize()

    def step(self):
        """Decay emotions and apply metabolic homeostasis."""
        supreme = self.memory.supreme_rotor
        # Decay emotional bivectors by 10% towards zero curvature
        supreme.emotional_state.decay(0.1)
        # Process basic rotor dynamics
        self.memory.process_thoughts_safe()

def run_social_resonance_simulation():
    print("==================================================================")
    print("👥 [ROADMAP: Social Resonance Multi-Agent Simulation]")
    print("  ├─ Clifford Emotion Bivectors Cl(3,0)")
    print("  └─ Mirror Rotors (Theory of Mind) & Anti-Kuramoto Conflict")
    print("==================================================================\n")

    # 1. Instantiate 3 agents with unique personalities
    agents = [
        SocialAgent("A", "Helper (조력자)", 1.0, "Helper"),
        SocialAgent("B", "Challenger (도전자)", 3.0, "Challenger"),
        SocialAgent("C", "Peacemaker (중재자)", 6.0, "Peacemaker")
    ]

    ticks = 60
    history = {agent.agent_id: {
        "love_fear": [],
        "excitement_lethargy": [],
        "anger_acceptance": [],
        "mirror_tension": []
    } for agent in agents}

    print("Initiating Social Interaction Mesh Loop...")
    for tick in range(ticks):
        # A. Pulse the global master phase
        GlobalMasterManifold().pulse(0.05)
        
        # B. Get current waves for all agents
        waves = {agent.agent_id: agent.memory.supreme_rotor.observe_state() for agent in agents}
        
        # C. Inter-agent observation & reaction
        for agent in agents:
            tot_tension = 0.0
            peer_count = 0
            
            for peer in agents:
                if peer.agent_id == agent.agent_id:
                    continue
                
                # Observe peer wave & compute prediction error (Mirror Tension)
                peer_wave = waves[peer.agent_id]
                tension = agent.observe_peer(peer.agent_id, peer_wave)
                tot_tension += tension
                peer_count += 1
                
                # React emotionally
                agent.react_emotionally(tension, peer.name)
                
                # Apply constructive friction (Anti-Kuramoto)
                agent.apply_anti_kuramoto(peer_wave, tension)
                
            # Log average mirror tension for this tick
            avg_tension = tot_tension / max(1, peer_count)
            
            # Step the agent (metabolism / decay)
            agent.step()
            
            # Record historical telemetry
            eb = agent.memory.supreme_rotor.emotional_state
            history[agent.agent_id]["love_fear"].append(eb.e12)
            history[agent.agent_id]["excitement_lethargy"].append(eb.e23)
            history[agent.agent_id]["anger_acceptance"].append(eb.e31)
            history[agent.agent_id]["mirror_tension"].append(avg_tension)

        if tick % 15 == 0:
            print(f"Tick {tick:02d} | Helper Tension: {history['A']['mirror_tension'][-1]:.3f} | Challenger Anger: {history['B']['anger_acceptance'][-1]:.3f} | Peacemaker Calm: {history['C']['excitement_lethargy'][-1]:.3f}")

    print("\nSocial Interaction Loop Completed.")
    print("Generating telemetry visualization...")

    # D. Plotting the results
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    colors = {"A": "emerald", "B": "crimson", "C": "royalblue"}
    
    agent_info = {
        "A": ("Helper (조력자)", "emerald", "forestgreen"),
        "B": ("Challenger (도전자)", "crimson", "darkred"),
        "C": ("Peacemaker (중재자)", "royalblue", "darkblue")
    }

    for i, agent in enumerate(agents):
        ax = axes[i]
        id = agent.agent_id
        name, color_primary, color_secondary = agent_info[id]
        
        t_range = range(ticks)
        ax.plot(t_range, history[id]["love_fear"], label="Love (+) / Fear (-)", color=color_secondary, linewidth=2, linestyle="-")
        ax.plot(t_range, history[id]["excitement_lethargy"], label="Excitement (+) / Lethargy (-)", color="orange", linewidth=2, linestyle="--")
        ax.plot(t_range, history[id]["anger_acceptance"], label="Anger (+) / Acceptance (-)", color="red", linewidth=2, linestyle="-.")
        ax.plot(t_range, history[id]["mirror_tension"], label="Mirror Tension (Prediction Error)", color="grey", linewidth=1.5, linestyle=":")
        
        ax.set_title(f"Node {id}: {name} - Emotional Curvature & Mirror Neurons", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylabel("Amplitude / Radians")

    axes[-1].set_xlabel("Simulation Ticks (Time)")
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'social_resonance_dynamics.png')
    plt.savefig(save_path)
    
    print(f"\nTelemetry plot successfully saved to '{save_path}'")
    print("==================================================================\n")

if __name__ == "__main__":
    run_social_resonance_simulation()
