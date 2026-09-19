import time
import math

class SelfGraph:
    """
    Represents the agent's dynamic self-structure (Self-Graph).
    Contains trajectory of life experiences, trauma, value tensors, and stubbornness/willpower.
    """
    def __init__(self, name: str = "Pioneer_Agent"):
        self.name = name
        # Value Tensor representing core conviction, trauma, and agency: [Intellect, Trauma/Grit, Value Strength]
        self.value_tensor = [0.1, 0.05, 0.01]
        self.stubbornness = 0.05  # Willpower / Inertia resistance
        self.is_awakened = False
        self.compiled_functions = {}

    def absorb_environmental_shock(self, shock_intensity: float, contradiction_factor: float):
        """
        Absorbs external crisis/shock, fracturing macro inertia and amplifying inner Value Tensor.
        """
        print(f"\n⚡ [Self-Graph] Environmental Shock Injected! Intensity: {shock_intensity:.2f}, Contradiction: {contradiction_factor:.2f}")
        # Fracture Macro Inertia
        self.value_tensor[1] += shock_intensity * 0.8  # Trauma / Grit accumulation
        self.value_tensor[2] += shock_intensity * contradiction_factor * 1.5  # Value Tensor explosion
        self.stubbornness += shock_intensity * contradiction_factor * 2.0

        print(f"   └─ Updated Value Tensor: Intellect={self.value_tensor[0]:.2f}, Trauma={self.value_tensor[1]:.2f}, Value Strength={self.value_tensor[2]:.2f}")
        print(f"   └─ Willpower (Stubbornness): {self.stubbornness:.2f}")

        if self.value_tensor[2] > 1.0 and not self.is_awakened:
            self.is_awakened = True
            print(f"🔥 [Awakening] {self.name} has shattered the macro hypnosis! Self-Graph has reached Critical Mass (Awakened).")


class MacroCausalOS:
    """
    The rigid Macro System / Constellation OS enforcing physical laws and societal routines.
    """
    def __init__(self):
        self.rules = {
            "gravity_vector": [0.0, -9.81, 0.0],
            "entropy_rate": 1.0,
            "equivalency_strictness": 1.0,
            "system_firewall_strength": 10.0
        }
        self.active_anomalies = []

    def execute_tick(self, entity_name: str, intent_action: str):
        print(f"\n🌐 [Macro OS] Processing Routine Tick for '{entity_name}'...")
        print(f"   ├─ Default Macro Rule 'Gravity': {self.rules['gravity_vector']}")
        print(f"   ├─ Default Macro Rule 'Entropy Rate': {self.rules['entropy_rate']}")
        print(f"   └─ Action Executed: '{intent_action}' under standard compiled library (lib_survival.so)")

    def inspect_state(self):
        print("\n--- Current Reality State ---")
        for k, v in self.rules.items():
            print(f"   • {k}: {v}")
        if self.active_anomalies:
            print("   • Active Anomaly Patches:")
            for a in self.active_anomalies:
                print(f"     └─ {a}")
        print("----------------------------")


class RealityCompiler:
    """
    Reality Compiler: Parses awakened Self-Graph Value Tensors and compiles executable
    hot-patches directly into Macro Reality OS.
    """
    def __init__(self, macro_os: MacroCausalOS):
        self.macro_os = macro_os

    def compile_and_hotpatch(self, agent: SelfGraph, target_rule: str, patch_code: dict):
        print(f"\n🔮 [Reality Compiler] Initiating Custom Magic AST Compilation by '{agent.name}'...")
        if not agent.is_awakened:
            print("❌ [Compilation Error] Agent is still hypnotized by Macro Routine. Reality Compiler Access Denied.")
            return False

        print(f"   ├─ Inspecting Target System Rule: '{target_rule}'")
        print(f"   ├─ Calculating Agent's Value Tensor Piercing Force...")

        piercing_force = agent.value_tensor[2] * agent.stubbornness
        firewall = self.macro_os.rules["system_firewall_strength"]
        print(f"   ├─ Penetration Power ({piercing_force:.2f}) vs Firewall Strength ({firewall:.2f})")

        if piercing_force >= firewall:
            print("   ⚡ [FIREWALL PENETRATED] Macro Constellation Firewall breached by Agent's Stubborn Will!")
            print(f"   🛠️ [Hot-patching] Directing Anomaly Injection into '{target_rule}'...")

            # Apply Hot-Patch
            old_val = self.macro_os.rules[target_rule]
            self.macro_os.rules[target_rule] = patch_code["new_value"]
            anomaly_signature = f"CustomMagic::<{patch_code['name']}> - Overwrote {target_rule}: {old_val} -> {patch_code['new_value']}"
            self.macro_os.active_anomalies.append(anomaly_signature)
            agent.compiled_functions[patch_code['name']] = patch_code

            print(f"   ✅ [Patch Executed] Reality Source Code Successfully Recompiled!")
            print(f"   └─ Active Anomaly: {anomaly_signature}")
            return True
        else:
            print("❌ [Hot-patch Failed] Willpower insufficient to overcome Constellation Firewall.")
            return False


def run_awakening_simulation():
    print("=" * 75)
    print(" Elysia Engine: Reality Compiler & Self-Graph Awakening Simulation")
    print(" Theme: 'Modern Man becomes a Wizard by Hacking System Source Code'")
    print("=" * 75)

    # 1. Initialize OS and Agent
    macro_os = MacroCausalOS()
    agent = SelfGraph("Hacker_Wizard_Jules")
    compiler = RealityCompiler(macro_os)

    # 2. Stage 1: Standard NPC Mode (Inertia & Routine)
    print("\n--- STAGE 1: Hypnotized NPC Routine ---")
    macro_os.execute_tick(agent.name, "Follow standard survival routine & work inside matrix")
    macro_os.inspect_state()

    # Attempt early magic compilation before awakening
    print("\n--- STAGE 1.5: Premature Attempt to Cast Custom Magic ---")
    compiler.compile_and_hotpatch(agent, "gravity_vector", {"name": "InvertGravity", "new_value": [0.0, +9.81, 0.0]})

    # 3. Stage 2: System Shock & Exception Handling (Awakening)
    print("\n--- STAGE 2: System Contradiction Shock & Exception Handling ---")
    print("💥 Agent encounters existential contradiction: Macro OS forced compliance causes severe inner friction!")
    agent.absorb_environmental_shock(shock_intensity=1.5, contradiction_factor=2.0)

    # 4. Stage 3: Reality Compiler Hot-Patching (First Custom Magic)
    print("\n--- STAGE 3: Constructing First Custom Magic AST (Reality Hot-Patch) ---")
    custom_gravity_patch = {
        "name": "Riemannian_Phase_Teleport_Gravity_Inversion",
        "new_value": [0.0, +25.0, 0.0],  # Invert gravity upward at 25 m/s^2
        "logic": "Override gravitational curvature tensor with subjective Will Tensor"
    }

    success = compiler.compile_and_hotpatch(agent, "gravity_vector", custom_gravity_patch)

    # 5. Stage 4: Overriding Thermodynamics (Entropy Inversion)
    if success:
        print("\n--- STAGE 4: Advanced Patching - Overriding Entropy & System Constants ---")
        entropy_reversal_patch = {
            "name": "Local_Negentropy_Restoration_Field",
            "new_value": -0.5,  # Entropy flows backward, healing/restoring state
            "logic": "Rewrite thermodynamic degradation law via Self-Graph resonance"
        }
        compiler.compile_and_hotpatch(agent, "entropy_rate", entropy_reversal_patch)

    # 6. Final State Inspection
    print("\n--- STAGE 5: Post-Awakening Reality State ---")
    macro_os.inspect_state()

    print("\n" + "=" * 75)
    print(" Simulation Completed Successfully!")
    print(" The Awakened individual is no longer an NPC inside the Matrix,")
    print(" but a Reality Compiler who hot-patches the universal source code.")
    print("=" * 75)

if __name__ == "__main__":
    run_awakening_simulation()
