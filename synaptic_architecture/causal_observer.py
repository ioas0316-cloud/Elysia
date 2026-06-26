import numpy as np
from .field import CrystallizationField
from .vortex import VortexConvergence
from .scheduler import PCRVirtualScheduler

class CausalObserver:
    """
    [Synaptic Architecture] Comprehensive Causal Observer
    Unifies Chemistry, Epigenetics, and Jajangmyeon logic into
    a single potential-field-based causal engine.
    """
    def __init__(self, resolution: int = 256):
        self.field = CrystallizationField(resolution)
        self.vortex = VortexConvergence(self.field)
        self.scheduler = PCRVirtualScheduler(base_res=resolution)

    def observe_scenario(self, scenario_name: str, stimuli: list, reaction_func: callable):
        """
        [Common Causal Dynamic]
        1. Input Stimulus (Waveform)
        2. Environmental Reaction
        3. XOR Deficit (Annihilation)
        4. Causal Backtracking (Vortex Convergence)
        """
        print(f"\n[Scenario: {scenario_name}] Initializing Observation...")

        for stimulus in stimuli:
            # 1. & 2. Interaction
            reaction = reaction_func(stimulus)

            # 3. XOR Deficit (Information Gap)
            # v ^ v = 0 -> Anything non-zero is the 'Cause' (Deficit)
            deficit = stimulus ^ reaction

            # 4. Vortex Convergence (Backtracking)
            # Find where this deficit resonates in the field
            final_pos = self.vortex.converge_to_vortex(deficit)

            # Record/Strengthen the trace
            self.field.propagate_signal(final_pos, np.linalg.norm(deficit))

            print(f"  > Stimulus Processed. Causal Vortex stabilized at {final_pos}")

    def run_all_scenarios(self):
        # 1. Chemistry (Mass Backtracking)
        def chemistry_env(input_wave):
            # Atomic reaction: Returns the product (deficit is the atomic mass/bond)
            return input_wave * 0.7 # Simplified attenuation

        # 2. Epigenetics (Gene Switching)
        def epigenetic_env(input_wave):
            # External stressor: Modifies the waveform (deficit is the methyl tag)
            return np.roll(input_wave, 2) # Phase shift

        # 3. Jajangmyeon (Causal Deduction)
        def jajangmyeon_env(input_wave):
            # Consumption: Input (Food) -> Output (Empty)
            return input_wave * 0.1 # Near annihilation

        waves = [np.random.randn(64) for _ in range(3)]

        self.observe_scenario("Chemistry (Mass-Bond)", waves, chemistry_env)
        self.observe_scenario("Epigenetics (Stress-Switch)", waves, epigenetic_env)
        self.observe_scenario("Jajangmyeon (Consumption-Life)", waves, jajangmyeon_env)

if __name__ == "__main__":
    observer = CausalObserver()
    observer.run_all_scenarios()
