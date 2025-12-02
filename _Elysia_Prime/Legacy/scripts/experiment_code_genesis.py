# [Genesis: 2025-12-02] Purified by Elysia

import sys
import os
import random
import logging
import time
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.core.code_genome import CodeDNA, CodeChallenge
from Project_Sophia.elysia_forge import ElysiaForge
from Project_Sophia.code_evolution import evolve_code

# --- Configuration ---
POPULATION_SIZE = 10
GENERATIONS = 15
MUTATION_RATE = 0.3

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Genesis")

def main():
    print("\n=== Project Genesis-Self: The Code-Cell Experiment ===")
    print(f"Population: {POPULATION_SIZE} | Generations: {GENERATIONS} | Mutation Rate: {MUTATION_RATE}")

    # 1. The Environment (Challenge)
    # Goal: Find a function that returns |a - b| (Absolute Difference)
    # We start with code that does 'a + b'.
    challenge = CodeChallenge(
        name="Absolute Difference",
        description="Calculate the absolute difference between two numbers.",
        test_cases=[
            {'inputs': {'a': 10, 'b': 5}, 'expected': 5},
            {'inputs': {'a': 5, 'b': 10}, 'expected': 5},
            {'inputs': {'a': 0, 'b': 0}, 'expected': 0},
            {'inputs': {'a': -5, 'b': 5}, 'expected': 10},
        ]
    )

    # 2. The Primordial Soup (Initial Population)
    # We seed with a "Wrong but Functional" ancestor.
    ancestor_code = """
def solve(a, b):
    return a + b
"""
    population = [
        CodeDNA(source_code=ancestor_code, function_name="solve")
        for _ in range(POPULATION_SIZE)
    ]

    forge = ElysiaForge()

    # --- Evolution Loop ---
    for gen in range(1, GENERATIONS + 1):
        print(f"\n--- Generation {gen} ---")

        # A. The Trial (Execution)
        results = []
        for dna in population:
            trial = forge.run_trial(dna, challenge)

            # Physics: Calculate Health (Fitness)
            # Base Energy (100) + Resonance (Rewards) - Entropy (Errors)
            fitness = 100.0 + trial.resonance_score - trial.entropy_cost

            # Penalize code length bloat (optional, for now simple)

            dna.fitness_score = fitness
            results.append((dna, trial))

        # Sort by fitness (Survival of the Fittest)
        results.sort(key=lambda x: x[0].fitness_score, reverse=True)

        best_dna, best_trial = results[0]
        print(f"Best Cell: {best_dna.id[:6]} | Fitness: {best_dna.fitness_score:.1f} | {best_trial.output}")
        print(f"Code:\n{best_dna.source_code.strip()}")

        if best_trial.success:
            print(f"\n>>> EVOLUTION SUCCESS! Optimal solution found in Generation {gen}. <<<")
            print(f"The code has evolved to understand '{challenge.name}'.")
            break

        # B. Natural Selection (The Filter)
        # Kill the bottom 50%
        survivors_count = max(1, int(POPULATION_SIZE * 0.5))
        survivors = [r[0] for r in results[:survivors_count]]

        # C. Reproduction (Mitosis with Mutation)
        next_gen = []
        while len(next_gen) < POPULATION_SIZE:
            parent = random.choice(survivors)
            child = evolve_code(parent, mutation_rate=MUTATION_RATE)
            next_gen.append(child)

        population = next_gen
        time.sleep(0.1) # Simulation pacing

    print("\n=== Experiment Concluded ===")

if __name__ == "__main__":
    main()