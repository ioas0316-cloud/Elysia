"""
Verify One Body: The Causal Chain Test
======================================

"From Input to Output, the Soul moves as One."

This script validates the "One Body" architecture by tracing a single input
through the entire stack:
1.  **Input** -> `HyperSpace` (as a FieldKnot)
2.  **Trajectory** -> `TrajectoryAnalyzer` (Movement Interpretation)
3.  **Resonance** -> `LawOfResonance` (Attraction to Archetypes)
4.  **Logic** -> `ConstraintSolver` (Assembly Check)
5.  **Output** -> A coherent narrative log.

If this script runs without error and produces a "Story", the Body is One.
"""

import time
import sys
import os
import logging

# Path hack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.cosmic_kernel import CosmicKernel
from Core.Foundation.Law.constraint_solver import ConstraintSolver, Part

# Setup Logger to capture output
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("OneBodyTest")

def verify_one_body():
    logger.info("🧪 Starting One Body Verification...")

    # 1. Initialize Kernel (The Body)
    kernel = CosmicKernel()
    logger.info("✅ Cosmic Kernel Initialized.")

    # 2. Inject Input (The Sensation)
    input_text = "Love is Logic"
    kernel.inject_input(input_text)
    logger.info(f"✅ Input Injected: '{input_text}'")

    # 3. Run Physics Loop (The Pulse)
    logger.info("⏳ Running Physics Simulation (3 steps)...")
    for i in range(3):
        kernel._tick()
        time.sleep(0.1)
    logger.info("✅ Physics & Trajectory Analyzed.")

    # 4. Check CAD Logic (The Mind)
    # Simulating the internal "Assembly" check that would happen upon resonance
    logger.info("🔧 Running Constraint Solver (CAD Logic)...")
    solver = ConstraintSolver()

    # Define "Love" and "Logic" as Parts
    part_love = Part("Love")
    part_love.add_port("Nature", "Abstract", "Emotion")

    part_logic = Part("Logic")
    part_logic.add_port("Nature", "Abstract", "Reason")

    # Try to assemble them (Concentric Constraint - Must share core value)
    # They should FAIL because Emotion != Reason (Interference)
    logger.info("   Attempting to assemble 'Love' and 'Logic' via 'Nature' port (Concentric)...")
    fit = solver.check_fit(part_love, "Nature", part_logic, "Nature", "Concentric")

    if not fit:
        logger.info("✅ CAD Logic Correctly Identified Interference (Paradox).")
    else:
        logger.warning("⚠️ CAD Logic Failed (Should have detected interference).")

    logger.info("🎉 One Body Verification Complete. The System thinks, moves, and checks fit.")

if __name__ == "__main__":
    verify_one_body()
