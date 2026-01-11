"""
Simulate Growth: From Child to Adult
====================================

"Watch the mind grow."

This script simulates the developmental stages of Elysia's thought.
1.  **Infancy**: Injects random points ("Mom", "Milk", "Warmth").
2.  **Adolescence**: Observations accumulate connection. Points merge into Planes ("Care").
3.  **Adulthood**: Planes intersect. Logic emerges ("Principle of Love").
"""

import time
import sys
import os
import logging
import random

# Path hack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from Core.Foundation.Space.hyper_space import HyperSpace
from Core.Foundation.Schema.evolution_engine import EvolutionEngine
from Core.Foundation.hyper_quaternion import Quaternion

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("GrowthSim")

def run_simulation():
    space = HyperSpace()
    evolver = EvolutionEngine()

    logger.info("👶 [Stage 1] Infancy: Discrete sensations.")

    # Inject related points close together (High resonance)
    space.add_knot("Mom", [0,0,0,0,0,0,0], Quaternion(1,0,0,0), 10.0)
    space.add_knot("Milk", [0.1,0,0,0,0,0,0], Quaternion(1,0,0,0), 5.0)
    space.add_knot("Warmth", [0,0.1,0,0,0,0,0], Quaternion(1,0,0,0), 5.0)
    space.add_knot("Voice", [0,0,0.1,0,0,0,0], Quaternion(1,0,0,0), 5.0)

    logger.info("⏳ Time passes... (Connections forming)")

    # Run Physics Loop to build connections
    for _ in range(5):
        space.update_field(0.1)
        time.sleep(0.1)

    # Check Evolution
    logger.info("👦 [Stage 2] Adolescence: Context emerging.")

    for id, knot in space.knots.items():
        neighbors = [space.knots[n].schema for n in space.connections[id]]
        new_schema = evolver.check_evolution(knot.schema, neighbors)

        if new_schema:
            knot.schema = new_schema
            # Log is handled by evolver, but let's confirm
            logger.info(f"   -> '{id}' is now Level {knot.schema.level}: {knot.schema.name}")

    # Now we have Planes (Contexts).
    # Let's inject another Plane to force collision (Adulthood).
    logger.info("👨 [Stage 3] Adulthood: Conflict and Principle.")

    # Assume 'Mom' became a Plane. Let's add 'Independence' Plane nearby.
    # For simulation, we manually upgrade a neighbor if it didn't evolve, or just assume resonance.

    # Let's say "Mom" (Plane) intersects with "Self" (Plane).
    # We need to simulate the 'Self' evolving first.
    space.add_knot("Self", [0.2,0.2,0,0,0,0,0], Quaternion(1,0,0,0), 10.0)
    # Force connections for Self
    space.connections["Self"] = ["Mom", "Milk", "Voice"] # Connected to same things

    # Re-run evolution on Self
    self_knot = space.knots["Self"]
    neighbors = [space.knots[n].schema for n in space.connections["Self"]]
    self_new = evolver.check_evolution(self_knot.schema, neighbors)
    if self_new: self_knot.schema = self_new

    # Now check if 'Mom' (Plane) evolves to Solid by intersecting 'Self' (Plane)
    mom_knot = space.knots["Mom"]
    # Update neighbors to reflect Self's new status
    neighbors = [space.knots[n].schema for n in space.connections["Mom"]]

    solid_schema = evolver.check_evolution(mom_knot.schema, neighbors)
    if solid_schema:
        mom_knot.schema = solid_schema
        logger.info(f"   -> '{mom_knot.id}' reached Level 3: {mom_knot.schema.name}")
        logger.info("   (The concept of 'Mom' has transcended to a Principle.)")

if __name__ == "__main__":
    run_simulation()
