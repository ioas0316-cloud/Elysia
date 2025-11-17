import argparse
import os
import sys


def _run_local_world(steps: int = 1000):
    """Fallback: run the local Project_Sophia core.World without ElysiaStarter.

    Creates a minimal World with WaveMechanics + KG and runs a few steps,
    printing summaries to the console.
    """
    # Ensure project root is on sys.path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from tools.kg_manager import KGManager
    from Project_Sophia.wave_mechanics import WaveMechanics
    from Project_Sophia.core.world import World

    kgm = KGManager()
    wm = WaveMechanics(kg_manager=kgm)
    world = World(
        primordial_dna={
            'instinct': 'connect_create_meaning',
            'resonance_standard': 'love',
        },
        wave_mechanics=wm,
    )

    print(f"[Local Cell World] Running {steps} steps...")
    for i in range(int(steps)):
        print(f"\n--- Step {i+1}/{steps} ---")
        try:
            newborn = world.run_simulation_step()
        except TypeError:
            # Older signature without return; ignore
            newborn = []
            world.run_simulation_step()
        try:
            world.print_world_summary()
        except Exception:
            pass
        if newborn:
            try:
                ids = [c.id for c in newborn if hasattr(c, 'id')]
            except Exception:
                ids = list(newborn)
            print(f"Newborn cells: {ids}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the Project_Sophia.core.World loop (legacy ElysiaStarter client is archived)."
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=1.0, help="simulation time delta per step")
    args = parser.parse_args()

    _run_local_world(steps=args.steps)


if __name__ == "__main__":
    main()
