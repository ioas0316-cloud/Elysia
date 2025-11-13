import argparse
import importlib.util
import os
import sys


def _resolve_starter_run():
    """Locate and load the Starter run_world module."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    starter_scripts = os.path.join(repo_root, 'ElysiaStarter', 'ElysiaStarter', 'scripts')
    starter_module_path = os.path.join(starter_scripts, 'run_world.py')

    if not os.path.isfile(starter_module_path):
        raise FileNotFoundError(f"Starter script not found: {starter_module_path}")

    spec = importlib.util.spec_from_file_location("starter_run_world", starter_module_path)
    mod = importlib.util.module_from_spec(spec)
    # Ensure Starter 'core' can be imported by its sibling path
    sys.path.insert(0, os.path.join(repo_root, 'ElysiaStarter', 'ElysiaStarter'))
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


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
    parser = argparse.ArgumentParser(description="Run time-accelerated Cell World (Starter)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--render", type=int, default=0)
    parser.add_argument("--dt", type=float, default=1.0, help="simulation time delta per step (time acceleration)")
    args = parser.parse_args()

    try:
        starter = _resolve_starter_run()
        starter.run(steps=args.steps, render=bool(args.render), dt=args.dt)
    except FileNotFoundError:
        _run_local_world(steps=args.steps)


if __name__ == "__main__":
    main()
