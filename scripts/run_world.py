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


def main():
    parser = argparse.ArgumentParser(description="Run time-accelerated Cell World (Starter)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--render", type=int, default=0)
    parser.add_argument("--dt", type=float, default=1.0, help="simulation time delta per step (time acceleration)")
    args = parser.parse_args()

    starter = _resolve_starter_run()
    starter.run(steps=args.steps, render=bool(args.render), dt=args.dt)


if __name__ == "__main__":
    main()

