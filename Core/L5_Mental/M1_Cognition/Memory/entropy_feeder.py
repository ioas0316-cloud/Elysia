"""
Entropy Feeder: send diverse prompts (or self-generated prompts) to kernel to spread HyperQubit state.

Usage:
    python Tools/entropy_feeder.py
    python Tools/entropy_feeder.py --self-drive
"""

import argparse
import os
import sys
import random

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Core.L6_Structure.M5_Engine.Governance.System.System.System.Kernel import kernel  # noqa: E402

prompts = [
    "                        ?",
    "mountains and oceans,           ?",
    "friendship and conflict               ?",
    "logic and intuition                ?",
    "playfulness and discipline              ?",
    "history and future             ",
    "silence and music               ?",
    "fear and courage                 ?",
    "technology and empathy            ?",
    "dreams and reality                   ?",
]


def self_generate_prompts(n: int) -> list:
    """Let Elysia generate prompts based on capability deficits."""
    caps = getattr(kernel, "capabilities", None)
    generated = []
    if caps:
        deficits = caps.deficits(threshold=0.7)
        for _ in range(n):
            if deficits:
                rec = random.choice(deficits)
                generated.append(f"{rec.name}                 .")
            else:
                generated.append("              ?")
    else:
        generated = ["                ."] * n
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-drive", action="store_true", help="Let Elysia generate prompts based on deficits")
    parser.add_argument("--count", type=int, default=10, help="Number of prompts when self-drive enabled")
    args = parser.parse_args()

    batch = self_generate_prompts(args.count) if args.self_drive else prompts
    for p in batch:
        print("Q:", p)
        ans = kernel.process_thought(p)
        print("A:", ans)
        print("-" * 40)


if __name__ == "__main__":
    main()
