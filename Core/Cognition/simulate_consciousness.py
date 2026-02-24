"""
Fractal Consciousness Simulation

                           
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

from Core.System.Mind.fractal_consciousness import FractalConsciousness


def simulate():
    """             """
    
    print("\n" + "="*70)
    print("  ELYSIA FRACTAL CONSCIOUSNESS SIMULATION")
    print("="*70)
    print("\n                           .\n")
    
    consciousness = FractalConsciousness()
    
    # Test inputs
    test_cases = [
        "    ?",
        "        ",
        "      ?",
        "         ?",
    ]
    
    for test_input in test_cases:
        result = consciousness.process(test_input)
        print()
        input("Press Enter for next simulation...")
        print("\n")
    
    print("="*70)
    print("        !  ")
    print("="*70)


if __name__ == "__main__":
    simulate()
