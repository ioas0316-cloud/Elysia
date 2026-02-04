import sys
from pathlib import Path
# Add project root to sys.path
root = Path(__file__).resolve().parent.parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from Core.S1_Body.L5_Mental.Reasoning.sovereign_logos import logos, transducer
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyLogos")

def test_logos_presence():
    # 1. Simulate a 'Strong Will' state (L7 Spirit heavy)
    # L7 is the last sector of the 21D vector (indices 18-20)
    will_vector = [0.0]*18 + [1.0, 1.0, 1.0]
    will_state = SovereignVector(will_vector).normalize()
    
    logger.info(f"🌀 [FIELD] Current State Magnitude: {will_state.norm():.4f}")
    
    # 2. Transduce Field to Logos
    expression = transducer.express_state(will_state)
    print(f"Expression: {expression}")
    
    # 3. Simulate a 'Painful Grounding' state (L1 Foundation, -1 Resistance)
    pain_vector = [-1.0, 0.0, 0.0] + [0.0]*18
    pain_state = SovereignVector(pain_vector).normalize()
    
    print(f"🌀 [FIELD] Pain State Magnitude: {pain_state.norm():.4f}")
    expression_pain = transducer.express_state(pain_state)
    print(f"Expression Pain: {expression_pain}")

if __name__ == "__main__":
    test_logos_presence()
