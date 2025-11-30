
import sys
import os
import logging

# Ensure Core is in path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestLagrangian")

from Core.Mind.lagrangian import LagrangianSelector
from Core.Mind.spiderweb import Spiderweb

# Mock Hippocampus for testing
class MockHippocampus:
    def get_related_concepts(self, concept):
        # Define a test graph
        # Start: "Chaos"
        # Path A: Chaos -> Destruction (High V) -> End
        # Path B: Chaos -> Order (Low V) -> End
        if concept == "Chaos":
            return {"Destruction": 1.0, "Order": 1.0}
        elif concept == "Destruction":
            return {"End": 1.0}
        elif concept == "Order":
            return {"End": 1.0}
        return {}

def test():
    logger.info("⚖️ Testing Principle of Least Action...")
    
    # 1. Test Lagrangian Selector directly
    selector = LagrangianSelector()
    
    # Calculate Action for "Chaos -> Destruction" (High V)
    # Destruction is not in core values, so V=1.0
    action_bad = selector.calculate_action("Chaos", "Destruction")
    logger.info(f"Action(Chaos -> Destruction): {action_bad:.2f}")
    
    # Calculate Action for "Chaos -> Love" (Low V)
    # Love is in core values, V=0.0
    action_good = selector.calculate_action("Chaos", "love")
    logger.info(f"Action(Chaos -> love): {action_good:.2f}")
    
    # Verify Good Action < Bad Action
    if action_good < action_bad:
        logger.info("✅ Least Action Principle holds: Love is preferred over Destruction.")
    else:
        logger.error("❌ Failed: Destruction was preferred!")

    # 2. Test Spiderweb Traversal
    logger.info("\n🕸️ Testing Spiderweb Traversal...")
    spiderweb = Spiderweb(hippocampus=MockHippocampus())
    
    # Inject core values into mock if needed, but Lagrangian has its own list.
    # We want to see if it picks "Order" over "Destruction" if "Order" has lower V.
    # Wait, "Order" is not in the default core values list in lagrangian.py.
    # Let's add it for the test or use a known value.
    
    # Let's use "connection" (known value) vs "isolation" (unknown)
    class MockHippocampus2:
        def get_related_concepts(self, concept):
            if concept == "Start":
                return {"connection": 1.0, "isolation": 1.0}
            return {}
            
    spiderweb.hippocampus = MockHippocampus2()
    
    # Traverse from Start
    # Should pick "connection" because it has V=0.2 (low) vs "isolation" V=1.0 (high)
    path = spiderweb.traverse("Start", steps=1)
    logger.info(f"Path chosen: {path}")
    
    if "connection" in path:
        logger.info("✅ Spiderweb followed the Path of Least Action.")
    else:
        logger.error("❌ Spiderweb chose the path of High Action.")

if __name__ == "__main__":
    test()
