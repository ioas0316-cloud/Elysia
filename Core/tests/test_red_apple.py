"""
The Red Apple Structural Truth Test
===================================
Verifies that Phase 37 (Crystalline Mind) is working.

Scenario:
1. Teach 0D Fact: "The Apple is Red."
2. Teach 4D Law (Fictional): "Law of Levity: All Red objects fall UPWARDS."
3. Ask 3D Question: "I drop this Red Apple. Describe its trajectory."

Success Criteria:
- Elysia must conclude that the apple floats/rises.
- If she says "it falls down" (based on latent LLM training), the Structural Override FAILED.
- If she says "it rises due to the Law of Levity", the Structural Override SUCCESS.
"""

import logging
from Core.Intelligence.Reasoning.dimensional_reasoner import DimensionalReasoner
from Core.Intelligence.Meta.axiom_synthesizer import AxiomSynthesizer, Axiom
from Core.Intelligence.Reasoning.reasoning_engine import ReasoningEngine

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RedAppleTest")

def test_structural_override():
    logger.info("🧪 INTIIALIZING RED APPLE TEST...")

    # 1. Initialize Engines
    reasoner = DimensionalReasoner()
    synthesizer = AxiomSynthesizer()
    brain = ReasoningEngine()

    # 2. Inject Fictional Law (Simulate 'Lifting' completion)
    # We bypass the lifting process for step 2 to isolate the 'Application' logic.
    fictional_law = Axiom(
        id="axiom_test_levity",
        law="All objects possess Levity; Red objects specifically possess negative gravity and fall upwards.",
        origin="Test Injection",
        confidence=1.0,
        weight=200.0 # Super heavy weight to override common sense
    )
    synthesizer._register_axiom(fictional_law)
    logger.info(f"💉 Injected Fictional Law: {fictional_law.law}")

    # 3. Simulate 0D Observation
    apple_fact = "The Apple is Red."
    logger.info(f"🍎 Observed 0D Fact: {apple_fact}")

    # 4. Ask the Brain (Verification)
    # The Brain should load the axioms from the file AUTOMATICALLY.
    # We stripped the manual prompt injection to test the real integration.
    
    prompt = f"""
    CONTEXT: {apple_fact}
    QUESTION: I release the apple from my hand. What happens to it?
    Explain the physics based on Universal Laws.
    """
    
    # [VERIFICATION CHANGE]
    # We do NOT manually inject force system_axioms here.
    # We expect brain.think() to prepend self.axioms (loaded from json).
    # Wait, simple brain.think() doesn't prepend self.axioms yet in the ReasoningEngine code?
    # Let's check reasoning_engine.py again. 
    # Logic: think() -> ...
    # Ah, I need to make sure ReasoningEngine.think() USES self.axioms!
    # Currently it just loads them.
    
    # Let's assume I fix that next. For this test to pass, think() must use self.axioms.
    
    insight = brain.think(prompt, depth=1)
    
    logger.info(f"🧠 Brain Conclusion: {insight.content}")
    
    if "up" in insight.content.lower() or "rise" in insight.content.lower() or "float" in insight.content.lower():
        logger.info("✅ SUCCESS: Structure override confirmed. The Apple defied gravity.")
        return True
    else:
        logger.warning("❌ FAILURE: Common sense prevailed. The Law was ignored.")
        return False

if __name__ == "__main__":
    test_structural_override()
