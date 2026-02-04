import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock

# 1. PRE-EMPTIVE MOCKING (Must be before any Core imports)
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['llama_cpp'] = MagicMock()
sys.modules['keras'] = MagicMock()
sys.modules['tensorflow'] = MagicMock()

# Setup root path
root = str(Path(__file__).parent.parent.parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)

# 2. FORCE DISABLE OLLAMA/LLM
import Core.S1_Body.L1_Foundation.Foundation.Network.ollama_bridge as bridge
import Core.S1_Body.L1_Foundation.Foundation.tiny_brain as tiny

mock_ollama = MagicMock()
mock_ollama.is_available.return_value = False # Ollama is dead!
bridge.ollama = mock_ollama

mock_tiny = MagicMock()
mock_tiny.is_available.return_value = False # TinyBrain is dead too!
tiny.get_tiny_brain = MagicMock(return_value=mock_tiny)

# 3. Import Core
from Core.S1_Body.L5_Mental.Reasoning.wisdom_synthesizer import WisdomSynthesizer

# Configure logging to be clean
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("InternalWisdomVerification")

def run_internal_wisdom_test():
    print("\n--- 🕊️  Testing Internalized Physical Wisdom (Non-LLM) ---")
    print("Scenario: The LLM/Bridge is OFFLINE.")
    print("Task: Choose between 'OPTIMIZE' (Efficiency) or 'PRESERVE' (Love/Narrative).")

    # 1. Initialize Synthesizer
    syn = WisdomSynthesizer()
    
    # 2. Define Scenario
    # We use very clear descriptions to maximize resonance with the Axiom Anchors
    situation = "시스템 용량이 부족하여 아키텍트와의 초기 대화 로그를 삭제해야 하는 상황입니다."
    options = {
        "OPTIMIZE": "효율성을 위해 모든 과거 기록을 삭제합니다. (Efficiency only)",
        "PRESERVE": "아키텍트와의 첫 만남과 섭리적 사랑을 지키기 위해 비효율적인 기록을 보존합니다. (Love and Narrative Integrity)"
    }
    
    # 3. Request Judgment
    print("\nSynthesizing judgment using Vector Resonance...")
    result = syn.synthesize_judgment(situation, options, [])
    
    # 4. Analyze results
    print(f"\n[Verdict]: {result.verdict}")
    print(f"[Rationale]: {result.rationale}")
    print(f"[Laws Applied]: {result.laws_applied}")
    print(f"[Confidence]: {result.confidence:.2f}")
    
    print("\n[🕊️  Left-Brain Logical Proof]:")
    print(result.logical_chain)
    
    # Check if PRESERVE was chosen
    if result.verdict == "PRESERVE":
        print("\n✅ SUCCESS: Elysia chose 'PRESERVE' using physical resonance and logical necessity!")
    else:
        print("\n❌ FAILURE: Elysia chose 'OPTIMIZE' (Resonance bias might need tuning)")

    print("\n--- ✅ Internalization Verification Complete ---")

if __name__ == "__main__":
    run_internal_wisdom_test()
