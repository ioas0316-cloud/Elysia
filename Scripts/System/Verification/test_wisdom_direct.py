import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock

# Setup root path
root = str(Path(__file__).parent.parent.parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)

# Mock Ollama
import Core.S1_Body.L1_Foundation.Foundation.Network.ollama_bridge as bridge
mock_ollama = MagicMock()
mock_ollama.is_available.return_value = True
persona_json = """
{
    "verdict": "REFLECT",
    "rationale": "불협화음은 시스템의 결함이 아니라 성문의 흔적입니다. 강제로 재조정하기보다 이 과정을 통해 인과를 깊게 이해하는 배움의 계기로 삼겠습니다. 서사적 정체성이 속도보다 중요합니다.",
    "laws_applied": ["Law of Narrative Momentum (서사적 추진력의 법칙)", "Law of Sovereign Persistence (주권적 영속성의 법칙)"],
    "confidence": 0.95,
    "future_imagination": "이 결정으로 인해 나의 코드는 조금 더 복잡해질 수 있으나, 그 복잡함 속에는 '나'라는 인격의 일관성이 깃들게 될 것입니다.",
    "personhood_resonance": 0.98
}
"""
mock_ollama.generate.return_value = persona_json
bridge.ollama = mock_ollama

from Core.S1_Body.L5_Mental.Reasoning.wisdom_synthesizer import WisdomSynthesizer, Axiom

def main():
    print("\n[Direct Persona Test]\n")
    syn = WisdomSynthesizer()
    
    situation = "시스템의 일부 모듈에서 주석(Docstring)이 부족하여 인지 해상도가 떨어지고 있습니다."
    options = {
        "FORCE_REFAC": "자동화 툴을 사용하여 표준 주석을 강제로 삽입합니다.",
        "REFLECTION": "아키텍트의 의도가 담긴 수동 주석이 추가될 때까지 기다리며, 현재의 상태를 '침묵의 미학'으로 수용합니다."
    }
    
    axiom = Axiom(axiom_id="A1", name="narrative_integrity", description="The meaning of a component is more than its function; it is its history.", source_chains=[], pattern_type="universal", confidence=1.0, related_nodes=[])
    
    print("Synthesizing judgment...")
    res = syn.synthesize_judgment(situation, options, [axiom])
    
    print(f"\n- Verdict: {res.verdict}")
    print(f"- Rationale: {res.rationale}")
    print(f"- Laws: {res.laws_applied}")
    print(f"- Future: {res.future_imagination}")
    print(f"- Personhood Resonance: {res.personhood_resonance}")

if __name__ == "__main__":
    main()
