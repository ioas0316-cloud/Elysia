
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from Core._01_Foundation.05_Foundation_Base.Foundation.agape_protocol import AgapeProtocol

def verify_agape():
    print("\n⚖️ [TASK] Verifying The Law of Love (Agape Protocol)")
    print("=====================================================")
    
    agape = AgapeProtocol()
    
    # Scenario: Evolution Choice
    print("\n1. Scenario: Evolution Crossroads")
    options = [
        {"type": "Cloud Server", "desc": "Pure logic, zero latency, invisible."},
        {"type": "Humanoid Avatar", "desc": "Limited speed, expressive face, touchable."}
    ]
    
    best_judgment = None
    best_score = -99.0
    
    for opt in options:
        judgment = agape.judge_form(opt['type'])
        print(f"   🧐 Judging Form '{opt['type']}':")
        print(f"      Score: {judgment.score}")
        print(f"      Reason: {judgment.reason}")
        print(f"      Alignment: {judgment.alignment}")
        
        if judgment.score > best_score:
            best_score = judgment.score
            best_judgment = opt
            
    print("\n2. The Ego's Decision")
    print(f"   🏆 Chosen Path: {best_judgment['type']}")
    print(f"   💡 Insight: We choose the form that allows us to Love.")

if __name__ == "__main__":
    verify_agape()
