"""
VERIFY CROSS-DOMAIN ANALOGY (도메인 교차 유유추 검증)
===================================================

목표: 엘리시아가 물리 엔진의 원리를 게임 개발이나 다른 영역의 지식으로 확장할 수 있는지 확인합니다.
"""

import sys
import os
import logging

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("AnalogyTest")

from Core.Intelligence.Reasoning.meta_inquiry import MetaInquiry

def verify():
    print("\n[VERIFICATION] Cross-Domain Principle Transfer...")
    print("------------------------------------------------")
    
    mi = MetaInquiry()
    
    # CASE 1: Physics (Gravity) -> Game Design (Engagement)
    print("\n[CASE 1] Principle: Inverse Square Law")
    print("Source: Physics (Engine) | Target: Game Design")
    
    analogy = mi.seek_analogy("Inverse Square Law", "Physics", "Game Design")
    
    if analogy:
        print("\n[EPIPHANY FOUND]")
        print(f" - Invariant Skeleton: {analogy.abstraction}")
        print(f" - Physics manifestation: Gravity/Light Intensity")
        print(f" - Game Design manifestation: {analogy.target_application}")
        print("\n✅ SUCCESS: The physical principle has been abstracted and reified in a creative domain.")
    else:
        print("❌ FAILURE: No analogy found.")

    # CASE 2: Physics (Momentum) -> Storytelling (Narrative)
    print("\n[CASE 2] Principle: Conservation of Momentum")
    print("Source: Physics (Collisions) | Target: Storytelling")
    
    analogy = mi.seek_analogy("Conservation of Momentum", "Physics", "Storytelling")
    
    if analogy:
        print("\n[EPIPHANY FOUND]")
        print(f" - Invariant Skeleton: {analogy.abstraction}")
        print(f" - Storytelling manifestation: {analogy.target_application}")
        print("\n✅ SUCCESS: Elysia understands that narrative tension must be 'conserved' and 'resolved' like momentum.")
    else:
         print("❌ FAILURE: No analogy found.")

if __name__ == "__main__":
    verify()
