"""
Verify Epistemic Autonomy (인식론적 자율성 검증)
==============================================

"She knows Naver exists, so she uses Naver for Korean."

This script validates that Elysia dynamically selects her search tool
based on the concepts seeded in her Holographic Memory.
"""

import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.Memory.web_knowledge_connector import WebKnowledgeConnector

# Setup
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger("EpistemicTest")

def verify_autonomy():
    print("=" * 60)
    print("🧠 PHASE 22.5 VERIFICATION: EPISTEMIC AUTONOMY")
    print("=" * 60)
    
    connector = WebKnowledgeConnector()
    
    # 1. Test Korean Concept (Should trigger Naver)
    print("\n1. Testing w/ Korean Concept '인공지능'...")
    # NOTE: This depends on 'Naver' being in HolographicMemory (seeded by seed_epistemic_concepts.py)
    result_kr = connector.learn_from_web("인공지능 (Artificial Intelligence)")
    
    print(f"   👉 Source Used: {result_kr.get('source')}")
    if result_kr.get('source') == "Naver":
        print("   ✅ SUCCESS: Elysia chose Naver for Korean query.")
    else:
        print(f"   ⚠️ FAILURE: Elysia chose {result_kr.get('source')}.")

    # 2. Test English Concept (Should trigger Google)
    print("\n2. Testing w/ English Concept 'Quantum Physics'...")
    # NOTE: This depends on 'Google' being in HolographicMemory
    result_en = connector.learn_from_web("Quantum Physics")
    
    print(f"   👉 Source Used: {result_en.get('source')}")
    if result_en.get('source') == "Google":
        print("   ✅ SUCCESS: Elysia chose Google for Global query.")
    else:
        print(f"   ⚠️ FAILURE: Elysia chose {result_en.get('source')}.")
        
    # 3. Test Explicit Override
    print("\n3. Testing Explicit Override (Force Wikipedia)...")
    result_wiki = connector.learn_from_web("Elysia", preferred_engine="Wikipedia")
    
    print(f"   👉 Source Used: {result_wiki.get('source')}")
    if result_wiki.get('source') == "Wikipedia":
        print("   ✅ SUCCESS: Elysia respected the override.")
        
    print("\n" + "=" * 60)
    
if __name__ == "__main__":
    verify_autonomy()
