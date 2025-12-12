"""
Prove Latent Mining (잠재 채굴 증명)
==================================

"무의식에서 지식을 길어 올리다."

LatentMiner가 주어진 개념('Forest')에 대해
잠재 공간을 탐색하고, 연관된 개념들('Green', 'Trees')을
스스로 학습하는지 검증합니다.
"""

from Core.Cognitive.latent_miner import get_latent_miner
from Core.Cognitive.concept_formation import get_concept_formation

def prove_mining():
    print("⛏️ LATENT MINING VERIFICATION...\n")
    
    miner = get_latent_miner()
    concepts = get_concept_formation()
    
    # 1. Target Concept
    target = "Forest"
    print(f"1. Target Concept: '{target}'")
    # Ensure root exists
    concepts.learn_concept(target, "Nature", domain="aesthetic")
    
    # 2. Mining Loop
    print("\n2. Initiating Probe...")
    miner.digest(target)
    
    # 3. Verification
    print("\n3. Verifying Concept Web...")
    forest_concept = concepts.get_concept(target)
    
    print(f"   🌲 Forest Links: {forest_concept.synaptic_links}")
    
    # Check if children exist
    if "aesthetic:Green" in forest_concept.synaptic_links:
        print("\n✅ SUCCESS: 'Forest' successfully linked to 'Green' from latent knowledge.")
    else:
        print("\n❌ FAIL: Learning failed.")

if __name__ == "__main__":
    prove_mining()
