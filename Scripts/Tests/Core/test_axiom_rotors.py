import pytest
import torch
from Core.L7_Spirit.M1_Monad.monad_core import Monad
from Core.L6_Structure.M5_Engine.governance_engine import GovernanceEngine

def test_monad_zero_frequency():
    # Both monads should have hashes derived from the same Zero-Frequency anchor
    m1 = Monad("SeedA")
    m2 = Monad("SeedB")
    
    assert m1.ZERO_FREQUENCY_ID == "나는 엘리시아다"
    assert m1._id_hash != m2._id_hash # Still unique
    # But internally they both used ZERO_FREQUENCY_ID in the hash chain (implicitly verified by code inspection or looking at construction logic)

def test_governance_axiom_rotors():
    gov = GovernanceEngine()
    
    # Check if axiom rotors exist
    assert "Identity" in gov.dials
    assert "Purpose" in gov.dials
    assert "Future" in gov.dials
    
    # Verify Identity DNA
    identity_rotor = gov.dials["Identity"]
    assert identity_rotor.dna.label == "나는 엘리시아다"
    
    # Verify Purpose DNA
    purpose_rotor = gov.dials["Purpose"]
    assert "Why" in purpose_rotor.dna.label
    
    print(f"\n[TEST] Identity Rotor: {identity_rotor}")
    print(f"[TEST] Purpose Rotor: {purpose_rotor}")
    print(f"[TEST] Future Rotor: {gov.dials['Future']}")

if __name__ == "__main__":
    pytest.main([__file__])
