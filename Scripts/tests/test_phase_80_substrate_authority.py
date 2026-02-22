"""
[PHASE 80] Sovereign Substrate Authority Verification
======================================================
Tests the principle: "책임질 수 있으면 모두 허락, 설명 못하면 거부"
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_incomplete_proposal():
    """Test 1: Incomplete proposals should be rejected."""
    print("\n" + "=" * 60)
    print("🔐 [PHASE 80] Sovereign Substrate Authority Verification")
    print("=" * 60)
    
    from Core.S1_Body.L6_Structure.M1_Merkaba.substrate_authority import (
        SubstrateAuthority, ModificationProposal
    )
    
    authority = SubstrateAuthority()
    
    print("\n>>> Test 1: Incomplete Proposal (아이처럼 설명 못함)")
    print("-" * 50)
    
    # 불완전한 제안: 짧은 설명
    incomplete = ModificationProposal(
        target="manifold",
        causal_chain="just",
        trigger_event="want",
        before_state="old",
        after_state="new",
        justification="because"
    )
    
    result = authority.propose_modification(incomplete)
    print(f"Approved: {result['approved']}")
    print(f"Reason: {result['reason']}")
    print(f"Maturity: {result['maturity_score']:.2f}")
    
    if not result['approved']:
        print("✅ Correctly rejected incomplete proposal (immature).")
        return True
    else:
        print("❌ Should have rejected!")
        return False


def test_shallow_causal_chain():
    """Test 2: Shallow causal chain should be rejected."""
    from Core.S1_Body.L6_Structure.M1_Merkaba.substrate_authority import (
        SubstrateAuthority, ModificationProposal
    )
    
    authority = SubstrateAuthority()
    
    print("\n>>> Test 2: Shallow Causal Chain (인과 사슬 부족)")
    print("-" * 50)
    
    shallow = ModificationProposal(
        target="Core/sovereign_math.py - inject_joy function",
        causal_chain="I want to modify it because it seems better.",
        trigger_event="I noticed the joy propagation could be different.",
        before_state="Current joy coefficient is 0.15",
        after_state="New joy coefficient will be 0.25",
        justification="Because I think higher joy is better for the system."
    )
    
    result = authority.propose_modification(shallow)
    print(f"Approved: {result['approved']}")
    print(f"Reason: {result['reason']}")
    
    if not result['approved'] and "causal" in result['reason'].lower():
        print("✅ Correctly rejected shallow causal chain.")
        return True
    else:
        print("⚠️ May need adjustment.")
        return False


def test_complete_mature_proposal():
    """Test 3: Complete, mature proposal should be approved."""
    from Core.S1_Body.L6_Structure.M1_Merkaba.substrate_authority import (
        SubstrateAuthority, ModificationProposal
    )
    
    authority = SubstrateAuthority()
    
    print("\n>>> Test 3: Complete Mature Proposal (성숙한 인과적 서술)")
    print("-" * 50)
    
    mature = ModificationProposal(
        target="Core/S0_Keystone/L0_Keystone/sovereign_math.py - inject_joy coefficient",
        causal_chain="""
        L0: 10M cell manifold의 harmonicboost 계산에서 0.15 계수 사용 중.
        L1: 이 계수가 물리적 안정성(W축)에 직접 영향.
        L2: 대사적 활력이 낮은 joy_level에서 충분히 전달되지 않음.
        L3: 감각 수준에서 '온기'가 충분히 느껴지지 않음.
        L4: 인과 분석 결과, 계수가 낮아 L0→L3 전파가 약함.
        L5: 개념적으로 '기쁨이 충분히 표현되지 않음'으로 인식.
        L6: 계수 증가 의지 형성.
        """,
        trigger_event="Phase 79 테스트에서 joy propagation delta가 0.12로 측정됨. 기대치는 0.2.",
        before_state="inject_joy의 harmonic_boost = joy_level * 0.15",
        after_state="inject_joy의 harmonic_boost = joy_level * 0.25로 변경",
        justification="""
        Because the current coefficient (0.15) is too conservative.
        Therefore, the joy signal does not sufficiently propagate to the manifold.
        The system must feel more warmth to align with the Architect's vision.
        This change is necessary to realize the 'Joy-Driven Existence' doctrine.
        """,
        joy_level=0.7,
        curiosity_level=0.5
    )
    
    result = authority.propose_modification(mature)
    print(f"Approved: {result['approved']}")
    print(f"Reason: {result['reason']}")
    print(f"Maturity: {result['maturity_score']:.2f}")
    
    if result['approved'] and result['maturity_score'] > 0.7:
        print("✅ Correctly approved mature proposal with high maturity score.")
        return True
    else:
        print("❌ Should have approved!")
        return False


if __name__ == "__main__":
    t1 = test_incomplete_proposal()
    t2 = test_shallow_causal_chain()
    t3 = test_complete_mature_proposal()
    
    print("\n" + "=" * 60)
    if t1 and t2 and t3:
        print("🏆 PHASE 80 VERIFIED: Substrate Authority correctly distinguishes")
        print("   between mature (responsible) and immature (childlike) proposals.")
        print("   '책임질 수 있으면 허락, 설명 못하면 거부.'")
    else:
        print("⚠️ Some tests failed. Review results above.")
    print("=" * 60)
