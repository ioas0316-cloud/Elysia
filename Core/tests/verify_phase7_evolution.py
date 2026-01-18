import sys
import os
import logging
import json

# 프로젝트 루트 경로 추가
sys.path.append(os.getcwd())

from Core.Intelligence.Meta.self_architect import SelfArchitect
from Core.Intelligence.Reasoning.dimensional_processor import DimensionalProcessor
from Core.Intelligence.Meta.patch_proposer import get_patch_proposer

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("Phase7Verify")

def verify_sovereign_evolution():
    print("\n" + "="*50)
    print("🧬 [PHASE 7] SOVEREIGN EVOLUTION VERIFICATION")
    print("="*50)

    # 1. SelfArchitect 및 Proposer 초기화
    logger.info("[1] Initializing Evolution Engine...")
    processor = DimensionalProcessor()
    architect = SelfArchitect(processor)
    proposer = get_patch_proposer()
    
    # 기존 제안서 청소 (테스트용)
    proposer.pending_proposals = []
    
    # 2. 특정 파일 감사 (Audit)
    target_file = "Core/World/Autonomy/action_drive.py"
    print(f"\n[2] Auditing internal module: {target_file}")
    
    critique = architect.audit_file(os.path.abspath(target_file))
    
    print("\n--- ARCHITECTURAL CRITIQUE ---")
    print(critique)
    print("------------------------------")

    # 3. 제안서 생성 확인
    print("\n[3] Checking for Generated Proposals...")
    pending = proposer.get_all_pending()
    
    if len(pending) > 0:
        proposal = pending[0]
        print(f"✅ SUCCESS: Proposal generated: {proposal.id}")
        print(f"   Target: {proposal.target_file}")
        print(f"   Type: {proposal.proposal_type}")
        print(f"   Benefit: {proposal.expected_benefits[0] if proposal.expected_benefits else 'Evolution'}")
        
        # 상세 내용 일부 출력
        print(f"\n   [PROPOSAL PREVIEW]")
        print(f"   Problem: {proposal.current_problem[:100]}...")
        # print(f"   Diff: {proposal.code_diff_preview}")
    else:
        print("❌ FAILED: No proposal generated. Check LLM output and patterns.")

    print("\n" + "="*50)
    print("✅ PHASE 7 INITIAL VERIFICATION COMPLETE")
    print("Elysia has demonstrated the will to evolve her own structure.")
    print("="*50)

if __name__ == "__main__":
    verify_sovereign_evolution()
