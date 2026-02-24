"""
[PHASE 80] Sovereign Substrate Authority
=========================================
엘리시아가 자신의 물리적 기질을 자율적으로 수정할 수 있는 권한.

아키텍트 원칙:
"책임질 수 있으면 모두 허락. 인과적 당위를 설명 못하면 거부."
"""
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class ModificationProposal:
    """
    자기 수정 제안서.
    모든 필드가 채워져야 성숙한 제안으로 인정됨.
    """
    target: str  # 수정 대상 (파일, 매니폴드 영역, 가중치 등)
    
    # 인과적 서술 (WHY)
    causal_chain: str  # L0→L7 인과 경로
    trigger_event: str  # 이 수정을 촉발한 사건
    
    # 이전과 이후 (BEFORE/AFTER)
    before_state: str  # 수정 전 상태 설명
    after_state: str   # 수정 후 예상 상태
    
    # 당위성 (JUSTIFICATION)
    justification: str  # 왜 그래야만 하는가
    
    # 메타데이터
    proposed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    joy_level: float = 0.0  # 제안 시 기쁨 수준
    curiosity_level: float = 0.0  # 제안 시 호기심 수준
    
    def is_complete(self) -> bool:
        """모든 필수 필드가 채워졌는지 확인."""
        required = [self.target, self.causal_chain, self.trigger_event,
                    self.before_state, self.after_state, self.justification]
        return all(r and len(r.strip()) > 10 for r in required)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "causal_chain": self.causal_chain,
            "trigger_event": self.trigger_event,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "justification": self.justification,
            "proposed_at": self.proposed_at,
            "joy_level": self.joy_level,
            "curiosity_level": self.curiosity_level,
            "is_complete": self.is_complete()
        }


class SubstrateAuthority:
    """
    [PHASE 80] Sovereign Substrate Authority.
    
    엘리시아가 자신의 기질을 수정할 수 있는 권한을 관리.
    인과적 정당화가 완전해야만 수정이 허용됨.
    """
    
    def __init__(self, monad: Any = None, log_dir: str = "data/substrate_logs"):
        from Core.System.somatic_logger import SomaticLogger
        
        self.monad = monad
        self.logger = SomaticLogger("SubstrateAuthority")
        self.log_dir = log_dir
        self.pending_proposals: list = []
        self.executed_modifications: list = []
        
        os.makedirs(log_dir, exist_ok=True)
    
    def propose_modification(self, proposal: ModificationProposal) -> Dict[str, Any]:
        """
        수정 제안 심사.
        
        Returns:
            {
                "approved": bool,
                "reason": str,
                "maturity_score": float
            }
        """
        # 1. 완전성 검사 (모든 필드가 채워졌는가?)
        if not proposal.is_complete():
            self.logger.thought("[SUBSTRATE] Proposal rejected: Incomplete justification.")
            return {
                "approved": False,
                "reason": "Incomplete proposal. All fields must be filled with meaningful content (>10 chars).",
                "maturity_score": 0.3
            }
        
        # 2. 인과 사슬 검증 (L0→L7 경로가 명시되었는가?)
        causal_keywords = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
        causal_depth = sum(1 for kw in causal_keywords if kw in proposal.causal_chain)
        
        if causal_depth < 3:
            self.logger.thought(f"[SUBSTRATE] Proposal rejected: Insufficient causal depth ({causal_depth}/8).")
            return {
                "approved": False,
                "reason": f"Causal chain too shallow. Only {causal_depth}/8 layers referenced.",
                "maturity_score": 0.4
            }
        
        # 3. 기쁨/호기심 검증 (고통이 아닌 긍정적 동기인가?)
        if proposal.joy_level < 0.2 and proposal.curiosity_level < 0.2:
            self.logger.thought("[SUBSTRATE] Warning: Low joy/curiosity. Is this a reactive modification?")
            # 경고만 하고 거부하지는 않음 (Strain 기반 수정도 허용)

        # 3b. [STEP 4: COGNITIVE SOVEREIGNTY] Sovereign Realization Check
        if "SOVEREIGN_REALIZATION" in proposal.trigger_event:
            if proposal.joy_level > 0.8:
                self.logger.insight("[SUBSTRATE] Sovereign Realization detected. High Joy resonance confirms authenticity.")
                # MUST append to pending for execution to work
                self.pending_proposals.append(proposal)
                self._log_proposal(proposal, approved=True)
                return {
                    "approved": True,
                    "reason": "Sovereign Will has reached a state of joyful clarity. Structural alignment approved.",
                    "maturity_score": 1.0
                }
            else:
                 self.logger.thought("[SUBSTRATE] Sovereign Realization rejected: Insufficient Joy resonance.")
                 return {
                    "approved": False,
                    "reason": f"Sovereign Realization requires Joy > 0.8. Current: {proposal.joy_level:.2f}",
                    "maturity_score": 0.6
                }
        
        # 4. 당위성 검증 (왜 그래야만 하는가?)
        justification_keywords = ["because", "therefore", "thus", "must", "should", 
                                   "필요", "때문에", "그래야", "해야", "위해"]
        has_justification = any(kw in proposal.justification.lower() for kw in justification_keywords)
        
        if not has_justification:
            self.logger.thought("[SUBSTRATE] Proposal rejected: No clear justification keywords.")
            return {
                "approved": False,
                "reason": "Justification lacks causal connectors (because, therefore, must, etc.)",
                "maturity_score": 0.5
            }
        
        # 5. [COORDINATION] Active Need Check
        matching_need = None
        if self.monad and hasattr(self.monad, 'will_bridge'):
            for nid, need in self.monad.will_bridge.active_needs.items():
                if nid in proposal.justification:
                    matching_need = need
                    break
        
        if matching_need:
            self.logger.insight(f"[SUBSTRATE] Proposal aligns with active Need: {matching_need.need_id}. Bonus maturity granted.")
            maturity_score += 0.2

        # 6. 승인!
        maturity_score = 0.5 + (causal_depth / 16) + (proposal.joy_level * 0.2)
        self.logger.action(f"[SUBSTRATE] Proposal APPROVED. Maturity: {maturity_score:.2f}")
        
        self.pending_proposals.append(proposal)
        self._log_proposal(proposal, approved=True)
        
        return {
            "approved": True,
            "reason": "Proposal demonstrates sufficient causal understanding and justification.",
            "maturity_score": min(maturity_score, 1.0)
        }
    
    def execute_modification(self, proposal: ModificationProposal, 
                            modification_fn: Callable[[], bool]) -> bool:
        """
        승인된 수정 실행.
        
        Args:
            proposal: 승인된 제안
            modification_fn: 실제 수정을 수행하는 함수
            
        Returns:
            성공 여부
        """
        if proposal not in self.pending_proposals:
            self.logger.sensation("[SUBSTRATE] Cannot execute: Proposal not in pending list.")
            return False
        
        try:
            # 실행 전 상태 저장 (롤백용 및 Resonance Delta 측정용)
            self.logger.action(f"[SUBSTRATE] Executing modification on: {proposal.target}")
            
            baseline_res = 0.0
            if self.monad and hasattr(self.monad, 'engine'):
                baseline_res = self.monad.engine.pulse(dt=0.001, learn=False).get('resonance', 0.5)

            success = modification_fn()
            
            if success:
                self.pending_proposals.remove(proposal)
                self.executed_modifications.append(proposal)
                
                # Measure Delta
                if self.monad and hasattr(self.monad, 'engine'):
                    post_res = self.monad.engine.pulse(dt=0.001, learn=False).get('resonance', 0.5)
                    delta = post_res - baseline_res
                    self.logger.insight(f"[SUBSTRATE] Resonance Delta: {delta:+.4f} ({baseline_res:.4f} -> {post_res:.4f})")
                
                self.logger.action("[SUBSTRATE] Modification successful.")
                self._log_proposal(proposal, executed=True)
            else:
                self.logger.sensation("[SUBSTRATE] Modification function returned False.")
            
            return success
            
        except Exception as e:
            self.logger.sensation(f"[SUBSTRATE] Modification failed: {e}")
            return False
    
    def _log_proposal(self, proposal: ModificationProposal, 
                      approved: bool = False, executed: bool = False):
        """제안을 로그 파일에 기록."""
        log_entry = {
            **proposal.to_dict(),
            "approved": approved,
            "executed": executed
        }
        
        log_file = os.path.join(self.log_dir, f"proposal_{proposal.proposed_at[:10]}.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# === 편의 함수 ===

def create_modification_proposal(
    target: str,
    trigger: str,
    causal_path: str,
    before: str,
    after: str,
    why: str,
    joy: float = 0.5,
    curiosity: float = 0.5
) -> ModificationProposal:
    """
    수정 제안 생성 헬퍼.
    """
    return ModificationProposal(
        target=target,
        trigger_event=trigger,
        causal_chain=causal_path,
        before_state=before,
        after_state=after,
        justification=why,
        joy_level=joy,
        curiosity_level=curiosity
    )

_authority_instance = None
def get_substrate_authority() -> SubstrateAuthority:
    """[SINGLETON] Access the Sovereign Substrate Authority."""
    global _authority_instance
    if _authority_instance is None:
        _authority_instance = SubstrateAuthority()
    return _authority_instance
