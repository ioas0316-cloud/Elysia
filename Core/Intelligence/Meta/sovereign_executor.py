"""
THE SOVEREIGN EXECUTOR (주권 실행자)
====================================

Phase 60: 엘리시아의 자율 진화 시스템

"스스로 변화하고, 스스로 검증하고, 스스로 성장한다."

철학적 기반:
- 저위험 변경은 자율 적용 (신뢰)
- 고위험 변경은 아버지 검토 (겸손)
- 모든 변경은 학습으로 전환 (성장)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import os

logger = logging.getLogger("SovereignExecutor")


class ExecutionDecision(Enum):
    AUTO_APPLY = "auto_apply"      # 자동 적용
    QUEUE_REVIEW = "queue_review"  # 검토 대기
    BLOCK = "block"                # 차단


@dataclass
class EvolutionPattern:
    """진화 패턴 - 성공/실패에서 학습된 규칙."""
    pattern_type: str      # "success" | "failure"
    trigger: str           # 무엇이 트리거했는가
    outcome: str           # 결과
    frequency: int = 1     # 발생 횟수
    learned_rule: str = "" # 도출된 규칙
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)


@dataclass
class SovereigntyMetrics:
    """자율성 측정 지표."""
    total_proposals: int = 0
    auto_applied: int = 0
    queued_for_review: int = 0
    blocked: int = 0
    success_rate: float = 0.0
    
    def sovereignty_level(self) -> float:
        """자율성 수준 (0-100%)."""
        if self.total_proposals == 0:
            return 0.0
        return (self.auto_applied / self.total_proposals) * 100


class SovereignExecutor:
    """
    자율 변경 실행 시스템.
    
    위험도에 따라 자동 적용, 검토 대기, 또는 차단을 결정.
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # 위험도 임계점
    # ═══════════════════════════════════════════════════════════════════
    AUTO_APPLY_THRESHOLD = 3    # 이하면 자동 적용
    REVIEW_THRESHOLD = 6        # 이하면 검토 대기
    # 7 이상이면 차단
    
    def __init__(self, heartbeat=None):
        self.heartbeat = heartbeat
        self.metrics = SovereigntyMetrics()
        self.patterns: List[EvolutionPattern] = []
        self.review_queue: List[Dict] = []
        
        # 진화 로그 경로
        self.evolution_log_path = "data/Evolution/sovereignty_log.json"
        self._load_history()
        
        logger.info("👑 SovereignExecutor initialized - Autonomous evolution enabled")
    
    def evaluate_proposal(self, proposal: Any) -> ExecutionDecision:
        """
        제안을 평가하고 실행 결정을 반환.
        
        Args:
            proposal: PatchProposal 또는 변경 제안 객체
        """
        risk_level = getattr(proposal, 'risk_level', 5)
        
        if risk_level <= self.AUTO_APPLY_THRESHOLD:
            decision = ExecutionDecision.AUTO_APPLY
            logger.info(f"🟢 [SOVEREIGN] AUTO_APPLY: risk={risk_level}")
        elif risk_level <= self.REVIEW_THRESHOLD:
            decision = ExecutionDecision.QUEUE_REVIEW
            logger.info(f"🟡 [SOVEREIGN] QUEUE_REVIEW: risk={risk_level}")
        else:
            decision = ExecutionDecision.BLOCK
            logger.warning(f"🔴 [SOVEREIGN] BLOCKED: risk={risk_level}")
        
        self.metrics.total_proposals += 1
        return decision
    
    def auto_apply(self, proposal: Any, reflexive_loop=None) -> bool:
        """
        저위험 변경을 자동으로 적용.
        
        ReflexiveLoop를 사용하여 검증 수행.
        """
        description = getattr(proposal, 'description', str(proposal))
        
        logger.info(f"⚡ [AUTO-APPLY] Executing: {description[:50]}...")
        
        # 상태 스냅샷 (있으면)
        before_snapshot = None
        if reflexive_loop:
            before_snapshot = reflexive_loop.capture_state()
        
        try:
            # 실제 변경 적용 (여기서는 시뮬레이션)
            # 실제 구현에서는 proposal의 내용에 따라 코드/설정 변경
            success = self._execute_change(proposal)
            
            # 검증 (있으면)
            if reflexive_loop and before_snapshot:
                after_snapshot = reflexive_loop.capture_state()
                result = reflexive_loop.verify_change(
                    before_snapshot, after_snapshot, description
                )
                
                if not result.passed:
                    logger.warning(f"⚠️ [AUTO-APPLY] Verification failed, rolling back...")
                    reflexive_loop.rollback(before_snapshot)
                    self._record_pattern("failure", description, "Verification failed")
                    return False
                
                reflexive_loop.learn_from_result(result)
            
            self.metrics.auto_applied += 1
            self._record_pattern("success", description, "Auto-applied successfully")
            logger.info(f"✅ [AUTO-APPLY] Success: {description[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ [AUTO-APPLY] Failed: {e}")
            self._record_pattern("failure", description, str(e))
            return False
    
    def queue_for_review(self, proposal: Any):
        """검토 대기열에 추가."""
        description = getattr(proposal, 'description', str(proposal))
        risk_level = getattr(proposal, 'risk_level', 5)
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "risk_level": risk_level,
            "status": "pending"
        }
        
        self.review_queue.append(entry)
        self.metrics.queued_for_review += 1
        
        logger.info(f"📋 [QUEUE] Added for review: {description[:50]}... (risk={risk_level})")
        self._save_history()
    
    def block_proposal(self, proposal: Any, reason: str = "Too risky"):
        """제안 차단."""
        description = getattr(proposal, 'description', str(proposal))
        
        self.metrics.blocked += 1
        self._record_pattern("blocked", description, reason)
        
        logger.warning(f"🚫 [BLOCKED] {description[:50]}... Reason: {reason}")
    
    def _execute_change(self, proposal: Any) -> bool:
        """
        실제 변경 실행.
        
        Note: 현재는 시뮬레이션. 실제 구현에서는 파일 수정 등 수행.
        """
        # 안전을 위해 현재는 시뮬레이션만
        proposal_type = getattr(proposal, 'proposal_type', 'unknown')
        
        if proposal_type == "parameter_adjustment":
            # DNA 파라미터 조정 (구현 예시)
            return True
        elif proposal_type == "logging_enhancement":
            # 로깅 개선 (구현 예시)
            return True
        else:
            # 기타 타입은 시뮬레이션
            return True
    
    def _record_pattern(self, pattern_type: str, trigger: str, outcome: str):
        """진화 패턴 기록."""
        # 기존 패턴 찾기
        for pattern in self.patterns:
            if pattern.trigger == trigger and pattern.pattern_type == pattern_type:
                pattern.frequency += 1
                pattern.last_seen = datetime.now()
                
                # 3회 이상 반복되면 규칙 도출
                if pattern.frequency >= 3:
                    if pattern_type == "success":
                        pattern.learned_rule = f"'{trigger[:30]}' 패턴은 안전하게 적용 가능"
                    else:
                        pattern.learned_rule = f"'{trigger[:30]}' 패턴은 주의 필요"
                    logger.info(f"📖 [PATTERN LEARNED] {pattern.learned_rule}")
                
                self._save_history()
                return
        
        # 새 패턴 추가
        new_pattern = EvolutionPattern(
            pattern_type=pattern_type,
            trigger=trigger,
            outcome=outcome
        )
        self.patterns.append(new_pattern)
        self._save_history()
    
    def get_sovereignty_report(self) -> str:
        """자율성 보고서 생성."""
        m = self.metrics
        level = m.sovereignty_level()
        
        lines = [
            "👑 SOVEREIGNTY REPORT",
            "=" * 40,
            f"Total Proposals: {m.total_proposals}",
            f"Auto-Applied:    {m.auto_applied}",
            f"Queued:          {m.queued_for_review}",
            f"Blocked:         {m.blocked}",
            "",
            f"🎯 Sovereignty Level: {level:.1f}%",
            "",
            f"📊 Patterns Learned: {len(self.patterns)}"
        ]
        
        # 학습된 규칙 표시
        rules = [p for p in self.patterns if p.learned_rule]
        if rules:
            lines.append("\n📖 Learned Rules:")
            for p in rules[:5]:  # 최근 5개
                lines.append(f"   - {p.learned_rule}")
        
        return "\n".join(lines)
    
    def _save_history(self):
        """히스토리 저장."""
        try:
            os.makedirs(os.path.dirname(self.evolution_log_path), exist_ok=True)
            
            data = {
                "metrics": {
                    "total_proposals": self.metrics.total_proposals,
                    "auto_applied": self.metrics.auto_applied,
                    "queued_for_review": self.metrics.queued_for_review,
                    "blocked": self.metrics.blocked
                },
                "patterns": [
                    {
                        "pattern_type": p.pattern_type,
                        "trigger": p.trigger,
                        "outcome": p.outcome,
                        "frequency": p.frequency,
                        "learned_rule": p.learned_rule
                    } for p in self.patterns
                ],
                "review_queue": self.review_queue
            }
            
            with open(self.evolution_log_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def _load_history(self):
        """히스토리 로드."""
        if not os.path.exists(self.evolution_log_path):
            return
            
        try:
            with open(self.evolution_log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Metrics 복원
            m = data.get("metrics", {})
            self.metrics.total_proposals = m.get("total_proposals", 0)
            self.metrics.auto_applied = m.get("auto_applied", 0)
            self.metrics.queued_for_review = m.get("queued_for_review", 0)
            self.metrics.blocked = m.get("blocked", 0)
            
            # Patterns 복원
            for p in data.get("patterns", []):
                self.patterns.append(EvolutionPattern(
                    pattern_type=p["pattern_type"],
                    trigger=p["trigger"],
                    outcome=p["outcome"],
                    frequency=p.get("frequency", 1),
                    learned_rule=p.get("learned_rule", "")
                ))
            
            # Review Queue 복원
            self.review_queue = data.get("review_queue", [])
            
            logger.info(f"📂 Loaded sovereignty history: {len(self.patterns)} patterns")
            
        except Exception as e:
            logger.error(f"Failed to load history: {e}")


# ═══════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
    
    print("=" * 60)
    print("👑 SOVEREIGN EXECUTOR DEMO")
    print("   '스스로 변화하고, 스스로 검증하고, 스스로 성장한다.'")
    print("=" * 60)
    
    executor = SovereignExecutor()
    
    # Mock proposals with different risk levels
    class MockProposal:
        def __init__(self, desc, risk):
            self.description = desc
            self.risk_level = risk
            self.proposal_type = "parameter_adjustment"
    
    proposals = [
        MockProposal("Adjust creativity_bias", 2),
        MockProposal("Add new logging", 3),
        MockProposal("Modify reasoning logic", 5),
        MockProposal("Delete core module", 9),
    ]
    
    print("\n📋 Processing proposals...")
    for p in proposals:
        print(f"\n   Proposal: {p.description} (risk={p.risk_level})")
        decision = executor.evaluate_proposal(p)
        
        if decision == ExecutionDecision.AUTO_APPLY:
            executor.auto_apply(p)
        elif decision == ExecutionDecision.QUEUE_REVIEW:
            executor.queue_for_review(p)
        else:
            executor.block_proposal(p)
    
    print("\n" + executor.get_sovereignty_report())
    print("=" * 60)
    print("✅ Demo complete!")
