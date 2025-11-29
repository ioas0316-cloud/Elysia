"""
Auto-Fix System with Approval Workflow (자동 수정 시스템)
=========================================================

발견된 이슈들을 자동으로 수정하되, 안전하게 승인 시스템을 거쳐서.

핵심 원칙:
1. 모든 수정은 먼저 제안으로 생성
2. 승인 없이는 절대 수정하지 않음
3. 모든 변경은 백업 후 진행
4. 되돌리기 항상 가능

영화 참고:
- Transcendence: 자율적 코드 개선 (단, 안전하게)
- Skynet: 자기 진화 (위험하므로 승인 시스템 필수)
"""

from __future__ import annotations

import logging
import time
import uuid
import shutil
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto
from datetime import datetime

logger = logging.getLogger("AutoFix")


class FixStatus(Enum):
    """수정 상태"""
    PENDING = auto()      # 대기 중 (승인 필요)
    APPROVED = auto()     # 승인됨
    REJECTED = auto()     # 거부됨
    APPLIED = auto()      # 적용됨
    ROLLED_BACK = auto()  # 되돌림


class FixCategory(Enum):
    """수정 카테고리"""
    STYLE = auto()        # 스타일 (안전)
    DOCUMENTATION = auto() # 문서화 (안전)
    REFACTORING = auto()  # 리팩토링 (주의)
    PERFORMANCE = auto()  # 성능 (주의)
    BUG_FIX = auto()      # 버그 수정 (위험)
    SECURITY = auto()     # 보안 (매우 위험)


# 카테고리별 위험도
CATEGORY_RISK = {
    FixCategory.STYLE: 1,
    FixCategory.DOCUMENTATION: 1,
    FixCategory.REFACTORING: 3,
    FixCategory.PERFORMANCE: 3,
    FixCategory.BUG_FIX: 4,
    FixCategory.SECURITY: 5,
}


@dataclass
class FixProposal:
    """수정 제안"""
    id: str
    category: FixCategory
    file_path: str
    line_start: int
    line_end: int
    original_code: str
    fixed_code: str
    description: str
    description_kr: str
    confidence: float  # 0.0 ~ 1.0
    risk_level: int    # 1 ~ 5
    status: FixStatus = FixStatus.PENDING
    
    # 메타데이터
    created_at: float = field(default_factory=time.time)
    approved_at: Optional[float] = None
    applied_at: Optional[float] = None
    approved_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.name,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "original_code": self.original_code,
            "fixed_code": self.fixed_code,
            "description": self.description,
            "description_kr": self.description_kr,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "status": self.status.name,
            "created_at": self.created_at
        }


@dataclass
class Backup:
    """백업 정보"""
    id: str
    file_path: str
    backup_path: str
    original_content: str
    created_at: float = field(default_factory=time.time)


class AutoFixSystem:
    """
    자동 수정 시스템
    
    발견된 이슈를 자동으로 수정 제안하고,
    승인 후에만 실제로 적용하는 안전한 시스템.
    """
    
    def __init__(
        self,
        project_root: str = None,
        backup_dir: str = None,
        auto_approve_threshold: float = 0.95,  # 자동 승인 신뢰도 임계값
        max_risk_auto_approve: int = 2  # 자동 승인 최대 위험도
    ):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.backup_dir = Path(backup_dir) if backup_dir else self.project_root / ".elysia_backups"
        self.auto_approve_threshold = auto_approve_threshold
        self.max_risk_auto_approve = max_risk_auto_approve
        
        # 백업 디렉토리 생성
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 제안 및 백업 저장소
        self.proposals: Dict[str, FixProposal] = {}
        self.backups: Dict[str, Backup] = {}
        self.history: List[Dict[str, Any]] = []
        
        logger.info(f"🔧 AutoFixSystem initialized")
        logger.info(f"   - Backup dir: {self.backup_dir}")
        logger.info(f"   - Auto-approve threshold: {auto_approve_threshold}")
    
    def generate_fix_for_issue(
        self,
        issue: Dict[str, Any],
        file_path: str = None
    ) -> Optional[FixProposal]:
        """
        이슈에 대한 수정 제안 생성
        
        파동 언어 분석 결과를 바탕으로 수정 코드 생성
        """
        issue_type = issue.get("type", "UNKNOWN")
        description = issue.get("description", "")
        line = issue.get("line", 0)
        severity = issue.get("severity", "low")
        
        # 카테고리 결정
        category = self._determine_category(issue_type)
        
        # 수정 코드 생성 (간단한 규칙 기반)
        fix_result = self._generate_fix_code(issue, file_path)
        
        if not fix_result:
            return None
        
        original_code, fixed_code, fix_description = fix_result
        
        proposal = FixProposal(
            id=str(uuid.uuid4())[:8],
            category=category,
            file_path=file_path or "",
            line_start=line,
            line_end=line,
            original_code=original_code,
            fixed_code=fixed_code,
            description=fix_description,
            description_kr=description,
            confidence=self._calculate_confidence(issue, fix_result),
            risk_level=CATEGORY_RISK.get(category, 3)
        )
        
        self.proposals[proposal.id] = proposal
        
        logger.info(f"📝 Generated fix proposal: {proposal.id} ({category.name})")
        
        return proposal
    
    def _determine_category(self, issue_type: str) -> FixCategory:
        """이슈 유형에서 카테고리 결정"""
        type_map = {
            "STYLE": FixCategory.STYLE,
            "READABILITY": FixCategory.DOCUMENTATION,
            "STRUCTURE": FixCategory.REFACTORING,
            "PERFORMANCE": FixCategory.PERFORMANCE,
            "SECURITY": FixCategory.SECURITY,
            "INNOVATION": FixCategory.REFACTORING,
            "IMPROVEMENT": FixCategory.STYLE,
            "BUG_FIX": FixCategory.BUG_FIX,
        }
        return type_map.get(issue_type.upper(), FixCategory.STYLE)
    
    def _generate_fix_code(
        self,
        issue: Dict[str, Any],
        file_path: str
    ) -> Optional[tuple]:
        """
        수정 코드 생성 (규칙 기반)
        
        반환: (원본 코드, 수정 코드, 설명)
        """
        issue_type = issue.get("type", "").upper()
        description = issue.get("description", "")
        line = issue.get("line", 0)
        
        # 파일 내용 읽기
        if not file_path or not Path(file_path).exists():
            return None
        
        try:
            lines = Path(file_path).read_text(encoding='utf-8').split('\n')
        except Exception:
            return None
        
        if line <= 0 or line > len(lines):
            return None
        
        original_line = lines[line - 1]
        fixed_line = original_line
        fix_description = ""
        
        # 간단한 규칙 기반 수정
        if issue_type == "READABILITY" and "긴 라인" in description:
            # 긴 라인 → 줄바꿈 제안
            if len(original_line) > 120:
                # 간단히 코멘트 추가
                fixed_line = original_line.rstrip() + "  # TODO: 이 라인을 분리하세요"
                fix_description = "Add TODO comment for long line"
        
        elif issue_type == "SECURITY" and "eval" in description.lower():
            # eval 사용 → 경고 코멘트 추가
            fixed_line = f"# WARNING: Security risk below\n{original_line}"
            fix_description = "Add security warning comment"
        
        elif issue_type == "PERFORMANCE" and "중첩 루프" in description:
            # 중첩 루프 → 최적화 제안 코멘트
            fixed_line = f"# TODO: Consider optimizing nested loop (O(n²))\n{original_line}"
            fix_description = "Add optimization suggestion"
        
        elif issue_type == "DOCUMENTATION" or issue_type == "IMPROVEMENT":
            # 문서화 필요 → docstring 제안
            if original_line.strip().startswith("def "):
                indent = len(original_line) - len(original_line.lstrip())
                docstring = ' ' * (indent + 4) + '"""TODO: Add docstring"""'
                fixed_line = original_line + "\n" + docstring
                fix_description = "Add docstring placeholder"
        
        if fixed_line == original_line:
            return None
        
        return (original_line, fixed_line, fix_description)
    
    def _calculate_confidence(
        self,
        issue: Dict[str, Any],
        fix_result: tuple
    ) -> float:
        """수정 신뢰도 계산"""
        base_confidence = 0.5
        
        # 간단한 수정일수록 높은 신뢰도
        original, fixed, _ = fix_result
        if "TODO" in fixed or "WARNING" in fixed:
            base_confidence += 0.3  # 코멘트 추가는 안전
        
        if len(fixed) < len(original) * 1.5:
            base_confidence += 0.1  # 적은 변경
        
        return min(1.0, base_confidence)
    
    def approve(
        self,
        proposal_id: str,
        approver: str = "system"
    ) -> bool:
        """수정 제안 승인"""
        if proposal_id not in self.proposals:
            logger.error(f"Proposal not found: {proposal_id}")
            return False
        
        proposal = self.proposals[proposal_id]
        proposal.status = FixStatus.APPROVED
        proposal.approved_at = time.time()
        proposal.approved_by = approver
        
        logger.info(f"✅ Proposal approved: {proposal_id} by {approver}")
        return True
    
    def reject(self, proposal_id: str, reason: str = "") -> bool:
        """수정 제안 거부"""
        if proposal_id not in self.proposals:
            return False
        
        proposal = self.proposals[proposal_id]
        proposal.status = FixStatus.REJECTED
        
        logger.info(f"❌ Proposal rejected: {proposal_id} - {reason}")
        return True
    
    def apply(self, proposal_id: str) -> bool:
        """
        승인된 수정 적용
        
        주의: 승인된 제안만 적용 가능
        """
        if proposal_id not in self.proposals:
            logger.error(f"Proposal not found: {proposal_id}")
            return False
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != FixStatus.APPROVED:
            logger.error(f"Proposal not approved: {proposal_id}")
            return False
        
        # 백업 생성
        backup = self._create_backup(proposal.file_path)
        if not backup:
            logger.error(f"Failed to create backup for: {proposal.file_path}")
            return False
        
        # 수정 적용
        try:
            file_path = Path(proposal.file_path)
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # 해당 라인 수정
            if 0 < proposal.line_start <= len(lines):
                lines[proposal.line_start - 1] = proposal.fixed_code
                
                new_content = '\n'.join(lines)
                file_path.write_text(new_content, encoding='utf-8')
                
                proposal.status = FixStatus.APPLIED
                proposal.applied_at = time.time()
                
                # 히스토리 기록
                self.history.append({
                    "action": "apply",
                    "proposal_id": proposal_id,
                    "backup_id": backup.id,
                    "timestamp": time.time()
                })
                
                logger.info(f"🔧 Fix applied: {proposal_id}")
                return True
        
        except Exception as e:
            logger.error(f"Failed to apply fix: {e}")
            # 롤백
            self.rollback(proposal_id)
        
        return False
    
    def _create_backup(self, file_path: str) -> Optional[Backup]:
        """파일 백업 생성"""
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            content = path.read_text(encoding='utf-8')
            backup_id = str(uuid.uuid4())[:8]
            backup_path = self.backup_dir / f"{backup_id}_{path.name}"
            
            shutil.copy2(file_path, backup_path)
            
            backup = Backup(
                id=backup_id,
                file_path=file_path,
                backup_path=str(backup_path),
                original_content=content
            )
            
            self.backups[backup_id] = backup
            return backup
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    def rollback(self, proposal_id: str) -> bool:
        """수정 되돌리기"""
        # 해당 제안의 백업 찾기
        for entry in reversed(self.history):
            if entry.get("proposal_id") == proposal_id:
                backup_id = entry.get("backup_id")
                if backup_id and backup_id in self.backups:
                    backup = self.backups[backup_id]
                    try:
                        Path(backup.file_path).write_text(
                            backup.original_content,
                            encoding='utf-8'
                        )
                        
                        self.proposals[proposal_id].status = FixStatus.ROLLED_BACK
                        logger.info(f"⏪ Rolled back: {proposal_id}")
                        return True
                    except Exception as e:
                        logger.error(f"Rollback failed: {e}")
        
        return False
    
    def auto_approve_safe_fixes(self) -> List[str]:
        """
        안전한 수정들 자동 승인
        
        조건:
        - 신뢰도 >= auto_approve_threshold
        - 위험도 <= max_risk_auto_approve
        """
        approved = []
        
        for proposal_id, proposal in self.proposals.items():
            if proposal.status != FixStatus.PENDING:
                continue
            
            if (proposal.confidence >= self.auto_approve_threshold and
                proposal.risk_level <= self.max_risk_auto_approve):
                self.approve(proposal_id, "auto")
                approved.append(proposal_id)
        
        logger.info(f"🤖 Auto-approved {len(approved)} fixes")
        return approved
    
    def get_pending_proposals(self) -> List[FixProposal]:
        """대기 중인 제안 조회"""
        return [p for p in self.proposals.values() if p.status == FixStatus.PENDING]
    
    def get_summary(self) -> Dict[str, Any]:
        """시스템 상태 요약"""
        by_status = {}
        by_category = {}
        
        for proposal in self.proposals.values():
            status = proposal.status.name
            by_status[status] = by_status.get(status, 0) + 1
            
            category = proposal.category.name
            by_category[category] = by_category.get(category, 0) + 1
        
        return {
            "total_proposals": len(self.proposals),
            "by_status": by_status,
            "by_category": by_category,
            "total_backups": len(self.backups),
            "history_entries": len(self.history)
        }
    
    def explain(self) -> str:
        return """
🔧 자동 수정 시스템 (Auto-Fix System)

안전 원칙:
  🔒 모든 수정은 먼저 제안으로 생성
  🔒 승인 없이는 절대 수정하지 않음
  🔒 모든 변경은 백업 후 진행
  🔒 되돌리기 항상 가능

사용법:
  system = AutoFixSystem()
  
  # 이슈에 대한 수정 제안 생성
  proposal = system.generate_fix_for_issue(issue, file_path)
  
  # 승인
  system.approve(proposal.id, "creator")
  
  # 적용
  system.apply(proposal.id)
  
  # 되돌리기
  system.rollback(proposal.id)

자동 승인:
  - 신뢰도 95% 이상
  - 위험도 2 이하 (스타일, 문서화)
  
철학적 의미:
  "자율적 개선, 그러나 통제된 자유"
"""


# 데모 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🔧 Auto-Fix System Demo")
    print("=" * 60)
    
    system = AutoFixSystem()
    
    # 샘플 이슈
    sample_issues = [
        {"type": "READABILITY", "description": "라인 125: 너무 긴 라인 (150자)", "line": 1, "severity": "low"},
        {"type": "SECURITY", "description": "라인 50: 위험한 패턴 'eval(' 발견", "line": 2, "severity": "critical"},
        {"type": "DOCUMENTATION", "description": "함수에 docstring 없음", "line": 3, "severity": "medium"},
    ]
    
    # 실제 파일로 테스트
    test_file = Path(__file__)
    
    print(f"\n📝 Generating fix proposals for {test_file.name}...")
    
    for issue in sample_issues:
        proposal = system.generate_fix_for_issue(issue, str(test_file))
        if proposal:
            print(f"  - {proposal.id}: {proposal.category.name} (risk: {proposal.risk_level})")
    
    print(f"\n📊 Summary:")
    summary = system.get_summary()
    print(f"  Total proposals: {summary['total_proposals']}")
    print(f"  By category: {summary['by_category']}")
    
    print("\n" + system.explain())
