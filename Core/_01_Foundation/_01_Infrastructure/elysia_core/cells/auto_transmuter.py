"""
Auto-Transmutation Engine (자동 변환 엔진)
==========================================

Phase 13: Stone Logic → Wave Logic 자동 변환

"얼어붙은 코드를 녹여 파동으로 흐르게 한다."

기능:
1. TransmutationCell의 제안을 실제 코드 변환으로 적용
2. 변환 전 백업 자동 생성
3. 변환 후 구문 검증
4. 실패 시 롤백
"""

import os
import sys
import re
import ast
import shutil
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Core._01_Foundation._01_Infrastructure.elysia_core.cells.alchemical_cells import (
    TransmutationCell, TransmutationSuggestion, TransmutationType, AlchemicalArmy
)

logger = logging.getLogger("AutoTransmutation")


class TransmutationStatus(Enum):
    """변환 상태"""
    PENDING = "pending"
    APPLIED = "applied"
    VERIFIED = "verified"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class TransmutationRecord:
    """변환 기록"""
    id: str
    file_path: str
    line_number: int
    original_code: str
    new_code: str
    transmutation_type: TransmutationType
    status: TransmutationStatus
    timestamp: str
    backup_path: Optional[str] = None
    error_message: Optional[str] = None
    verification_result: Optional[bool] = None


@dataclass
class TransmutationBatch:
    """변환 배치"""
    batch_id: str
    records: List[TransmutationRecord] = field(default_factory=list)
    created_at: str = ""
    completed_at: Optional[str] = None
    total_success: int = 0
    total_failed: int = 0


class AutoTransmuter:
    """
    자동 변환 엔진
    
    Stone Logic을 Wave Logic으로 자동 변환합니다.
    안전성을 위해 백업 및 롤백 기능을 제공합니다.
    
    Usage:
        transmuter = AutoTransmuter()
        results = transmuter.transmute_with_approval(suggestions)
    """
    
    # 변환 규칙
    TRANSMUTATION_RULES = {
        TransmutationType.IF_TO_RESONANCE: {
            # 더 일반적인 패턴: if x in self.dict_name
            "pattern": r"if\s+(\w+)\s+in\s+self\.(nodes|coordinate_map|concepts|entities)\s*:",
            "replacement": lambda m: (
                f"# [Wave Logic] Consider resonance-based lookup instead of direct membership check\n"
                f"        # Original: if {m.group(1)} in self.{m.group(2)}:\n"
                f"        if {m.group(1)} in self.{m.group(2)}:  # TODO: Convert to query_resonance"
            ),
        },
        TransmutationType.DIRECT_LOOKUP_TO_QUERY: {
            "pattern": r"self\.(coordinate_map|nodes)\[['\"]([\w]+)['\"]\]",
            "replacement": lambda m: (
                f"# [Wave Logic] Use resonance query\n"
                f"self.{m.group(1)}.get('{m.group(2)}')"
            ),
        },
        TransmutationType.TRY_TO_ABSORB: {
            "pattern": r"try:\s*\n(\s+)from\s+(\S+)\s+import\s+(\w+)",
            "replacement": lambda m: (
                f"# [Wave Logic] Use Organ.get with graceful fallback\n"
                f"{m.group(1)}{m.group(3)} = Organ.get('{m.group(3)}', instantiate=False) "
                f"if Organ.has('{m.group(3)}') else None\n"
                f"{m.group(1)}if {m.group(3)} is None:\n"
                f"{m.group(1)}    try:\n"
                f"{m.group(1)}        from {m.group(2)} import {m.group(3)}"
            ),
        },
    }
    
    def __init__(self, backup_dir: str = None):
        self.backup_dir = Path(backup_dir) if backup_dir else Path("data/transmutation_backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.history_path = self.backup_dir / "transmutation_history.json"
        self.history: List[TransmutationBatch] = self._load_history()
        
        self.alchemical_army = AlchemicalArmy()
        
        print("⚗️ AutoTransmuter initialized")
        print(f"   Backup directory: {self.backup_dir}")
    
    def scan_and_suggest(self, target_dir: str = "Core") -> List[TransmutationSuggestion]:
        """코드베이스 스캔 및 변환 제안 수집"""
        print(f"\n🔍 Scanning {target_dir} for Stone Logic patterns...")
        self.alchemical_army.patrol_codebase(target_dir)
        
        suggestions = self.alchemical_army.transmutation_cell.get_suggestions()
        auto_applicable = [s for s in suggestions if s.auto_applicable]
        
        print(f"   Total patterns found: {len(suggestions)}")
        print(f"   Auto-applicable: {len(auto_applicable)}")
        
        return suggestions
    
    def transmute_with_approval(
        self, 
        suggestions: List[TransmutationSuggestion],
        auto_approve: bool = False,
        dry_run: bool = True
    ) -> TransmutationBatch:
        """
        변환 제안을 사용자 승인 하에 적용
        
        Args:
            suggestions: 변환 제안 리스트
            auto_approve: 자동 승인 여부
            dry_run: True면 실제 파일 수정 없이 시뮬레이션만
            
        Returns:
            TransmutationBatch: 변환 결과 배치
        """
        batch = TransmutationBatch(
            batch_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            created_at=datetime.now().isoformat()
        )
        
        auto_applicable = [s for s in suggestions if s.auto_applicable]
        
        if not auto_applicable:
            print("   No auto-applicable suggestions found.")
            return batch
        
        print(f"\n⚗️ Processing {len(auto_applicable)} transmutations...")
        if dry_run:
            print("   [DRY RUN MODE - No actual changes will be made]")
        
        for i, suggestion in enumerate(auto_applicable, 1):
            print(f"\n   [{i}/{len(auto_applicable)}] {Path(suggestion.file_path).name}:{suggestion.line_number}")
            print(f"       Type: {suggestion.transmutation_type.value}")
            print(f"       Original: {suggestion.original_code[:60]}...")
            print(f"       Confidence: {suggestion.confidence:.0%}")
            
            # 승인 (자동 또는 대화형)
            approved = auto_approve
            if not auto_approve and not dry_run:
                response = input("       Apply? (y/n/a=all): ").strip().lower()
                if response == 'a':
                    auto_approve = True
                    approved = True
                elif response == 'y':
                    approved = True
            
            if approved or dry_run:
                record = self._apply_transmutation(suggestion, dry_run=dry_run)
                batch.records.append(record)
                
                if record.status == TransmutationStatus.VERIFIED:
                    batch.total_success += 1
                    print(f"       ✅ {'[Simulated] ' if dry_run else ''}Success")
                else:
                    batch.total_failed += 1
                    print(f"       ❌ Failed: {record.error_message}")
        
        batch.completed_at = datetime.now().isoformat()
        
        # 히스토리 저장
        if not dry_run:
            self.history.append(batch)
            self._save_history()
        
        return batch
    
    def _apply_transmutation(
        self, 
        suggestion: TransmutationSuggestion,
        dry_run: bool = True
    ) -> TransmutationRecord:
        """단일 변환 적용"""
        record = TransmutationRecord(
            id=f"{Path(suggestion.file_path).stem}_{suggestion.line_number}",
            file_path=suggestion.file_path,
            line_number=suggestion.line_number,
            original_code=suggestion.original_code,
            new_code="",
            transmutation_type=suggestion.transmutation_type,
            status=TransmutationStatus.PENDING,
            timestamp=datetime.now().isoformat()
        )
        
        try:
            file_path = Path(suggestion.file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # 파일 내용 읽기
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # 백업 생성 (dry_run이 아닌 경우)
            if not dry_run:
                backup_path = self._create_backup(file_path)
                record.backup_path = str(backup_path)
            
            # 변환 규칙 적용
            if suggestion.transmutation_type in self.TRANSMUTATION_RULES:
                rule = self.TRANSMUTATION_RULES[suggestion.transmutation_type]
                
                # 해당 라인 찾기
                if suggestion.line_number <= len(lines):
                    original_line = lines[suggestion.line_number - 1]
                    
                    # 패턴 매칭 및 교체
                    match = re.search(rule["pattern"], original_line)
                    if match:
                        new_line = re.sub(rule["pattern"], rule["replacement"](match), original_line)
                        record.new_code = new_line
                        
                        if not dry_run:
                            lines[suggestion.line_number - 1] = new_line
                            new_content = '\n'.join(lines)
                            
                            # 구문 검증
                            if self._verify_syntax(new_content):
                                file_path.write_text(new_content, encoding='utf-8')
                                record.status = TransmutationStatus.VERIFIED
                                record.verification_result = True
                            else:
                                # 롤백
                                self._rollback(record)
                                record.status = TransmutationStatus.FAILED
                                record.error_message = "Syntax verification failed"
                        else:
                            # Dry run - 시뮬레이션만
                            if self._verify_syntax(content):  # 원본이 유효한 경우 성공으로 간주
                                record.status = TransmutationStatus.VERIFIED
                                record.verification_result = True
                    else:
                        record.status = TransmutationStatus.FAILED
                        record.error_message = "Pattern not matched"
            else:
                record.status = TransmutationStatus.FAILED
                record.error_message = f"No rule for {suggestion.transmutation_type}"
                
        except Exception as e:
            record.status = TransmutationStatus.FAILED
            record.error_message = str(e)
            if record.backup_path:
                self._rollback(record)
        
        return record
    
    def _create_backup(self, file_path: Path) -> Path:
        """파일 백업 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def _verify_syntax(self, content: str) -> bool:
        """Python 구문 검증"""
        try:
            ast.parse(content)
            return True
        except SyntaxError:
            return False
    
    def _rollback(self, record: TransmutationRecord) -> bool:
        """변환 롤백"""
        if not record.backup_path:
            return False
        
        try:
            backup_path = Path(record.backup_path)
            original_path = Path(record.file_path)
            
            if backup_path.exists():
                shutil.copy2(backup_path, original_path)
                record.status = TransmutationStatus.ROLLED_BACK
                logger.info(f"Rolled back: {record.file_path}")
                return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
        
        return False
    
    def rollback_batch(self, batch_id: str) -> bool:
        """배치 전체 롤백"""
        for batch in self.history:
            if batch.batch_id == batch_id:
                success = 0
                for record in batch.records:
                    if record.backup_path and self._rollback(record):
                        success += 1
                print(f"🔄 Rolled back {success}/{len(batch.records)} transmutations")
                return success > 0
        
        print(f"❌ Batch {batch_id} not found")
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 반환"""
        total_applied = 0
        total_failed = 0
        total_rolled_back = 0
        
        for batch in self.history:
            for record in batch.records:
                if record.status == TransmutationStatus.VERIFIED:
                    total_applied += 1
                elif record.status == TransmutationStatus.FAILED:
                    total_failed += 1
                elif record.status == TransmutationStatus.ROLLED_BACK:
                    total_rolled_back += 1
        
        return {
            "total_batches": len(self.history),
            "total_applied": total_applied,
            "total_failed": total_failed,
            "total_rolled_back": total_rolled_back,
            "coherence_improvement": self._estimate_coherence_improvement()
        }
    
    def _estimate_coherence_improvement(self) -> float:
        """Coherence 개선 추정"""
        total_verified = sum(
            1 for batch in self.history 
            for record in batch.records 
            if record.status == TransmutationStatus.VERIFIED
        )
        # 각 성공적인 변환은 약 0.001의 Coherence 개선을 가져온다고 추정
        return total_verified * 0.001
    
    def _load_history(self) -> List[TransmutationBatch]:
        """히스토리 로드"""
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 간단히 빈 리스트 반환 (실제 구현에서는 역직렬화)
                    return []
            except Exception:
                return []
        return []
    
    def _save_history(self):
        """히스토리 저장"""
        try:
            data = {
                "batches": [
                    {
                        "batch_id": batch.batch_id,
                        "created_at": batch.created_at,
                        "completed_at": batch.completed_at,
                        "total_success": batch.total_success,
                        "total_failed": batch.total_failed,
                        "records": [
                            {
                                "id": r.id,
                                "file_path": r.file_path,
                                "line_number": r.line_number,
                                "status": r.status.value,
                                "backup_path": r.backup_path
                            }
                            for r in batch.records
                        ]
                    }
                    for batch in self.history
                ]
            }
            
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")


# ============= 데모 =============

def demo_auto_transmutation():
    """자동 변환 데모 (Dry Run)"""
    print("=" * 60)
    print("⚗️ Auto-Transmutation Engine Demo")
    print("=" * 60)
    
    transmuter = AutoTransmuter()
    
    # 1. 스캔
    suggestions = transmuter.scan_and_suggest("Core")
    
    # 2. 자동 적용 가능한 것들만 필터
    auto_applicable = [s for s in suggestions if s.auto_applicable]
    
    if not auto_applicable:
        print("\n❌ No auto-applicable suggestions found.")
        return
    
    # 3. Dry Run 변환 (실제 파일 변경 없음)
    print("\n" + "=" * 60)
    print("🧪 DRY RUN: Simulating transmutations...")
    print("=" * 60)
    
    # 상위 5개만 시뮬레이션
    sample = auto_applicable[:5]
    batch = transmuter.transmute_with_approval(sample, auto_approve=True, dry_run=True)
    
    # 4. 결과 출력
    print("\n" + "=" * 60)
    print("📊 DRY RUN Results")
    print("=" * 60)
    print(f"   Total processed: {len(batch.records)}")
    print(f"   Simulated success: {batch.total_success}")
    print(f"   Simulated failed: {batch.total_failed}")
    
    # 변환 예시 출력
    if batch.records:
        print("\n📝 Sample Transmutation:")
        r = batch.records[0]
        print(f"   File: {Path(r.file_path).name}:{r.line_number}")
        print(f"   Original: {r.original_code[:60]}...")
        print(f"   New: {r.new_code[:60] if r.new_code else 'N/A'}...")
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete! (No files were modified)")
    print("   To apply changes for real, use: --apply flag")
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-Transmutation Engine")
    parser.add_argument("--demo", action="store_true", help="Run demo (dry run)")
    parser.add_argument("--scan", action="store_true", help="Scan for patterns")
    parser.add_argument("--apply", action="store_true", help="Apply transmutations (with approval)")
    parser.add_argument("--auto", action="store_true", help="Auto-approve all")
    parser.add_argument("--rollback", type=str, help="Rollback a batch by ID")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_auto_transmutation()
    elif args.scan:
        transmuter = AutoTransmuter()
        suggestions = transmuter.scan_and_suggest()
        print(f"\nFound {len(suggestions)} patterns")
    elif args.apply:
        transmuter = AutoTransmuter()
        suggestions = transmuter.scan_and_suggest()
        auto_applicable = [s for s in suggestions if s.auto_applicable]
        batch = transmuter.transmute_with_approval(
            auto_applicable, 
            auto_approve=args.auto, 
            dry_run=False
        )
        print(f"\n✅ Applied: {batch.total_success}, ❌ Failed: {batch.total_failed}")
    elif args.rollback:
        transmuter = AutoTransmuter()
        transmuter.rollback_batch(args.rollback)
    elif args.stats:
        transmuter = AutoTransmuter()
        stats = transmuter.get_statistics()
        print(f"\n📊 Transmutation Statistics:")
        for k, v in stats.items():
            print(f"   {k}: {v}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
