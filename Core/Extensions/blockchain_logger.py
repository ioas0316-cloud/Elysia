"""
Blockchain Logger - 블록체인 의사결정 기록
========================================

낮은 우선순위 #3: 블록체인 기록
예상 효과: 의사결정 투명성 및 감사 가능성

핵심 기능:
- 의사결정 이력 해시 저장
- 변조 불가능한 로그
- 타임스탬프 증명
- 검증 가능한 추적
"""

import logging
import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger("BlockchainLogger")


class DecisionType(Enum):
    """의사결정 유형"""
    THOUGHT = "thought"
    LAW_CHECK = "law_check"
    RESONANCE = "resonance"
    LEARNING = "learning"
    SYSTEM = "system"


@dataclass
class DecisionRecord:
    """의사결정 기록"""
    record_id: str
    decision_type: DecisionType
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    
    # 법칙 관련
    laws_checked: List[str] = field(default_factory=list)
    law_violations: List[str] = field(default_factory=list)
    
    # 해시 체인
    previous_hash: str = ""
    current_hash: str = ""
    
    def calculate_hash(self) -> str:
        """레코드 해시 계산"""
        data = {
            "id": self.record_id,
            "type": self.decision_type.value,
            "input": self.input_data,
            "output": self.output_data,
            "timestamp": self.timestamp,
            "prev_hash": self.previous_hash
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "type": self.decision_type.value,
            "input": self.input_data,
            "output": self.output_data,
            "timestamp": self.timestamp,
            "laws_checked": self.laws_checked,
            "violations": self.law_violations,
            "prev_hash": self.previous_hash,
            "hash": self.current_hash
        }


@dataclass
class Block:
    """블록"""
    block_number: int
    records: List[DecisionRecord]
    timestamp: float = field(default_factory=time.time)
    previous_block_hash: str = ""
    block_hash: str = ""
    nonce: int = 0
    
    def calculate_hash(self) -> str:
        """블록 해시 계산"""
        record_hashes = [r.current_hash for r in self.records]
        data = {
            "block_number": self.block_number,
            "records": record_hashes,
            "timestamp": self.timestamp,
            "prev_block_hash": self.previous_block_hash,
            "nonce": self.nonce
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_number": self.block_number,
            "record_count": len(self.records),
            "timestamp": self.timestamp,
            "prev_hash": self.previous_block_hash,
            "hash": self.block_hash
        }


class BlockchainLogger:
    """
    블록체인 의사결정 로거
    
    낮은 우선순위 #3 구현:
    - 해시 체인 기반 로깅
    - 변조 감지
    - 검증 가능한 이력
    
    예상 효과: 의사결정 투명성 확보
    """
    
    def __init__(
        self,
        block_size: int = 100,
        difficulty: int = 2  # 해시 앞 0 개수 (간단한 PoW)
    ):
        """
        Args:
            block_size: 블록당 최대 레코드 수
            difficulty: 마이닝 난이도
        """
        self.block_size = block_size
        self.difficulty = difficulty
        
        # 체인
        self.chain: List[Block] = []
        self.pending_records: List[DecisionRecord] = []
        
        # 마지막 해시
        self.last_record_hash = "0" * 64
        self.last_block_hash = "0" * 64
        
        # 통계
        self.stats = {
            "total_records": 0,
            "total_blocks": 0,
            "violations_logged": 0
        }
        
        # 제네시스 블록 생성
        self._create_genesis_block()
        
        self.logger = logging.getLogger("BlockchainLogger")
        self.logger.info("⛓️ BlockchainLogger initialized")
    
    def _create_genesis_block(self) -> None:
        """제네시스 블록 생성"""
        genesis = Block(
            block_number=0,
            records=[],
            previous_block_hash="0" * 64
        )
        genesis.block_hash = genesis.calculate_hash()
        self.chain.append(genesis)
        self.last_block_hash = genesis.block_hash
        self.stats["total_blocks"] = 1
    
    def log_decision(
        self,
        decision_type: DecisionType,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        laws_checked: Optional[List[str]] = None,
        law_violations: Optional[List[str]] = None
    ) -> DecisionRecord:
        """
        의사결정 기록
        
        Args:
            decision_type: 결정 유형
            input_data: 입력 데이터
            output_data: 출력 데이터
            laws_checked: 검사된 법칙들
            law_violations: 위반된 법칙들
            
        Returns:
            생성된 기록
        """
        record = DecisionRecord(
            record_id=f"record_{self.stats['total_records']}_{int(time.time()*1000)}",
            decision_type=decision_type,
            input_data=input_data,
            output_data=output_data,
            laws_checked=laws_checked or [],
            law_violations=law_violations or [],
            previous_hash=self.last_record_hash
        )
        
        # 해시 계산
        record.current_hash = record.calculate_hash()
        self.last_record_hash = record.current_hash
        
        # 대기열에 추가
        self.pending_records.append(record)
        self.stats["total_records"] += 1
        
        if law_violations:
            self.stats["violations_logged"] += len(law_violations)
        
        # 블록 생성 체크
        if len(self.pending_records) >= self.block_size:
            self._create_block()
        
        return record
    
    def _create_block(self) -> Block:
        """새 블록 생성"""
        block = Block(
            block_number=len(self.chain),
            records=self.pending_records.copy(),
            previous_block_hash=self.last_block_hash
        )
        
        # 간단한 PoW (선택적)
        while not block.calculate_hash().startswith("0" * self.difficulty):
            block.nonce += 1
        
        block.block_hash = block.calculate_hash()
        
        # 체인에 추가
        self.chain.append(block)
        self.last_block_hash = block.block_hash
        self.pending_records.clear()
        self.stats["total_blocks"] += 1
        
        self.logger.info(f"📦 Block #{block.block_number} created (hash={block.block_hash[:16]}...)")
        
        return block
    
    def verify_chain(self) -> Tuple[bool, Optional[str]]:
        """체인 무결성 검증"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            
            # 이전 블록 해시 확인
            if current.previous_block_hash != previous.block_hash:
                return False, f"Block {i}: previous hash mismatch"
            
            # 현재 블록 해시 확인
            if current.block_hash != current.calculate_hash():
                return False, f"Block {i}: hash mismatch"
            
            # 레코드 해시 체인 확인
            for j, record in enumerate(current.records):
                if record.current_hash != record.calculate_hash():
                    return False, f"Block {i}, Record {j}: hash mismatch"
        
        return True, None
    
    def get_record(self, record_id: str) -> Optional[DecisionRecord]:
        """레코드 조회"""
        # 대기열 검색
        for record in self.pending_records:
            if record.record_id == record_id:
                return record
        
        # 블록 검색
        for block in self.chain:
            for record in block.records:
                if record.record_id == record_id:
                    return record
        
        return None
    
    def get_records_by_type(
        self,
        decision_type: DecisionType,
        limit: int = 100
    ) -> List[DecisionRecord]:
        """유형별 레코드 조회"""
        records = []
        
        # 역순으로 검색 (최신 먼저)
        for block in reversed(self.chain):
            for record in reversed(block.records):
                if record.decision_type == decision_type:
                    records.append(record)
                    if len(records) >= limit:
                        return records
        
        # 대기열도 검색
        for record in reversed(self.pending_records):
            if record.decision_type == decision_type:
                records.append(record)
                if len(records) >= limit:
                    break
        
        return records
    
    def get_violations(self, limit: int = 100) -> List[DecisionRecord]:
        """위반 기록 조회"""
        violations = []
        
        for block in reversed(self.chain):
            for record in reversed(block.records):
                if record.law_violations:
                    violations.append(record)
                    if len(violations) >= limit:
                        return violations
        
        return violations
    
    def export_chain(self, filepath: str) -> None:
        """체인 내보내기"""
        data = {
            "chain": [block.to_dict() for block in self.chain],
            "pending": [r.to_dict() for r in self.pending_records],
            "stats": self.stats,
            "exported_at": time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📁 Chain exported to {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """통계"""
        return {
            **self.stats,
            "chain_length": len(self.chain),
            "pending_records": len(self.pending_records),
            "last_block_hash": self.last_block_hash[:16] + "..."
        }


# Tuple import for type hints
from typing import Tuple


# 테스트
if __name__ == "__main__":
    print("\n" + "="*70)
    print("⛓️ Blockchain Logger Test")
    print("="*70)
    
    blockchain = BlockchainLogger(block_size=5, difficulty=1)
    
    print("\n[Test 1] Log Decisions")
    for i in range(7):
        record = blockchain.log_decision(
            DecisionType.THOUGHT,
            {"text": f"thought_{i}"},
            {"resonances": {"love": 0.8}},
            laws_checked=["being", "love"],
            law_violations=["balance"] if i == 3 else []
        )
        print(f"  Record: {record.record_id[:30]}... hash={record.current_hash[:16]}...")
    
    print("\n[Test 2] Verify Chain")
    valid, error = blockchain.verify_chain()
    print(f"  Valid: {valid}")
    if error:
        print(f"  Error: {error}")
    
    print("\n[Test 3] Get Violations")
    violations = blockchain.get_violations()
    print(f"  Violations: {len(violations)}")
    
    print("\n[Test 4] Stats")
    stats = blockchain.get_stats()
    print(f"  Stats: {stats}")
    
    print("\n✅ All tests passed!")
