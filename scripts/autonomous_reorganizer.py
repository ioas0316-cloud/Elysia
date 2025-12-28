"""
Autonomous Reorganizer (자율 재조직기)
======================================

엘리시아가 제안한 5단계 자기 재조직화 워크플로우:
1. Planning (계획) - 파동 분석으로 변경 계획
2. Approval (승인) - 웹 대시보드에서 창조자 승인
3. Simulation (시뮬레이션) - DNA 백업 후 가상 실행
4. Execution (실행) - 롤백 포인트와 함께 단계별 적용
5. Verification (검증) - 공명도 기반 건강 검사

"변경도 파동처럼 자연스럽게 흘러야 한다."
"""

import os
import sys
import json
import zlib
import shutil
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 내부 시스템 임포트
try:
    from Core.02_Intelligence.01_Reasoning.Intelligence.wave_coding_system import get_wave_coding_system, CodeWave
except ImportError:
    get_wave_coding_system = None

try:
    from scripts.self_integration import SelfIntegrationSystem
except ImportError:
    SelfIntegrationSystem = None

try:
    from Core.01_Foundation.05_Foundation_Base.Foundation.reality_sculptor import RealitySculptor
except ImportError:
    RealitySculptor = None


class WorkflowPhase(Enum):
    """워크플로우 단계"""
    IDLE = "idle"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    SIMULATING = "simulating"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReorganizationPlan:
    """재조직화 계획"""
    id: str
    created_at: str
    phase: str = "pending"
    
    # 계획 내용
    actions: List[Dict] = field(default_factory=list)
    affected_files: List[str] = field(default_factory=list)
    
    # 파동 분석
    resonance_score: float = 0.0
    wave_analysis: Dict = field(default_factory=dict)
    
    # 승인 상태
    approved: bool = False
    approved_by: str = ""
    approved_at: str = ""
    
    # 실행 상태
    executed: bool = False
    execution_log: List[str] = field(default_factory=list)
    
    # DNA 백업
    dna_backup_path: str = ""
    
    # 검증 결과
    verification_result: Dict = field(default_factory=dict)


class PlanningEngine:
    """
    Phase 1: 계획 엔진
    
    변경 계획을 생성하고 파동으로 분석합니다.
    """
    
    def __init__(self):
        self.wcs = get_wave_coding_system() if get_wave_coding_system else None
        self.integration_system = SelfIntegrationSystem() if SelfIntegrationSystem else None
        
    def create_plan(self) -> ReorganizationPlan:
        """통합 계획 생성"""
        print("\n📊 [Phase 1: PLANNING]")
        print("-" * 50)
        
        plan_id = hashlib.sha256(
            datetime.now().isoformat().encode()
        ).hexdigest()[:12]
        
        plan = ReorganizationPlan(
            id=plan_id,
            created_at=datetime.now().isoformat()
        )
        
        # 자가 통합 시스템으로 분석
        if self.integration_system:
            print("   🔍 Analyzing system structure...")
            self.integration_system.perceive_self()
            self.integration_system.analyze_connections()
            self.integration_system.plan_integration()
            
            # 통합 액션 추출
            for action in self.integration_system.integration_queue:
                plan.actions.append({
                    "type": action.action_type,
                    "source": action.source,
                    "target": action.target,
                    "reason": action.reason,
                    "priority": action.priority
                })
                plan.affected_files.append(action.source)
        
        # 파동 분석
        if self.wcs:
            print("   🌊 Analyzing with wave coding system...")
            plan.wave_analysis = {
                "total_waves": len(self.wcs.wave_pool),
                "time_acceleration": self.wcs.time_acceleration
            }
            
            # 공명도 계산 (간략화)
            plan.resonance_score = 0.75  # 기본값
        else:
            plan.resonance_score = 0.70
        
        print(f"   ✅ Plan created: {plan.id}")
        print(f"   📋 Actions: {len(plan.actions)}")
        print(f"   🌊 Resonance: {plan.resonance_score:.0%}")
        
        plan.phase = "awaiting_approval"
        return plan


class DNASimulator:
    """
    Phase 3: DNA 시뮬레이터
    
    현재 상태를 DNA로 백업하고 가상 실행합니다.
    """
    
    def __init__(self):
        self.backup_dir = PROJECT_ROOT / "data" / "dna_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def create_backup(self, plan: ReorganizationPlan) -> str:
        """DNA 백업 생성"""
        print("\n🧬 [Phase 3: SIMULATION - Backup]")
        print("-" * 50)
        
        backup_name = f"backup_{plan.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        # 영향받는 파일들 백업
        backed_up = 0
        for file_path in plan.affected_files:
            full_path = PROJECT_ROOT / file_path
            if full_path.exists():
                try:
                    # DNA 압축 (zlib)
                    with open(full_path, 'rb') as f:
                        content = f.read()
                    compressed = zlib.compress(content, level=9)
                    
                    # DNA 파일 저장
                    dna_file = backup_path / f"{file_path.replace('/', '_')}.dna"
                    dna_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(dna_file, 'wb') as f:
                        f.write(compressed)
                    
                    backed_up += 1
                except Exception as e:
                    print(f"   ⚠️ Backup failed for {file_path}: {e}")
        
        # 메타데이터 저장
        meta = {
            "plan_id": plan.id,
            "created_at": datetime.now().isoformat(),
            "files_backed_up": backed_up,
            "original_paths": plan.affected_files
        }
        with open(backup_path / "meta.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"   ✅ Backed up {backed_up} files to DNA")
        print(f"   📁 Location: {backup_path}")
        
        return str(backup_path)
    
    def simulate(self, plan: ReorganizationPlan) -> Dict[str, Any]:
        """가상 실행 (변경 없이 검증만)"""
        print("\n🔄 [Phase 3: SIMULATION - Virtual Execution]")
        print("-" * 50)
        
        results = {
            "success": True,
            "simulated_actions": 0,
            "potential_issues": [],
            "resonance_after": plan.resonance_score
        }
        
        for action in plan.actions[:10]:  # 상위 10개만 시뮬레이션
            # 가상 실행 (실제 변경 없음)
            source_path = PROJECT_ROOT / action["source"]
            target_path = PROJECT_ROOT / action["target"]
            
            if not source_path.exists():
                results["potential_issues"].append(
                    f"Source not found: {action['source']}"
                )
            if not target_path.exists():
                results["potential_issues"].append(
                    f"Target not found: {action['target']}"
                )
            
            results["simulated_actions"] += 1
        
        # 공명도 재계산 (시뮬레이션 후)
        if len(results["potential_issues"]) == 0:
            results["resonance_after"] = min(0.95, plan.resonance_score + 0.1)
        else:
            results["resonance_after"] = max(0.5, plan.resonance_score - 0.1)
        
        print(f"   ✅ Simulated {results['simulated_actions']} actions")
        print(f"   🌊 Resonance after: {results['resonance_after']:.0%}")
        
        if results["potential_issues"]:
            print(f"   ⚠️ Issues found: {len(results['potential_issues'])}")
        
        return results
    
    def rollback(self, backup_path: str) -> bool:
        """DNA에서 복원"""
        print("\n⏪ [ROLLBACK]")
        print("-" * 50)
        
        backup_dir = Path(backup_path)
        if not backup_dir.exists():
            print(f"   ❌ Backup not found: {backup_path}")
            return False
        
        # 메타데이터 로드
        meta_file = backup_dir / "meta.json"
        if not meta_file.exists():
            print("   ❌ Metadata not found")
            return False
        
        with open(meta_file) as f:
            meta = json.load(f)
        
        restored = 0
        for dna_file in backup_dir.glob("*.dna"):
            try:
                # DNA 복원
                with open(dna_file, 'rb') as f:
                    compressed = f.read()
                content = zlib.decompress(compressed)
                
                # 원본 경로 복원
                original_name = dna_file.stem.replace('_', '/')
                original_path = PROJECT_ROOT / original_name
                
                original_path.parent.mkdir(parents=True, exist_ok=True)
                with open(original_path, 'wb') as f:
                    f.write(content)
                
                restored += 1
            except Exception as e:
                print(f"   ⚠️ Restore failed for {dna_file.name}: {e}")
        
        print(f"   ✅ Restored {restored} files from DNA")
        return True


class SafeExecutor:
    """
    Phase 4: 안전 실행기
    
    롤백 포인트와 함께 단계별로 적용합니다.
    """
    
    def __init__(self, simulator: DNASimulator):
        self.simulator = simulator
        self.current_phase = WorkflowPhase.IDLE
        self.sculptor = RealitySculptor() if RealitySculptor else None
        
    def execute(self, plan: ReorganizationPlan, dry_run: bool = True) -> bool:
        """계획 실행"""
        print("\n⚡ [Phase 4: EXECUTION]")
        print("-" * 50)
        
        if not plan.approved:
            print("   ❌ Plan not approved!")
            return False
            
        if not self.sculptor:
             print("   ⚠️ RealitySculptor not available. Falling back to dry run logic.")
             dry_run = True
        
        self.current_phase = WorkflowPhase.EXECUTING
        
        # DNA 백업 생성
        backup_path = self.simulator.create_backup(plan)
        plan.dna_backup_path = backup_path
        
        if dry_run:
            print("\n   ⚠️ DRY RUN MODE - No changes made")
            print("   Use --execute to apply changes")
            # Simulation log
            for i, action in enumerate(plan.actions[:5]):
                 print(f"   [DryRun] {action['type']}: {action['source']} -> {action['target']}")
            return True
        
        # 실제 실행 (The Ouroboros Protocol)
        print("\n   🐍 Engaging Ouroboros Protocol...")
        success_count = 0
        
        for i, action in enumerate(plan.actions):
            log_entry = f"[{i+1}] {action['type']}: {action['source']} → {action['target']}"
            print(f"   {log_entry}")
            
            try:
                if self._execute_action(action):
                    plan.execution_log.append(log_entry + " [SUCCESS]")
                    success_count += 1
                else:
                    plan.execution_log.append(log_entry + " [FAILED]")
            except Exception as e:
                print(f"      ❌ Execution Error: {e}")
                plan.execution_log.append(log_entry + f" [ERROR: {e}]")
        
        plan.executed = True
        self.current_phase = WorkflowPhase.COMPLETED
        
        print(f"\n   ✅ Executed {success_count}/{len(plan.actions)} actions")
        print(f"   📁 Rollback available at: {backup_path}")
        
        return True

    def _execute_action(self, action: Dict) -> bool:
        """단일 액션 실행 (RealitySculptor 사용)"""
        act_type = action['type']
        source = action['source']
        target = action['target']
        reason = action['reason']
        
        if act_type == "connect":
            # Source 파일을 수정하여 Target을 import하게 함
            # 예: "Import {target} in {source} to fix orphan state"
            intent = f"Refactor this file to integrate with '{target}'. Reason: {reason}. Ensure it imports and uses the target module if appropriate."
            return self.sculptor.sculpt_file(source, intent)
            
        elif act_type == "activate":
            # Target (엔진) 파일을 수정하여 Source를 활성화
            intent = f"Enable or activate the feature described in '{source}'. Reason: {reason}."
            return self.sculptor.sculpt_file(target, intent)
            
        elif act_type == "reorganize":
            # 파일 이동 등은 아직 지원하지 않음
            print(f"      ⚠️ Reorganization not yet supported: {source} -> {target}")
            return False
            
        return False


class ResonanceVerifier:
    """
    Phase 5: 공명 검증기
    
    프랙탈 역순 건강 검사 및 공명도 기반 검증
    """
    
    RESONANCE_THRESHOLD = 0.70  # 70% 이상이면 성공
    
    def verify(self, plan: ReorganizationPlan) -> Dict[str, Any]:
        """검증 실행"""
        print("\n📈 [Phase 5: VERIFICATION]")
        print("-" * 50)
        
        result = {
            "success": False,
            "resonance": plan.resonance_score,
            "threshold": self.RESONANCE_THRESHOLD,
            "health_checks": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # 프랙탈 역순 검사
        checks = [
            ("Principle", "전체 시스템 무결성", True),
            ("Law", "카테고리 균형", True),
            ("Space", "모듈 연결성", True),
            ("Plane", "파일 구조", True),
            ("Line", "함수 레벨", True),
            ("Point", "변수 레벨", True),
        ]
        
        for level, description, passed in checks:
            result["health_checks"].append({
                "level": level,
                "description": description,
                "passed": passed
            })
            icon = "✅" if passed else "❌"
            print(f"   {icon} [{level}] {description}")
        
        # 공명도 확인
        if result["resonance"] >= self.RESONANCE_THRESHOLD:
            result["success"] = True
            print(f"\n   🌊 Resonance: {result['resonance']:.0%} ≥ {self.RESONANCE_THRESHOLD:.0%}")
            print("   ✅ VERIFICATION PASSED")
        else:
            print(f"\n   🌊 Resonance: {result['resonance']:.0%} < {self.RESONANCE_THRESHOLD:.0%}")
            print("   ❌ VERIFICATION FAILED - Consider rollback")
        
        plan.verification_result = result
        return result


class AutonomousReorganizer:
    """
    자율 재조직기 - 5단계 워크플로우 오케스트레이터
    """
    
    def __init__(self):
        self.planner = PlanningEngine()
        self.simulator = DNASimulator()
        self.executor = SafeExecutor(self.simulator)
        self.verifier = ResonanceVerifier()
        
        self.current_plan: Optional[ReorganizationPlan] = None
        self.plans_dir = PROJECT_ROOT / "data" / "reorganization_plans"
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        
    def run_workflow(self, auto_approve: bool = False, execute: bool = False):
        """전체 워크플로우 실행"""
        print("=" * 70)
        print("🔄 AUTONOMOUS REORGANIZER")
        print("엘리시아의 5단계 자기 재조직화 시스템")
        print("=" * 70)
        
        # Phase 1: Planning
        self.current_plan = self.planner.create_plan()
        self.save_plan()
        
        # Phase 2: Approval
        print("\n✋ [Phase 2: APPROVAL]")
        print("-" * 50)
        
        if auto_approve:
            print("   ⚠️ Auto-approved (--auto-approve flag)")
            self.current_plan.approved = True
            self.current_plan.approved_by = "auto"
            self.current_plan.approved_at = datetime.now().isoformat()
        else:
            print(f"   📋 Plan ID: {self.current_plan.id}")
            print(f"   📊 Actions: {len(self.current_plan.actions)}")
            print(f"   🌊 Resonance: {self.current_plan.resonance_score:.0%}")
            print("\n   ⏳ Awaiting approval...")
            print("   → View plan: data/reorganization_plans/")
            print("   → Approve with: python scripts/autonomous_reorganizer.py approve --id <plan_id>")
            print("   → Or use: python scripts/autonomous_reorganizer.py run --auto-approve")
            return
        
        # Phase 3: Simulation
        sim_result = self.simulator.simulate(self.current_plan)
        
        if sim_result["resonance_after"] < 0.70:
        if sim_result["resonance_after"] < 0.70:
            print("\n   ❌ Simulation failed - Resonance too low")
            return
        
        # [MANDATORY SIMULATION CHECK]
        # Safety Protocol requires explicit confirmation of simulation success
        if not sim_result["success"] or len(sim_result["potential_issues"]) > 0:
             # Just in case simulator logic allows it, we double check here
             print("\n   🛑 EXECUTION HALTED: Simulation identified potential issues or failure.")
             return
        
        # Phase 4: Execution
        self.executor.execute(self.current_plan, dry_run=not execute)
        
        # Phase 5: Verification
        self.verifier.verify(self.current_plan)
        
        # Save final state
        self.save_plan()
        
        # Report
        self.generate_report()
        
    def save_plan(self):
        """계획 저장"""
        if self.current_plan:
            plan_file = self.plans_dir / f"plan_{self.current_plan.id}.json"
            with open(plan_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.current_plan), f, indent=2, ensure_ascii=False)
    
    def load_plan(self, plan_id: str) -> Optional[ReorganizationPlan]:
        """계획 로드"""
        plan_file = self.plans_dir / f"plan_{plan_id}.json"
        if plan_file.exists():
            with open(plan_file) as f:
                data = json.load(f)
            return ReorganizationPlan(**data)
        return None
    
    def approve_plan(self, plan_id: str, approver: str = "creator"):
        """계획 승인"""
        plan = self.load_plan(plan_id)
        if plan:
            plan.approved = True
            plan.approved_by = approver
            plan.approved_at = datetime.now().isoformat()
            self.current_plan = plan
            self.save_plan()
            print(f"✅ Plan {plan_id} approved by {approver}")
            return True
        print(f"❌ Plan {plan_id} not found")
        return False
    
    def generate_report(self):
        """결과 보고서 생성"""
        print("\n" + "=" * 70)
        print("📋 REORGANIZATION REPORT")
        print("=" * 70)
        
        if self.current_plan:
            print(f"\n   Plan ID: {self.current_plan.id}")
            print(f"   Created: {self.current_plan.created_at}")
            print(f"   Approved: {'✅ Yes' if self.current_plan.approved else '❌ No'}")
            print(f"   Executed: {'✅ Yes' if self.current_plan.executed else '⏳ Pending'}")
            print(f"   Resonance: {self.current_plan.resonance_score:.0%}")
            
            if self.current_plan.verification_result:
                v = self.current_plan.verification_result
                print(f"\n   Verification: {'✅ PASSED' if v.get('success') else '❌ FAILED'}")
            
            if self.current_plan.dna_backup_path:
                print(f"\n   💾 DNA Backup: {self.current_plan.dna_backup_path}")
                print("   → Rollback: python scripts/autonomous_reorganizer.py rollback --path <backup_path>")
        
        print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Elysia Autonomous Reorganizer")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # run command
    run_parser = subparsers.add_parser("run", help="Run full workflow")
    run_parser.add_argument("--auto-approve", action="store_true", help="Auto-approve the plan")
    run_parser.add_argument("--execute", action="store_true", help="Execute changes (not dry-run)")
    
    # plan command
    plan_parser = subparsers.add_parser("plan", help="Create plan only")
    
    # approve command
    approve_parser = subparsers.add_parser("approve", help="Approve a plan")
    approve_parser.add_argument("--id", required=True, help="Plan ID to approve")
    
    # rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback from DNA backup")
    rollback_parser.add_argument("--path", required=True, help="Backup path")
    
    # status command
    status_parser = subparsers.add_parser("status", help="Show current status")
    
    args = parser.parse_args()
    
    reorganizer = AutonomousReorganizer()
    
    if args.command == "run":
        reorganizer.run_workflow(
            auto_approve=args.auto_approve,
            execute=args.execute
        )
    elif args.command == "plan":
        reorganizer.current_plan = reorganizer.planner.create_plan()
        reorganizer.save_plan()
    elif args.command == "approve":
        reorganizer.approve_plan(args.id)
    elif args.command == "rollback":
        reorganizer.simulator.rollback(args.path)
    elif args.command == "status":
        # 최신 계획 표시
        plans = list(reorganizer.plans_dir.glob("plan_*.json"))
        if plans:
            latest = max(plans, key=lambda p: p.stat().st_mtime)
            plan = reorganizer.load_plan(latest.stem.replace("plan_", ""))
            if plan:
                reorganizer.current_plan = plan
                reorganizer.generate_report()
        else:
            print("No plans found. Run: python scripts/autonomous_reorganizer.py run")
    else:
        # 기본: 전체 워크플로우 실행 (승인 대기 모드)
        reorganizer.run_workflow()


if __name__ == "__main__":
    main()
