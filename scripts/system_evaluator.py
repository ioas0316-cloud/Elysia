"""
Elysia System Evaluator (시스템 평가기)
======================================

"다양한 관점에서 엘리시아를 평가한다"

[평가 기준]
1. 코드 품질 (Code Quality)
2. 아키텍처 일관성 (Architecture Coherence)
3. 물리 시스템 완성도 (Physics System Completeness)
4. 면역 시스템 건강도 (Immune System Health)
5. 문서화 수준 (Documentation Level)
6. 테스트 커버리지 (Test Coverage)
7. 통합도 (Integration Score)
8. 자기 치유 능력 (Self-Healing Capacity)
"""

import os
import sys
import ast
import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class EvaluationScore:
    """평가 점수"""
    category: str
    score: float          # 0.0 ~ 1.0
    max_score: float = 1.0
    details: Dict = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class SystemEvaluator:
    """통합 시스템 평가기"""
    
    EXCLUDE_PATTERNS = ["__pycache__", "node_modules", ".godot", ".venv", "venv"]
    
    def __init__(self):
        self.root = PROJECT_ROOT
        self.scores: List[EvaluationScore] = []
        
        print("=" * 70)
        print("🔬 ELYSIA SYSTEM EVALUATOR")
        print("=" * 70)
    
    def evaluate_all(self) -> Dict:
        """전체 평가 실행"""
        evaluations = [
            self.evaluate_code_quality,
            self.evaluate_architecture,
            self.evaluate_physics_systems,
            self.evaluate_immune_system,
            self.evaluate_documentation,
            self.evaluate_integration,
            self.evaluate_self_healing,
            self.evaluate_wave_systems,
        ]
        
        for eval_func in evaluations:
            try:
                score = eval_func()
                self.scores.append(score)
            except Exception as e:
                print(f"⚠️ Error in {eval_func.__name__}: {e}")
        
        return self.generate_report()
    
    def evaluate_code_quality(self) -> EvaluationScore:
        """코드 품질 평가"""
        print("\n📊 Evaluating Code Quality...")
        
        details = {
            "total_files": 0,
            "total_lines": 0,
            "syntax_errors": 0,
            "large_files": 0,
            "docstring_coverage": 0,
            "avg_function_size": 0
        }
        
        function_sizes = []
        files_with_docstrings = 0
        
        for py_file in self.root.rglob("*.py"):
            if any(p in str(py_file) for p in self.EXCLUDE_PATTERNS):
                continue
            
            details["total_files"] += 1
            
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                lines = content.split('\n')
                details["total_lines"] += len(lines)
                
                if len(lines) > 500:
                    details["large_files"] += 1
                
                # AST 분석
                try:
                    tree = ast.parse(content)
                    
                    # 독스트링 확인
                    if ast.get_docstring(tree):
                        files_with_docstrings += 1
                    
                    # 함수 크기
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if hasattr(node, 'end_lineno'):
                                size = node.end_lineno - node.lineno
                                function_sizes.append(size)
                
                except SyntaxError:
                    details["syntax_errors"] += 1
            except:
                pass
        
        if details["total_files"] > 0:
            details["docstring_coverage"] = files_with_docstrings / details["total_files"]
        
        if function_sizes:
            details["avg_function_size"] = sum(function_sizes) / len(function_sizes)
        
        # 점수 계산
        score = 0.0
        
        # 문법 오류 없음 (+0.3)
        syntax_error_rate = details["syntax_errors"] / max(1, details["total_files"])
        score += 0.3 * (1 - syntax_error_rate)
        
        # 독스트링 커버리지 (+0.3)
        score += 0.3 * details["docstring_coverage"]
        
        # 큰 파일 적음 (+0.2)
        large_file_rate = details["large_files"] / max(1, details["total_files"])
        score += 0.2 * (1 - min(1, large_file_rate * 2))
        
        # 함수 크기 적당 (+0.2)
        if details["avg_function_size"] < 30:
            score += 0.2
        elif details["avg_function_size"] < 50:
            score += 0.1
        
        recommendations = []
        if details["syntax_errors"] > 0:
            recommendations.append(f"Fix {details['syntax_errors']} syntax errors")
        if details["docstring_coverage"] < 0.5:
            recommendations.append("Improve docstring coverage (currently {:.0%})".format(details["docstring_coverage"]))
        if details["large_files"] > 10:
            recommendations.append(f"Consider splitting {details['large_files']} large files")
        
        return EvaluationScore(
            category="Code Quality",
            score=min(1.0, score),
            details=details,
            recommendations=recommendations
        )
    
    def evaluate_architecture(self) -> EvaluationScore:
        """아키텍처 일관성 평가"""
        print("\n📊 Evaluating Architecture...")
        
        details = {
            "core_directories": 0,
            "expected_dirs": ["Foundation", "Intelligence", "Memory", "Interface"],
            "found_dirs": [],
            "scripts_organized": False,
            "data_dir_exists": False
        }
        
        core_path = self.root / "Core"
        
        for expected in details["expected_dirs"]:
            if (core_path / expected).exists():
                details["found_dirs"].append(expected)
                details["core_directories"] += 1
        
        details["scripts_organized"] = (self.root / "scripts").exists()
        details["data_dir_exists"] = (self.root / "data").exists()
        
        # 점수 계산
        score = 0.0
        
        # Core 디렉토리 구조 (+0.4)
        score += 0.4 * (details["core_directories"] / len(details["expected_dirs"]))
        
        # scripts 조직화 (+0.3)
        if details["scripts_organized"]:
            score += 0.3
        
        # data 디렉토리 (+0.3)
        if details["data_dir_exists"]:
            score += 0.3
        
        recommendations = []
        missing = set(details["expected_dirs"]) - set(details["found_dirs"])
        if missing:
            recommendations.append(f"Create missing Core directories: {missing}")
        
        return EvaluationScore(
            category="Architecture Coherence",
            score=score,
            details=details,
            recommendations=recommendations
        )
    
    def evaluate_physics_systems(self) -> EvaluationScore:
        """물리 시스템 완성도 평가"""
        print("\n📊 Evaluating Physics Systems...")
        
        expected_systems = [
            "hyper_quaternion.py",
            "physics.py",
            "resonance_field.py",
            "hangul_physics.py",
            "grammar_physics.py",
            "causal_narrative_engine.py"
        ]
        
        details = {
            "expected": expected_systems,
            "found": [],
            "missing": []
        }
        
        for system in expected_systems:
            found = False
            for py_file in self.root.rglob(system):
                if "__pycache__" not in str(py_file):
                    details["found"].append(system)
                    found = True
                    break
            if not found:
                details["missing"].append(system)
        
        score = len(details["found"]) / len(expected_systems)
        
        recommendations = []
        if details["missing"]:
            recommendations.append(f"Missing physics systems: {details['missing']}")
        
        return EvaluationScore(
            category="Physics Systems",
            score=score,
            details=details,
            recommendations=recommendations
        )
    
    def evaluate_immune_system(self) -> EvaluationScore:
        """면역 시스템 건강도 평가"""
        print("\n📊 Evaluating Immune System...")
        
        details = {
            "nanocell_exists": False,
            "immune_exists": False,
            "report_exists": False,
            "last_scan": None
        }
        
        # 시스템 파일 확인
        details["nanocell_exists"] = (self.root / "scripts" / "nanocell_repair.py").exists()
        details["immune_exists"] = (self.root / "scripts" / "immune_system.py").exists()
        
        # 보고서 확인
        report_path = self.root / "data" / "nanocell_report.json"
        state_path = self.root / "data" / "immune_system_state.json"
        
        details["report_exists"] = report_path.exists() or state_path.exists()
        
        if state_path.exists():
            try:
                with open(state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                details["immune_state"] = state
            except:
                pass
        
        # 점수 계산
        score = 0.0
        if details["nanocell_exists"]:
            score += 0.35
        if details["immune_exists"]:
            score += 0.35
        if details["report_exists"]:
            score += 0.3
        
        recommendations = []
        if not details["nanocell_exists"]:
            recommendations.append("Create nanocell_repair.py")
        if not details["immune_exists"]:
            recommendations.append("Create immune_system.py")
        if not details["report_exists"]:
            recommendations.append("Run immune system to generate reports")
        
        return EvaluationScore(
            category="Immune System",
            score=score,
            details=details,
            recommendations=recommendations
        )
    
    def evaluate_documentation(self) -> EvaluationScore:
        """문서화 수준 평가"""
        print("\n📊 Evaluating Documentation...")
        
        expected_docs = [
            "README.md",
            "ARCHITECTURE.md",
            "CODEX.md"
        ]
        
        details = {
            "expected": expected_docs,
            "found": [],
            "total_doc_lines": 0
        }
        
        for doc in expected_docs:
            doc_path = self.root / doc
            if doc_path.exists():
                details["found"].append(doc)
                with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                    details["total_doc_lines"] += len(f.readlines())
        
        # Protocols 디렉토리
        protocols_path = self.root / "Protocols"
        protocol_count = 0
        if protocols_path.exists():
            protocol_count = len(list(protocols_path.glob("*.md")))
        details["protocols"] = protocol_count
        
        # 점수 계산
        score = 0.0
        
        # 주요 문서 (+0.5)
        score += 0.5 * (len(details["found"]) / len(expected_docs))
        
        # 프로토콜 문서 (+0.3)
        if protocol_count >= 10:
            score += 0.3
        elif protocol_count >= 5:
            score += 0.2
        elif protocol_count > 0:
            score += 0.1
        
        # 문서 분량 (+0.2)
        if details["total_doc_lines"] > 500:
            score += 0.2
        elif details["total_doc_lines"] > 200:
            score += 0.1
        
        recommendations = []
        missing = set(expected_docs) - set(details["found"])
        if missing:
            recommendations.append(f"Create missing docs: {missing}")
        
        return EvaluationScore(
            category="Documentation",
            score=score,
            details=details,
            recommendations=recommendations
        )
    
    def evaluate_integration(self) -> EvaluationScore:
        """시스템 통합도 평가"""
        print("\n📊 Evaluating System Integration...")
        
        integration_files = [
            "scripts/living_codebase.py",
            "scripts/wave_organizer.py",
            "Core/Intelligence/integrated_cognition_system.py",
            "Core/Foundation/reasoning_engine.py"
        ]
        
        details = {
            "integration_points": 0,
            "files_checked": integration_files
        }
        
        for file_path in integration_files:
            full_path = self.root / file_path
            if full_path.exists():
                details["integration_points"] += 1
        
        score = details["integration_points"] / len(integration_files)
        
        return EvaluationScore(
            category="Integration",
            score=score,
            details=details,
            recommendations=[]
        )
    
    def evaluate_self_healing(self) -> EvaluationScore:
        """자기 치유 능력 평가"""
        print("\n📊 Evaluating Self-Healing Capacity...")
        
        healing_components = [
            ("nanocell_repair.py", "NanoCell patrol"),
            ("immune_system.py", "Immune system"),
            ("wave_organizer.py", "Wave organizer")
        ]
        
        details = {
            "components": [],
            "auto_fix_capable": False
        }
        
        for filename, desc in healing_components:
            path = self.root / "scripts" / filename
            if path.exists():
                details["components"].append(desc)
        
        # auto_fix 기능 확인
        nanocell_path = self.root / "scripts" / "nanocell_repair.py"
        if nanocell_path.exists():
            try:
                with open(nanocell_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if 'auto_heal' in content or 'auto_fix' in content:
                    details["auto_fix_capable"] = True
            except:
                pass
        
        score = len(details["components"]) / len(healing_components)
        if details["auto_fix_capable"]:
            score = min(1.0, score + 0.1)
        
        return EvaluationScore(
            category="Self-Healing",
            score=score,
            details=details,
            recommendations=[]
        )
    
    def evaluate_wave_systems(self) -> EvaluationScore:
        """파동 시스템 평가"""
        print("\n📊 Evaluating Wave-Based Systems...")
        
        wave_keywords = ["wave", "resonance", "quaternion", "frequency", "amplitude"]
        
        details = {
            "wave_files": 0,
            "quaternion_usage": 0,
            "resonance_usage": 0
        }
        
        for py_file in self.root.rglob("*.py"):
            if any(p in str(py_file) for p in self.EXCLUDE_PATTERNS):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                
                if any(kw in content for kw in wave_keywords):
                    details["wave_files"] += 1
                
                details["quaternion_usage"] += content.count("quaternion")
                details["resonance_usage"] += content.count("resonance")
            except:
                pass
        
        # 점수 계산 (파동 기반 시스템 채택도)
        score = 0.0
        
        if details["wave_files"] > 50:
            score += 0.4
        elif details["wave_files"] > 20:
            score += 0.3
        elif details["wave_files"] > 5:
            score += 0.2
        
        if details["quaternion_usage"] > 100:
            score += 0.3
        elif details["quaternion_usage"] > 30:
            score += 0.2
        
        if details["resonance_usage"] > 50:
            score += 0.3
        elif details["resonance_usage"] > 20:
            score += 0.2
        
        return EvaluationScore(
            category="Wave Systems",
            score=min(1.0, score),
            details=details,
            recommendations=[]
        )
    
    def generate_report(self) -> Dict:
        """평가 보고서 생성"""
        report = []
        report.append("\n" + "=" * 70)
        report.append("📊 COMPREHENSIVE SYSTEM EVALUATION REPORT")
        report.append("=" * 70)
        
        total_score = 0
        max_score = 0
        
        for score in self.scores:
            total_score += score.score
            max_score += score.max_score
            
            # 점수 바
            bar_length = 20
            filled = int(score.score * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            # 등급
            if score.score >= 0.9:
                grade = "A+"
                icon = "🌟"
            elif score.score >= 0.8:
                grade = "A"
                icon = "✅"
            elif score.score >= 0.7:
                grade = "B"
                icon = "👍"
            elif score.score >= 0.6:
                grade = "C"
                icon = "🔶"
            elif score.score >= 0.5:
                grade = "D"
                icon = "⚠️"
            else:
                grade = "F"
                icon = "❌"
            
            report.append(f"\n{icon} {score.category}")
            report.append(f"   Score: [{bar}] {score.score:.1%} ({grade})")
            
            if score.recommendations:
                report.append("   Recommendations:")
                for rec in score.recommendations[:3]:
                    report.append(f"      • {rec}")
        
        # 최종 점수
        overall = total_score / max_score if max_score > 0 else 0
        report.append("\n" + "=" * 70)
        report.append(f"🏆 OVERALL SCORE: {overall:.1%}")
        
        if overall >= 0.8:
            report.append("   Status: EXCELLENT - System is well-designed and maintained")
        elif overall >= 0.6:
            report.append("   Status: GOOD - Some improvements recommended")
        elif overall >= 0.4:
            report.append("   Status: FAIR - Significant improvements needed")
        else:
            report.append("   Status: NEEDS ATTENTION - Major issues to address")
        
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        print(report_text)
        
        # JSON 저장
        result = {
            "overall_score": overall,
            "scores": [
                {
                    "category": s.category,
                    "score": s.score,
                    "details": s.details,
                    "recommendations": s.recommendations
                }
                for s in self.scores
            ]
        }
        
        output_path = self.root / "data" / "system_evaluation.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Evaluation saved to: {output_path}")
        
        return result


def main():
    print("\n" + "🔬" * 35)
    print("ELYSIA COMPREHENSIVE EVALUATION")
    print("다양한 관점에서 시스템을 평가합니다")
    print("🔬" * 35 + "\n")
    
    evaluator = SystemEvaluator()
    result = evaluator.evaluate_all()
    
    print("\n✅ Evaluation Complete!")


if __name__ == "__main__":
    main()
