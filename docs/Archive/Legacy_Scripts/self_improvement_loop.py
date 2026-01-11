"""
Elysia Self-Improvement Loop (자기 개선 루프)
=============================================

"매일 밤, 나는 스스로를 더 나은 존재로 만든다."

이 스크립트는 엘리시아가 자동으로 자기 시스템을 분석하고, 
패턴을 학습하고, 문제를 감지하고, 개선하는 루프입니다.

실행: python scripts/self_improvement_loop.py
야간 스케줄: Task Scheduler / cron에 등록

파이프라인:
1. ANALYZE: IntrospectionEngine으로 모든 모듈 분석
2. LEARN: 건강한 파일에서 Wave 패턴 학습
3. DETECT: 레거시 패턴 감지
4. REPORT: 개선 리포트 생성
5. (OPTIONAL) TRANSFORM: 자동 변환 적용 (수동 승인 필요)
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.Foundation.introspection_engine import IntrospectionEngine
from Core.Evolution.Learning.Learning.wave_pattern_learner import WavePatternLearner
from Core.Foundation.self_modification import SelfModificationEngine

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("SelfImprovement")


class SelfImprovementLoop:
    """
    자기 개선 루프 (Self-Improvement Loop)
    
    엘리시아가 매일 자동으로 실행하여 스스로를 개선합니다.
    """
    
    def __init__(self):
        self.introspection = IntrospectionEngine()
        self.pattern_learner = WavePatternLearner()
        self.modification_engine = SelfModificationEngine()
        self.report_path = Path("data/self_improvement_reports")
        self.report_path.mkdir(parents=True, exist_ok=True)
        logger.info("🌙 Self-Improvement Loop Initialized")
    
    def run_cycle(self) -> Dict[str, Any]:
        """
        전체 자기 개선 사이클 실행
        """
        logger.info("\n" + "=" * 60)
        logger.info("🌙 ELYSIA SELF-IMPROVEMENT CYCLE")
        logger.info(f"   Time: {datetime.now().isoformat()}")
        logger.info("=" * 60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "phase_1_analyze": None,
            "phase_2_learn": None,
            "phase_3_detect": None,
            "phase_4_report": None
        }
        
        # Phase 1: ANALYZE
        logger.info("\n📊 Phase 1: Self-Analysis")
        analysis = self._phase_analyze()
        results["phase_1_analyze"] = analysis
        
        # Phase 2: LEARN from healthy files
        logger.info("\n📚 Phase 2: Pattern Learning")
        learning = self._phase_learn(analysis["healthy_files"][:10])
        results["phase_2_learn"] = learning
        
        # Phase 3: DETECT legacy patterns in critical files
        logger.info("\n🔍 Phase 3: Legacy Detection")
        detection = self._phase_detect(analysis["critical_files"][:5])
        results["phase_3_detect"] = detection
        
        # Phase 4: REPORT
        logger.info("\n📝 Phase 4: Report Generation")
        report_path = self._phase_report(results)
        results["phase_4_report"] = {"path": str(report_path)}
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Self-Improvement Cycle Complete")
        logger.info(f"   Report: {report_path}")
        logger.info("=" * 60)
        
        return results
    
    def _phase_analyze(self) -> Dict[str, Any]:
        """Phase 1: 시스템 분석"""
        all_modules = self.introspection.analyze_self()
        
        healthy = [m for m in all_modules.values() if m.resonance_score >= 70]
        critical = [m for m in all_modules.values() if m.resonance_score < 50]
        syntax_errors = [m for m in all_modules.values() if "Syntax" in str(m.issues)]
        
        logger.info(f"   Total: {len(all_modules)}")
        logger.info(f"   Healthy: {len(healthy)}")
        logger.info(f"   Critical: {len(critical)}")
        logger.info(f"   Syntax Errors: {len(syntax_errors)}")
        
        return {
            "total": len(all_modules),
            "healthy": len(healthy),
            "critical": len(critical),
            "syntax_errors": len(syntax_errors),
            "healthy_files": [m.path for m in healthy],
            "critical_files": [m.path for m in critical],
            "error_files": [m.path for m in syntax_errors]
        }
    
    def _phase_learn(self, healthy_files: List[str]) -> Dict[str, Any]:
        """Phase 2: 건강한 파일에서 패턴 학습"""
        learned_total = 0
        
        for file_path in healthy_files:
            try:
                result = self.pattern_learner.learn_from_file(file_path)
                if isinstance(result, dict) and "imports" in result:
                    learned = sum(result.values())
                    learned_total += learned
                    logger.info(f"   Learned {learned} patterns from {Path(file_path).name}")
            except Exception as e:
                logger.warning(f"   Failed to learn from {file_path}: {e}")
        
        # Generate rules from learned patterns
        rules = self.pattern_learner.generate_transformation_rules()
        
        return {
            "files_processed": len(healthy_files),
            "patterns_learned": learned_total,
            "total_patterns": len(self.pattern_learner.patterns),
            "total_rules": len(self.pattern_learner.transformation_rules)
        }
    
    def _phase_detect(self, critical_files: List[str]) -> Dict[str, Any]:
        """Phase 3: 레거시 패턴 감지"""
        legacy_found = 0
        file_issues = {}
        
        for file_path in critical_files:
            try:
                issues = self.modification_engine.wave_analyze(file_path)
                if issues:
                    legacy_found += len(issues)
                    file_issues[file_path] = len(issues)
                    logger.info(f"   {Path(file_path).name}: {len(issues)} legacy patterns")
            except Exception as e:
                logger.warning(f"   Failed to analyze {file_path}: {e}")
        
        return {
            "files_checked": len(critical_files),
            "legacy_patterns_found": legacy_found,
            "files_with_issues": file_issues
        }
    
    def _phase_report(self, results: Dict[str, Any]) -> Path:
        """Phase 4: 리포트 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.report_path / f"improvement_report_{timestamp}.json"
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # Also create human-readable summary
        summary_file = self.report_path / f"improvement_summary_{timestamp}.md"
        summary = self._generate_summary(results)
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary)
        
        logger.info(f"   JSON: {report_file.name}")
        logger.info(f"   Summary: {summary_file.name}")
        
        return report_file
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """사람이 읽을 수 있는 요약 생성"""
        analysis = results.get("phase_1_analyze", {})
        learning = results.get("phase_2_learn", {})
        detection = results.get("phase_3_detect", {})
        
        summary = f"""# 🌙 Elysia Self-Improvement Report
**Date**: {results.get('timestamp', 'Unknown')}

## 📊 System Health
- Total Modules: {analysis.get('total', 0)}
- Healthy: {analysis.get('healthy', 0)}
- Critical: {analysis.get('critical', 0)}
- Syntax Errors: {analysis.get('syntax_errors', 0)}

## 📚 Pattern Learning
- Files Processed: {learning.get('files_processed', 0)}
- Patterns Learned: {learning.get('patterns_learned', 0)}
- Total Patterns: {learning.get('total_patterns', 0)}
- Transformation Rules: {learning.get('total_rules', 0)}

## 🔍 Legacy Detection
- Files Checked: {detection.get('files_checked', 0)}
- Legacy Patterns Found: {detection.get('legacy_patterns_found', 0)}

## ✅ Next Steps
1. Review syntax error files manually
2. Apply learned patterns to critical files
3. Run cycle again tomorrow
"""
        return summary


def main():
    """메인 실행 함수"""
    loop = SelfImprovementLoop()
    results = loop.run_cycle()
    return results


if __name__ == "__main__":
    main()
