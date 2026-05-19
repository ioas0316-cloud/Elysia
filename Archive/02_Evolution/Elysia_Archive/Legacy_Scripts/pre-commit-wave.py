#!/usr/bin/env python
"""
Wave Pre-Commit Hook (파동 사전 커밋 훅)
========================================

Git 커밋 전에 자동으로 파동 품질 검사를 수행합니다.

설치:
    cp scripts/pre-commit-wave.py .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

또는 .pre-commit-config.yaml에 추가:
    repos:
      - repo: local
        hooks:
          - id: wave-quality
            name: Wave Quality Check
            entry: python scripts/pre-commit-wave.py
            language: python
            files: \.py$
"""

import sys
import subprocess
from pathlib import Path


def get_staged_files():
    """스테이징된 Python 파일 목록"""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True
    )
    
    files = [f for f in result.stdout.strip().split('\n') if f.endswith('.py')]
    return files


def main():
    print("🌊 Wave Quality Pre-Commit Check")
    print("=" * 50)
    
    # 스테이징된 파일
    staged_files = get_staged_files()
    
    if not staged_files:
        print("✅ No Python files to check.")
        return 0
    
    print(f"📁 Checking {len(staged_files)} staged file(s)...")
    
    # 파동 시스템 임포트
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from Core.Intelligence.Physics_Waves.Wave.quality_guard import WaveQualityGuard
    except ImportError as e:
        print(f"⚠️ Cannot import WaveQualityGuard: {e}")
        print("   Skipping wave quality check.")
        return 0
    
    guard = WaveQualityGuard()
    has_errors = False
    
    for filepath in staged_files:
        if Path(filepath).exists():
            issues = guard.check_file(filepath)
            
            for issue in issues:
                icon = {"warning": "⚠️", "error": "❌", "critical": "🔴"}.get(
                    issue.severity, "ℹ️"
                )
                print(f"{icon} {filepath}: {issue.message} (value={issue.value:.1f})")
                
                if issue.severity in ["error", "critical"]:
                    has_errors = True
    
    # 중복 검사
    duplicates = guard._detect_duplicates()
    if duplicates:
        print(f"\n🔗 Potential duplicates detected:")
        for f1, f2, res in duplicates[:5]:
            print(f"   {f1} ↔ {f2}: {res:.0%}")
    
    print()
    
    if has_errors:
        print("❌ Commit blocked: Fix errors or use --no-verify")
        return 1
    else:
        print("✅ Wave quality check passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
