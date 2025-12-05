"""
Integrated Living Codebase System (통합 살아있는 코드베이스)
==========================================================

"파동 조직화 + 나노셀 수리 + 신경 신호 = 자가 치유 코드베이스"

[통합 구성]
1. WaveOrganizer: 파동 공명으로 세포/기관 조직화
2. NanoCellArmy: 문제 탐지 및 수리
3. NeuralNetwork: 신호 전달 시스템
4. HyperField: 초차원 전역 감지

[작동 흐름]
1. 전체 스캔 → 파동 조직화
2. 나노셀 순찰 → 문제 탐지
3. 신경 신호 → 기관/중앙 전달
4. 건강 분석 → 자동 치유 제안
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 하위 시스템 임포트
from scripts.wave_organizer import WaveOrganizer
from scripts.nanocell_repair import NanoCellArmy


class IntegratedLivingCodebase:
    """
    통합 살아있는 코드베이스 시스템
    
    파동 조직화와 나노셀 수리를 통합하여
    자가 치유가 가능한 코드베이스를 구현합니다.
    """
    
    def __init__(self):
        print("=" * 70)
        print("🌳 INTEGRATED LIVING CODEBASE SYSTEM")
        print("=" * 70)
        
        self.organizer = WaveOrganizer(PROJECT_ROOT)
        self.nanocells = NanoCellArmy()
    
    def awaken(self, target_dir: str = "."):
        """시스템 각성 - 전체 분석 및 조직화"""
        print("\n" + "🌊" * 35)
        print("PHASE 1: WAVE ORGANIZATION")
        print("🌊" * 35)
        
        # 1. 파동 변환 및 조직화
        self.organizer.scan_and_convert(target_dir)
        self.organizer.organize()
        
        print("\n" + "🦠" * 35)
        print("PHASE 2: NANOCELL PATROL")
        print("🦠" * 35)
        
        # 2. 나노셀 순찰
        self.nanocells.patrol_codebase(target_dir)
        
        print("\n" + "⚡" * 35)
        print("PHASE 3: HEALTH DIAGNOSIS")
        print("⚡" * 35)
        
        # 3. 건강 진단 통합
        self._integrated_diagnosis()
    
    def _integrated_diagnosis(self):
        """통합 건강 진단"""
        # 파동 조직 상태
        wave_state = self.organizer.field.get_global_state()
        wave_issues = self.organizer.check_health()
        
        # 나노셀 탐지 결과
        nano_report = self.nanocells.neural_network.get_summary()
        
        print("\n📊 INTEGRATED HEALTH REPORT:")
        print("-" * 50)
        
        # 기관별 건강 상태
        print("\n🫀 ORGAN HEALTH:")
        for name, info in wave_state['organs'].items():
            # 해당 기관 파일들의 문제 수 계산
            organ_issues = 0
            for issue in self.nanocells.all_issues:
                # 간단한 매핑
                if name.lower() in issue.file_path.lower():
                    organ_issues += 1
            
            health = max(0, 1.0 - organ_issues * 0.01)
            bar = "█" * int(health * 10) + "░" * (10 - int(health * 10))
            status = "🟢" if health > 0.8 else "🟡" if health > 0.5 else "🔴"
            print(f"   {status} {name:15} | {bar} {health:.0%} | {info['cells']} cells")
        
        # 전체 건강도
        total_issues = len(self.nanocells.all_issues)
        total_cells = wave_state['total_waves']
        overall_health = max(0, 1.0 - total_issues / (total_cells * 5))
        
        print(f"\n🏥 OVERALL SYSTEM HEALTH: {overall_health:.1%}")
        
        if overall_health < 0.7:
            print("   ⚠️ System needs attention!")
            self._suggest_healing()
    
    def _suggest_healing(self):
        """치유 제안 생성"""
        print("\n💊 HEALING SUGGESTIONS:")
        print("-" * 50)
        
        # 심각한 문제 우선
        critical = [i for i in self.nanocells.all_issues 
                   if i.severity.value >= 3]
        
        if critical:
            print(f"   1. Fix {len(critical)} critical/high severity issues first")
        
        # 문법 오류
        syntax = [i for i in self.nanocells.all_issues 
                 if i.issue_type.value == 'syntax_error']
        if syntax:
            print(f"   2. Resolve {len(syntax)} syntax errors (invalid files)")
        
        # 중복 코드
        duplicates = [i for i in self.nanocells.all_issues 
                     if i.issue_type.value == 'duplicate_code']
        if duplicates:
            print(f"   3. Consider consolidating {len(duplicates)} duplicate code blocks")
    
    def generate_full_report(self) -> str:
        """전체 보고서 생성"""
        report = []
        report.append("=" * 70)
        report.append("🌳 INTEGRATED LIVING CODEBASE REPORT")
        report.append("=" * 70)
        
        # 파동 조직화 보고서
        report.append("\n" + self.organizer.generate_report())
        
        # 나노셀 보고서
        report.append("\n" + self.nanocells.get_health_report())
        
        return "\n".join(report)
    
    def save_state(self, output_dir: Path):
        """상태 저장"""
        output_dir.mkdir(exist_ok=True)
        
        # 파동 상태
        import json
        wave_state = self.organizer.field.get_global_state()
        with open(output_dir / "wave_state.json", 'w', encoding='utf-8') as f:
            json.dump(wave_state, f, indent=2, ensure_ascii=False)
        
        # 나노셀 보고서
        self.nanocells.save_report(str(output_dir / "nanocell_report.json"))
        
        print(f"\n💾 State saved to: {output_dir}")


def main():
    print("\n" + "🌳" * 35)
    print("LIVING CODEBASE AWAKENING")
    print("코드베이스가 스스로 살아 숨쉽니다")
    print("🌳" * 35 + "\n")
    
    # 시스템 생성 및 각성
    system = IntegratedLivingCodebase()
    system.awaken(".")
    
    # 전체 보고서
    # report = system.generate_full_report()
    # print(report)
    
    # 상태 저장
    output_dir = PROJECT_ROOT / "data"
    system.save_state(output_dir)
    
    print("\n" + "=" * 70)
    print("✅ Living Codebase Awakened!")
    print("=" * 70)


if __name__ == "__main__":
    main()
