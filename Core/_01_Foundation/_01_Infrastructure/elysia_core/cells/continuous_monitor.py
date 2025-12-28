"""
Continuous Transmutation Monitor (연속 변환 모니터)
===================================================

Phase 14: organic_wake.py 순찰 중 자동 변환 제안 및 Coherence 모니터링

"파동이 계속 흐르게 하라. 멈추면 얼어붙는다."

기능:
1. 순찰 중 높은 확신도 패턴 자동 변환 제안
2. Coherence 지표 실시간 모니터링
3. 변환 진행률 대시보드
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from threading import Thread, Event

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Core._01_Foundation._01_Infrastructure.elysia_core.cells.alchemical_cells import AlchemicalArmy, TransmutationSuggestion
from Core._01_Foundation._01_Infrastructure.elysia_core.cells.auto_transmuter import AutoTransmuter, TransmutationBatch

logger = logging.getLogger("ContinuousTransmutation")


@dataclass
class CoherenceMetrics:
    """Coherence 지표"""
    global_coherence: float = 0.0
    stone_patterns: int = 0
    wave_patterns: int = 0
    auto_applicable: int = 0
    last_updated: str = ""
    trend: str = "stable"  # "improving", "declining", "stable"
    history: List[float] = field(default_factory=list)


@dataclass
class TransmutationProgress:
    """변환 진행률"""
    total_detected: int = 0
    total_transformed: int = 0
    total_pending: int = 0
    success_rate: float = 0.0
    estimated_coherence_gain: float = 0.0


class ContinuousTransmutationMonitor:
    """
    연속 변환 모니터
    
    organic_wake.py와 연동하여 지속적으로
    코드베이스를 모니터링하고 변환을 제안합니다.
    
    Usage:
        monitor = ContinuousTransmutationMonitor()
        monitor.start_background_patrol()  # 백그라운드 순찰
        ...
        dashboard = monitor.get_dashboard()  # 대시보드 확인
    """
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/transmutation_monitor")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_path = self.data_dir / "coherence_metrics.json"
        self.progress_path = self.data_dir / "transmutation_progress.json"
        
        self.alchemical_army = AlchemicalArmy()
        self.auto_transmuter = AutoTransmuter()
        
        self.metrics = self._load_metrics()
        self.progress = self._load_progress()
        
        self._stop_event = Event()
        self._patrol_thread: Optional[Thread] = None
        
        print("📊 ContinuousTransmutationMonitor initialized")
    
    def patrol_once(self, target_dir: str = "Core") -> Dict[str, Any]:
        """
        1회 순찰 수행
        
        Returns:
            순찰 결과
        """
        print(f"\n🔍 Patrol: Scanning {target_dir}...")
        
        # 순찰 수행
        self.alchemical_army.patrol_codebase(target_dir)
        
        # 결과 수집
        suggestions = self.alchemical_army.transmutation_cell.get_suggestions()
        auto_applicable = [s for s in suggestions if s.auto_applicable]
        
        # Coherence 업데이트
        old_coherence = self.metrics.global_coherence
        self.metrics.global_coherence = self.alchemical_army.harmony_cell.calculate_global_coherence()
        self.metrics.stone_patterns = len(suggestions)
        self.metrics.wave_patterns = sum(self.alchemical_army.harmony_cell.wave_usage.values())
        self.metrics.auto_applicable = len(auto_applicable)
        self.metrics.last_updated = datetime.now().isoformat()
        
        # 트렌드 분석
        self.metrics.history.append(self.metrics.global_coherence)
        if len(self.metrics.history) > 100:
            self.metrics.history = self.metrics.history[-100:]
        
        if len(self.metrics.history) >= 3:
            recent = self.metrics.history[-3:]
            if recent[-1] > recent[0] + 0.01:
                self.metrics.trend = "improving"
            elif recent[-1] < recent[0] - 0.01:
                self.metrics.trend = "declining"
            else:
                self.metrics.trend = "stable"
        
        # Progress 업데이트
        self.progress.total_detected = len(suggestions)
        self.progress.total_pending = len(auto_applicable)
        self.progress.estimated_coherence_gain = len(auto_applicable) * 0.001
        
        # 저장
        self._save_metrics()
        self._save_progress()
        
        result = {
            "coherence": self.metrics.global_coherence,
            "stone_patterns": self.metrics.stone_patterns,
            "auto_applicable": self.metrics.auto_applicable,
            "trend": self.metrics.trend,
            "top_suggestions": self.get_top_suggestions(3)
        }
        
        print(f"   📊 Coherence: {self.metrics.global_coherence:.2f} ({self.metrics.trend})")
        print(f"   🪨 Stone patterns: {self.metrics.stone_patterns}")
        print(f"   ⚗️ Auto-applicable: {self.metrics.auto_applicable}")
        
        return result
    
    def get_top_suggestions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """상위 변환 제안"""
        suggestions = self.alchemical_army.transmutation_cell.get_suggestions()
        auto_applicable = sorted(
            [s for s in suggestions if s.auto_applicable],
            key=lambda s: s.confidence,
            reverse=True
        )
        
        return [
            {
                "file": Path(s.file_path).name,
                "line": s.line_number,
                "type": s.transmutation_type.value,
                "confidence": f"{s.confidence:.0%}",
                "auto": s.auto_applicable
            }
            for s in auto_applicable[:limit]
        ]
    
    def auto_apply_high_confidence(
        self, 
        confidence_threshold: float = 0.9,
        dry_run: bool = True
    ) -> TransmutationBatch:
        """
        높은 확신도 패턴 자동 적용
        
        Args:
            confidence_threshold: 최소 확신도 (0-1)
            dry_run: True면 시뮬레이션만
            
        Returns:
            TransmutationBatch: 변환 결과
        """
        suggestions = self.alchemical_army.transmutation_cell.get_suggestions()
        high_confidence = [
            s for s in suggestions 
            if s.auto_applicable and s.confidence >= confidence_threshold
        ]
        
        if not high_confidence:
            print(f"   No suggestions with confidence >= {confidence_threshold:.0%}")
            return TransmutationBatch(
                batch_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
                created_at=datetime.now().isoformat()
            )
        
        print(f"   Found {len(high_confidence)} high-confidence suggestions")
        
        batch = self.auto_transmuter.transmute_with_approval(
            high_confidence,
            auto_approve=True,
            dry_run=dry_run
        )
        
        # Progress 업데이트
        if not dry_run:
            self.progress.total_transformed += batch.total_success
            self.progress.success_rate = (
                self.progress.total_transformed / max(self.progress.total_detected, 1)
            )
            self._save_progress()
        
        return batch
    
    def get_dashboard(self) -> Dict[str, Any]:
        """
        대시보드 데이터 반환
        
        Returns:
            대시보드 정보
        """
        return {
            "coherence": {
                "current": self.metrics.global_coherence,
                "trend": self.metrics.trend,
                "history_length": len(self.metrics.history),
            },
            "patterns": {
                "stone": self.metrics.stone_patterns,
                "wave": self.metrics.wave_patterns,
                "ratio": (
                    self.metrics.wave_patterns / 
                    max(self.metrics.wave_patterns + self.metrics.stone_patterns, 1)
                ),
            },
            "progress": {
                "detected": self.progress.total_detected,
                "transformed": self.progress.total_transformed,
                "pending": self.progress.total_pending,
                "success_rate": self.progress.success_rate,
                "estimated_gain": self.progress.estimated_coherence_gain,
            },
            "last_updated": self.metrics.last_updated,
        }
    
    def print_dashboard(self):
        """대시보드 출력"""
        dashboard = self.get_dashboard()
        
        print("\n" + "=" * 60)
        print("📊 CONTINUOUS TRANSMUTATION DASHBOARD")
        print("=" * 60)
        
        # Coherence
        coherence = dashboard["coherence"]
        trend_emoji = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(coherence["trend"], "❓")
        print(f"\n🌊 COHERENCE")
        print(f"   Current: {coherence['current']:.2%} {trend_emoji} {coherence['trend']}")
        print(f"   History points: {coherence['history_length']}")
        
        # Patterns
        patterns = dashboard["patterns"]
        print(f"\n🔍 PATTERNS")
        print(f"   Stone Logic: {patterns['stone']}")
        print(f"   Wave Logic: {patterns['wave']}")
        print(f"   Wave Ratio: {patterns['ratio']:.1%}")
        
        # Progress
        progress = dashboard["progress"]
        print(f"\n⚗️ TRANSMUTATION PROGRESS")
        print(f"   Detected: {progress['detected']}")
        print(f"   Transformed: {progress['transformed']}")
        print(f"   Pending: {progress['pending']}")
        print(f"   Success Rate: {progress['success_rate']:.1%}")
        print(f"   Est. Coherence Gain: +{progress['estimated_gain']:.3f}")
        
        print(f"\n   Last updated: {dashboard['last_updated']}")
        print("=" * 60)
    
    def start_background_patrol(self, interval_seconds: int = 300):
        """
        백그라운드 순찰 시작
        
        Args:
            interval_seconds: 순찰 간격 (초)
        """
        if self._patrol_thread and self._patrol_thread.is_alive():
            print("   ⚠️ Patrol already running")
            return
        
        self._stop_event.clear()
        
        def patrol_loop():
            while not self._stop_event.is_set():
                try:
                    self.patrol_once()
                except Exception as e:
                    logger.error(f"Patrol error: {e}")
                
                self._stop_event.wait(interval_seconds)
        
        self._patrol_thread = Thread(target=patrol_loop, daemon=True)
        self._patrol_thread.start()
        print(f"   🔄 Background patrol started (every {interval_seconds}s)")
    
    def stop_background_patrol(self):
        """백그라운드 순찰 중지"""
        self._stop_event.set()
        if self._patrol_thread:
            self._patrol_thread.join(timeout=5)
        print("   ⏹️ Background patrol stopped")
    
    def _load_metrics(self) -> CoherenceMetrics:
        """Metrics 로드"""
        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return CoherenceMetrics(
                        global_coherence=data.get("global_coherence", 0.0),
                        stone_patterns=data.get("stone_patterns", 0),
                        wave_patterns=data.get("wave_patterns", 0),
                        auto_applicable=data.get("auto_applicable", 0),
                        last_updated=data.get("last_updated", ""),
                        trend=data.get("trend", "stable"),
                        history=data.get("history", [])
                    )
            except Exception:
                pass
        return CoherenceMetrics()
    
    def _save_metrics(self):
        """Metrics 저장"""
        try:
            data = {
                "global_coherence": self.metrics.global_coherence,
                "stone_patterns": self.metrics.stone_patterns,
                "wave_patterns": self.metrics.wave_patterns,
                "auto_applicable": self.metrics.auto_applicable,
                "last_updated": self.metrics.last_updated,
                "trend": self.metrics.trend,
                "history": self.metrics.history
            }
            with open(self.metrics_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def _load_progress(self) -> TransmutationProgress:
        """Progress 로드"""
        if self.progress_path.exists():
            try:
                with open(self.progress_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return TransmutationProgress(
                        total_detected=data.get("total_detected", 0),
                        total_transformed=data.get("total_transformed", 0),
                        total_pending=data.get("total_pending", 0),
                        success_rate=data.get("success_rate", 0.0),
                        estimated_coherence_gain=data.get("estimated_coherence_gain", 0.0)
                    )
            except Exception:
                pass
        return TransmutationProgress()
    
    def _save_progress(self):
        """Progress 저장"""
        try:
            data = {
                "total_detected": self.progress.total_detected,
                "total_transformed": self.progress.total_transformed,
                "total_pending": self.progress.total_pending,
                "success_rate": self.progress.success_rate,
                "estimated_coherence_gain": self.progress.estimated_coherence_gain
            }
            with open(self.progress_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")


# ============= organic_wake.py 연동 함수 =============

_monitor_instance: Optional[ContinuousTransmutationMonitor] = None

def get_monitor() -> ContinuousTransmutationMonitor:
    """싱글톤 모니터 인스턴스 반환"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ContinuousTransmutationMonitor()
    return _monitor_instance


def patrol_and_report() -> Dict[str, Any]:
    """organic_wake.py에서 호출할 순찰 함수"""
    monitor = get_monitor()
    return monitor.patrol_once()


def get_coherence_status() -> str:
    """Coherence 상태 문자열 반환"""
    monitor = get_monitor()
    m = monitor.metrics
    return f"Coherence: {m.global_coherence:.2%} ({m.trend}) | Stone: {m.stone_patterns} | Auto-fix: {m.auto_applicable}"


# ============= 데모 =============

def demo_continuous_monitor():
    """연속 모니터 데모"""
    print("=" * 60)
    print("📊 Continuous Transmutation Monitor Demo")
    print("=" * 60)
    
    monitor = ContinuousTransmutationMonitor()
    
    # 1. 순찰 1회 수행
    print("\n[1] Single Patrol")
    print("-" * 40)
    result = monitor.patrol_once("Core")
    
    # 2. 상위 제안 출력
    print("\n[2] Top Suggestions")
    print("-" * 40)
    top = result.get("top_suggestions", [])
    for i, s in enumerate(top, 1):
        print(f"   [{i}] {s['file']}:{s['line']} - {s['type']} ({s['confidence']})")
    
    # 3. 대시보드 출력
    print("\n[3] Dashboard")
    monitor.print_dashboard()
    
    # 4. 높은 확신도 자동 적용 (Dry Run)
    print("\n[4] High-Confidence Auto-Apply (Dry Run)")
    print("-" * 40)
    batch = monitor.auto_apply_high_confidence(confidence_threshold=0.8, dry_run=True)
    print(f"   Would apply: {batch.total_success} transformations")
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous Transmutation Monitor")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--patrol", action="store_true", help="Single patrol")
    parser.add_argument("--dashboard", action="store_true", help="Show dashboard")
    parser.add_argument("--status", action="store_true", help="Quick status")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_continuous_monitor()
    elif args.patrol:
        monitor = ContinuousTransmutationMonitor()
        monitor.patrol_once()
    elif args.dashboard:
        monitor = ContinuousTransmutationMonitor()
        monitor.patrol_once()
        monitor.print_dashboard()
    elif args.status:
        print(get_coherence_status())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
