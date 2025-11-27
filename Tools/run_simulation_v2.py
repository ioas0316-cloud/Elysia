"""
Ultra-Dense Simulation V2 - 완전 통합 시뮬레이션 루프

이전 버전과의 차이:
- MetaTimeStrategy로 지능적 공명 계산
- IntegrationBridge로 모든 이벤트 통합
- AgentDecisionEngine으로 동적 전략 선택
- 성능 피드백 루프 (10,000 틱마다 분석)

효과:
- 이전: 71분 실행
- 현재: ~15분 예상 (메모리 효율 + 계산 최적화)
"""

import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import logging
import time as real_time
import json
from typing import Dict, Optional

from Core.Physics.fluctlight import FluctlightEngine
from Core.Physics.meta_time_engine import create_safe_meta_engine
from Core.Integration.experience_digester import ExperienceDigester
from Core.Integration.meta_time_strategy import MetaTimeStrategy, TemporalMode, ComputationProfile
from Core.Integration.integration_bridge import IntegrationBridge
from Core.Consciousness.agent_decision_engine import AgentDecisionEngine, AgentContext
from Core.Mind.hippocampus import Hippocampus
from Core.Mind.alchemy import Alchemy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SimulationV2")


class SimulationMetrics:
    """시뮬레이션 메트릭 추적"""
    
    def __init__(self):
        self.total_ticks = 0
        self.ticks_since_checkpoint = 0
        self.total_real_time = 0.0
        self.checkpoint_times = []
        
        # 전략 통계
        self.strategy_usage = {}
        self.strategy_performance = {}
        
        # 메모리 통계
        self.memory_usage = []
        self.peak_memory = 0
        
        # 개념 통계
        self.concept_count = 0
        self.relationships_count = 0
    
    def record_tick(self, real_time_ms: float):
        """틱 기록"""
        self.total_ticks += 1
        self.ticks_since_checkpoint += 1
        self.total_real_time += real_time_ms
    
    def record_checkpoint(self, time_ms: float, strategy: str):
        """체크포인트 기록"""
        self.checkpoint_times.append({
            'tick': self.total_ticks,
            'time': time_ms,
            'strategy': strategy
        })
    
    def get_avg_tick_time(self) -> float:
        """평균 틱 시간 (ms)"""
        if self.total_ticks == 0:
            return 0.0
        return self.total_real_time / self.total_ticks
    
    def get_speedup_from_baseline(self) -> float:
        """기준 대비 속도 향상 (기준: 1ms/틱)"""
        avg_tick_time = self.get_avg_tick_time()
        if avg_tick_time == 0:
            return 0.0
        return 1.0 / avg_tick_time  # ms를 역수로 변환


def run_simulation_v2(
    total_ticks: int = 50000,
    checkpoint_interval: int = 10000,
    max_particles: int = 2000,
    interference_interval: int = 2,
    depth: int = 2
) -> Dict:
    """
    통합 시뮬레이션 실행
    
    Args:
        total_ticks: 총 틱 수
        checkpoint_interval: 분석 체크포인트 간격
        max_particles: 최대 입자 수
        interference_interval: 간섭 빈도
        depth: 재귀 깊이
    
    Returns:
        결과 딕셔너리
    """
    
    logger.info("\n" + "🚀"*35)
    logger.info(" "*10 + "SIMULATION V2 - INTEGRATED")
    logger.info(" "*5 + "MetaTime + Integration Bridge + Agent Decision")
    logger.info("🚀"*35 + "\n")
    
    # 설정 출력
    logger.info("Configuration:")
    logger.info(f"  Total ticks: {total_ticks:,}")
    logger.info(f"  Checkpoint: {checkpoint_interval:,} ticks")
    logger.info(f"  Max particles: {max_particles:,}")
    logger.info(f"  Interference: every {interference_interval} ticks")
    logger.info(f"  Recursion depth: {depth}\n")
    
    # 1. 엔진 초기화
    logger.info("Initializing engines...")
    start_real_time = real_time.time()
    
    fluctlight = FluctlightEngine(world_size=256)
    meta_time = create_safe_meta_engine(
        recursion_depth=depth,
        base_compression=1000.0,
        enable_black_holes=True
    )
    hippocampus = Hippocampus()
    alchemy = Alchemy()
    
    # 통합 모듈
    meta_strategy = MetaTimeStrategy()
    bridge = IntegrationBridge()
    agent_engine = AgentDecisionEngine(enable_learning=True)
    digester = ExperienceDigester(hippocampus)
    
    metrics = SimulationMetrics()
    
    logger.info("✅ Engines initialized\n")
    
    # 2. 시뮬레이션 루프
    logger.info(f"Starting simulation loop ({total_ticks:,} ticks)...")
    
    checkpoint_num = 0
    
    try:
        for tick in range(total_ticks):
            tick_start = real_time.time()
            
            # A. 에이전트가 전략 결정
            context = AgentContext(
                focus="universal_learning",
                goal=f"Tick {tick}: Generate diverse concepts",
                tick=tick,
                available_memory_mb=max(300, 2000 - metrics.get_avg_tick_time()),
                time_pressure=0.1 + (tick / total_ticks) * 0.3,  # 시간 압박 증가
                concept_count=metrics.concept_count
            )
            
            decision = agent_engine.decide(context)
            
            # B. 전략 적용
            meta_strategy.set_temporal_mode(decision.temporal_mode)
            meta_strategy.set_computation_profile(decision.computation_profile)
            
            # C. 시뮬레이션 스텝
            experience = fluctlight.step(
                detect_interference=(tick % interference_interval == 0)
            )
            
            # D. 공명 계산 (지능적)
            resonances = {}
            if hippocampus.causal_graph.nodes():
                concept_ids = list(hippocampus.causal_graph.nodes())[:10]  # 최대 10개
                resonances = meta_strategy.get_intelligent_resonances(
                    concept_ids[0] if concept_ids else "universal",
                    {cid: {"resonance": 0.5} for cid in concept_ids}
                )
            
            # E. 이벤트 발행 (IntegrationBridge)
            if resonances:
                event = bridge.publish_resonance(
                    source="fluctlight",
                    resonances=resonances,
                    tick=tick
                )
            
            # F. 경험 소화
            if tick % 100 == 0:  # 100 틱마다
                agent_engine.record_performance(
                    decision.temporal_mode,
                    decision.predicted_speedup
                )
            
            # 체크포인트
            tick_end = real_time.time()
            tick_time_ms = (tick_end - tick_start) * 1000
            metrics.record_tick(tick_time_ms)
            
            if (tick + 1) % checkpoint_interval == 0:
                checkpoint_num += 1
                elapsed = (tick_end - start_real_time) / 60  # 분
                avg_tick = metrics.get_avg_tick_time()
                speedup = metrics.get_speedup_from_baseline()
                
                logger.info(
                    f"Checkpoint {checkpoint_num}: "
                    f"Tick {tick+1:,} | "
                    f"Time {elapsed:.1f}min | "
                    f"Avg {avg_tick:.2f}ms/tick | "
                    f"Speedup {speedup:.1f}x | "
                    f"Mode {decision.temporal_mode.value}"
                )
                
                metrics.record_checkpoint(elapsed, decision.temporal_mode.value)
        
        logger.info("\n" + "="*70)
        logger.info("FINAL RESULTS")
        logger.info("="*70)
        
        total_real_time = (real_time.time() - start_real_time) / 60
        avg_tick_time = metrics.get_avg_tick_time()
        speedup = metrics.get_speedup_from_baseline()
        
        logger.info(f"Total ticks: {metrics.total_ticks:,}")
        logger.info(f"Real time: {total_real_time:.2f} minutes")
        logger.info(f"Avg tick: {avg_tick_time:.3f}ms")
        logger.info(f"Speedup: {speedup:.1f}x")
        logger.info(f"Concepts: {metrics.concept_count}")
        logger.info(f"Relationships: {metrics.relationships_count}")
        
        # 최고 성능 전략
        logger.info("\nBest strategies learned:")
        best_strategies = agent_engine.get_best_strategy_history(limit=3)
        for mode, score in best_strategies:
            logger.info(f"  {mode}: {score:.2f}x speedup")
        
        # 통계 내보내기
        logger.info("\nExporting statistics...")
        agent_engine.export_statistics("data/simulation_v2_statistics.json")
        bridge.export_event_log("data/simulation_v2_events.jsonl")
        
        logger.info("✅ Simulation completed successfully!")
        logger.info("="*70 + "\n")
        
        return {
            "status": "success",
            "total_ticks": metrics.total_ticks,
            "real_time_minutes": total_real_time,
            "avg_tick_ms": avg_tick_time,
            "speedup_factor": speedup,
            "concepts": metrics.concept_count,
            "relationships": metrics.relationships_count,
            "checkpoints": len(metrics.checkpoint_times)
        }
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Simulation interrupted by user")
        return {
            "status": "interrupted",
            "total_ticks": metrics.total_ticks,
            "real_time_minutes": (real_time.time() - start_real_time) / 60
        }
    
    except Exception as e:
        logger.error(f"\n❌ Error during simulation: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "total_ticks": metrics.total_ticks,
            "real_time_minutes": (real_time.time() - start_real_time) / 60
        }


if __name__ == "__main__":
    # 빠른 테스트 모드 (1000 틱)
    result = run_simulation_v2(
        total_ticks=1000,
        checkpoint_interval=250,
        max_particles=500,
        interference_interval=5,
        depth=1
    )
    
    # 결과 출력
    print("\n[RESULT SUMMARY]")
    print(json.dumps(result, indent=2, ensure_ascii=False))
