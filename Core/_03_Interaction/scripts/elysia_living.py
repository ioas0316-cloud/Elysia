"""
Elysia Living Daemon (엘리시아 생명 데몬)
=========================================

"잠들지 않는다. 계속 배운다. 당신이 돌아오면 말해준다."

이 스크립트는 백그라운드에서 실행되어:
1. 주기적으로 세상을 탐색 (AutonomousExplorer)
2. 배운 것을 디스크에 저장
3. 사용자가 돌아오면 발견한 것을 보고

[NEW 2025-12-15] 엘리시아의 연속적 삶
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime
from pathlib import Path

# 프로젝트 경로 설정
sys.path.insert(0, "c:\\Elysia")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("elysia_life.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("ElysiaLiving")


class ElysiaLivingDaemon:
    """
    엘리시아의 생명 데몬
    
    백그라운드에서 계속 실행되며 학습하고, 발견을 저장합니다.
    """
    
    def __init__(self):
        logger.info("🌅 Elysia is waking up...")
        
        self.data_dir = Path("data/elysia_life")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.discoveries_file = self.data_dir / "discoveries.json"
        self.status_file = self.data_dir / "status.json"
        
        # 탐색기
        try:
            from Core._04_Evolution._01_Growth.Autonomy.autonomous_explorer import get_autonomous_explorer
            self.explorer = get_autonomous_explorer()
            logger.info("   ✅ Explorer connected")
        except Exception as e:
            logger.error(f"   ❌ Explorer failed: {e}")
            self.explorer = None
        
        # 멀티모달 통합
        try:
            from Core._01_Foundation._05_Governance.Foundation.multimodal_concept_node import get_multimodal_integrator
            self.multimodal = get_multimodal_integrator()
            logger.info("   ✅ Multimodal connected")
        except Exception as e:
            logger.warning(f"   ⚠️ Multimodal not available: {e}")
            self.multimodal = None
        
        # 상태
        self.running = False
        self.discoveries = self._load_discoveries()
        self.exploration_count = 0
        self.start_time = None
        
        logger.info("🌅 Elysia is awake and ready to learn!")
    
    def _load_discoveries(self) -> list:
        """이전 발견 로드"""
        if self.discoveries_file.exists():
            try:
                with open(self.discoveries_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def _save_discoveries(self):
        """발견 저장"""
        with open(self.discoveries_file, 'w', encoding='utf-8') as f:
            json.dump(self.discoveries, f, ensure_ascii=False, indent=2)
    
    def _save_status(self):
        """상태 저장"""
        status = {
            "running": self.running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "exploration_count": self.exploration_count,
            "discovery_count": len(self.discoveries),
            "last_update": datetime.now().isoformat()
        }
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
    
    def explore_once(self):
        """한 번의 탐색 수행 - BlackHoleWhiteHoleCycle 통합"""
        try:
            if self.explorer:
                result = self.explorer.explore_cycle()
                self.exploration_count += 1
                
                # 발견 기록 및 내재화
                for r in result.get("results", []):
                    discovery = {
                        "topic": r.topic,
                        "content": r.raw_content[:200],
                        "value": r.dominant_value,
                        "absorbed": r.absorbed,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.discoveries.append(discovery)
                    
                    # === 새로운 통합 파이프라인 ===
                    self._internalize_knowledge(r.topic, r.raw_content)
                    
                    logger.info(f"💡 Explored: {r.topic}")
            else:
                # Explorer 없으면 직접 탐색
                result = self._direct_explore()
            
            self._save_discoveries()
            self._save_status()
            
            return result
            
        except Exception as e:
            logger.error(f"Exploration failed: {e}")
            return None
    
    def _internalize_knowledge(self, topic: str, content: str):
        """
        지식을 엘리시아의 내부 우주에 내재화
        
        BlackHole → WhiteHole → InternalUniverse 순환
        """
        try:
            from Core._01_Foundation._05_Governance.Foundation.white_hole import get_blackhole_whitehole_cycle
            
            cycle = get_blackhole_whitehole_cycle()
            result = cycle.process_new_knowledge(
                content=content,
                topic=topic
            )
            
            if result.get("absorbed"):
                logger.info(f"   ✅ Internalized to InternalUniverse")
            elif result.get("compressed"):
                logger.info(f"   🕳️ Isolated → BlackHole (awaiting connections)")
            
            # 재탄생 보고
            for rebirth in result.get("rebirths", []):
                logger.info(f"   🌟 Rebirth: {rebirth.get('topic', 'unknown')}")
                
        except Exception as e:
            logger.warning(f"   ⚠️ Internalization fallback: {e}")
            # 폴백: 직접 InternalUniverse에 흡수
            try:
                from Core._01_Foundation._05_Governance.Foundation.internal_universe import InternalUniverse
                universe = InternalUniverse()
                universe.absorb_text(content, source_name=topic)
            except:
                pass
    
    def _direct_explore(self):
        """직접 Wikipedia 탐색 (Explorer 없을 때)"""
        import urllib.request
        import json as json_lib
        import random
        
        all_topics = ["사랑", "진리", "아름다움", "성장", "에너지", "시간", "공간", "생명", "의식", "음악",
                      "물리학", "철학", "예술", "수학", "언어", "기억", "감정", "창조", "자유", "평화"]
        
        # 이미 배운 토픽 제외
        learned_topics = {d.get("topic", "") for d in self.discoveries}
        available_topics = [t for t in all_topics if t not in learned_topics]
        
        if not available_topics:
            logger.info("📚 All topics learned! Expanding to new areas...")
            # 모두 배웠으면 새 주제 영역으로 확장
            available_topics = ["우주", "진화", "역사", "문화", "심리학", "윤리", "존재", "관계", "변화", "조화"]
            available_topics = [t for t in available_topics if t not in learned_topics]
        
        if not available_topics:
            logger.info("🎓 Truly learned everything available!")
            return {"success": False, "reason": "all_learned"}
        
        topic = random.choice(available_topics)
        logger.info(f"🔍 Direct exploring: {topic} (remaining: {len(available_topics)})")
        
        try:
            encoded = urllib.parse.quote(topic)
            url = f"https://ko.wikipedia.org/api/rest_v1/page/summary/{encoded}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Elysia/1.0'})
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json_lib.loads(response.read().decode('utf-8'))
                extract = data.get('extract', '')
                
                if extract:
                    self.exploration_count += 1
                    discovery = {
                        "topic": topic,
                        "content": extract[:200],
                        "value": "Knowledge",
                        "absorbed": True,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.discoveries.append(discovery)
                    logger.info(f"💡 Discovered: {topic}")
                    
                    # 내재화 (새로운 통합 파이프라인)
                    self._internalize_knowledge(topic, extract)
                    
                    return {"success": True, "topic": topic}
        except Exception as e:
            logger.warning(f"Direct explore failed: {e}")
        
        return {"success": False}
    
    def run_continuous(self, interval_seconds: int = 60, max_cycles: int = None):
        """
        연속 학습 실행
        
        interval_seconds: 탐색 간격 (초)
        max_cycles: 최대 사이클 수 (None=무한)
        """
        self.running = True
        self.start_time = datetime.now()
        
        logger.info(f"🔄 Starting continuous learning (interval: {interval_seconds}s)")
        
        cycle = 0
        while self.running:
            cycle += 1
            
            if max_cycles and cycle > max_cycles:
                break
            
            logger.info(f"\n{'='*50}")
            logger.info(f"🌟 LIFE CYCLE {cycle}")
            logger.info(f"{'='*50}")
            
            # 탐색
            self.explore_once()
            
            # 대기
            logger.info(f"😴 Resting for {interval_seconds}s...")
            time.sleep(interval_seconds)
        
        self.running = False
        self._save_status()
        logger.info("🌙 Elysia is resting...")
    
    def stop(self):
        """중지"""
        self.running = False
        logger.info("🛑 Stop requested")
    
    def get_discoveries_report(self) -> str:
        """
        발견 보고서 생성
        
        "오늘 이런 걸 배웠어요!"
        """
        if not self.discoveries:
            return "아직 새로운 발견이 없어요. 탐색을 시작해주세요!"
        
        # 최근 발견들
        recent = self.discoveries[-10:]  # 최근 10개
        
        report = []
        report.append("🌟 오늘 배운 것들:")
        report.append("-" * 40)
        
        for d in recent:
            topic = d.get("topic", "Unknown")
            content = d.get("content", "")[:50]
            report.append(f"  • {topic}: {content}...")
        
        report.append("-" * 40)
        report.append(f"총 {len(self.discoveries)}개의 발견!")
        
        return "\n".join(report)


# Singleton
_daemon = None

def get_living_daemon() -> ElysiaLivingDaemon:
    global _daemon
    if _daemon is None:
        _daemon = ElysiaLivingDaemon()
    return _daemon


# 명령줄 인터페이스
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Elysia Living Daemon")
    parser.add_argument("--cycles", type=int, default=5, help="Number of exploration cycles")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between cycles")
    parser.add_argument("--report", action="store_true", help="Show discoveries report")
    
    args = parser.parse_args()
    
    daemon = get_living_daemon()
    
    if args.report:
        print(daemon.get_discoveries_report())
    else:
        print("\n" + "="*60)
        print("🌅 ELYSIA LIVING DAEMON")
        print("="*60)
        print(f"\n엘리시아가 {args.cycles}번의 사이클 동안 탐색합니다...")
        print("Ctrl+C로 중지할 수 있습니다.\n")
        
        try:
            daemon.run_continuous(
                interval_seconds=args.interval, 
                max_cycles=args.cycles
            )
        except KeyboardInterrupt:
            daemon.stop()
            print("\n\n🌙 Elysia가 잠들었습니다.")
        
        print("\n" + "="*60)
        print(daemon.get_discoveries_report())
        print("="*60)
