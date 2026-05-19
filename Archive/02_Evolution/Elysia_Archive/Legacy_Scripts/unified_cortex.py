"""
Unified Cortex (통합 대뇌피질)
==============================

"엘리시아의 모든 시스템을 통합 관리하는 중앙 피질"

[역할]
1. central_registry.json 읽어 모든 모듈 파악
2. 필요한 엔진 동적 로드
3. 허브 모듈 간 신호 조율
4. 고아 모듈 활성화
5. 자기 재조직화 실행

[엘리시아가 사용하는 방법]
```python
from scripts.unified_cortex import UnifiedCortex

cortex = UnifiedCortex()
cortex.awaken()  # 모든 시스템 깨우기

# 특정 능력 사용
emotion = cortex.get_engine("emotion")
sensation = cortex.get_engine("sensation")

# 자기 재조직화
cortex.reorganize()
```
"""

import sys
import json
import importlib
import importlib.util
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class EngineStatus:
    """엔진 상태"""
    name: str
    path: str
    loaded: bool = False
    instance: Any = None
    error: str = ""


class UnifiedCortex:
    """
    통합 대뇌피질
    
    엘리시아의 모든 시스템을 통합 관리합니다.
    """
    
    def __init__(self):
        self.root = PROJECT_ROOT
        self.registry: Dict = {}
        self.structure_map: Dict = {}
        self.engines: Dict[str, Dict[str, EngineStatus]] = defaultdict(dict)
        self.is_awakened = False
        
        # 레지스트리 로드
        self._load_registry()
    
    def _load_registry(self):
        """중앙 레지스트리 로드"""
        registry_path = self.root / "data" / "central_registry.json"
        structure_path = self.root / "data" / "self_structure_map.json"
        
        if registry_path.exists():
            with open(registry_path, 'r', encoding='utf-8') as f:
                self.registry = json.load(f)
            print("✅ Central Registry loaded")
        else:
            print("⚠️ Central Registry not found - run self_integration.py first")
        
        if structure_path.exists():
            with open(structure_path, 'r', encoding='utf-8') as f:
                self.structure_map = json.load(f)
            print("✅ Structure Map loaded")
    
    def awaken(self):
        """모든 시스템 깨우기"""
        print("\n" + "=" * 70)
        print("🧠 UNIFIED CORTEX AWAKENING")
        print("=" * 70)
        
        if not self.registry:
            print("❌ Cannot awaken - no registry loaded")
            return False
        
        # 핵심 허브 먼저 로드
        print("\n🌐 Loading Core Hubs...")
        for hub_name, hub_path in self.registry.get("core_hubs", {}).items():
            self._load_engine(hub_name, hub_path, "core")
        
        # 범주별 엔진 로드
        print("\n⚡ Loading Engines by Category...")
        for category, engines in self.registry.get("engines", {}).items():
            print(f"\n   📂 {category}:")
            for engine_path in engines:
                engine_name = Path(engine_path).stem
                self._load_engine(engine_name, engine_path, category)
        
        self.is_awakened = True
        print("\n" + "=" * 70)
        print("✅ Unified Cortex Fully Awakened")
        print("=" * 70)
        
        return True
    
    def _load_engine(self, name: str, path: str, category: str):
        """엔진 동적 로드"""
        full_path = self.root / path
        
        status = EngineStatus(name=name, path=path)
        
        if not full_path.exists():
            status.error = "File not found"
            print(f"      ❌ {name}: File not found")
        else:
            try:
                # 동적 모듈 로드
                spec = importlib.util.spec_from_file_location(name, full_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    status.loaded = True
                    status.instance = module
                    print(f"      ✅ {name}")
            except Exception as e:
                status.error = str(e)[:50]
                print(f"      ⚠️ {name}: {status.error}")
        
        self.engines[category][name] = status
    
    def get_engine(self, category: str, name: str = None) -> Optional[Any]:
        """
        엔진 가져오기
        
        Examples:
            cortex.get_engine("emotion")  # 첫 번째 감정 엔진
            cortex.get_engine("emotion", "empathy")  # 특정 엔진
        """
        if category not in self.engines:
            return None
        
        if name:
            status = self.engines[category].get(name)
            return status.instance if status and status.loaded else None
        else:
            # 첫 번째 로드된 엔진 반환
            for status in self.engines[category].values():
                if status.loaded:
                    return status.instance
            return None
    
    def list_available(self) -> Dict[str, List[str]]:
        """사용 가능한 엔진 목록"""
        available = {}
        for category, engines in self.engines.items():
            available[category] = [
                name for name, status in engines.items() if status.loaded
            ]
        return available
    
    def reorganize(self):
        """자기 재조직화 실행"""
        print("\n🔄 SELF-REORGANIZATION")
        print("-" * 50)
        
        if not self.structure_map:
            print("❌ No structure map - cannot reorganize")
            return
        
        actions = self.structure_map.get("integration_actions", [])
        print(f"   Pending actions: {len(actions)}")
        
        # 우선순위 높은 것부터
        sorted_actions = sorted(actions, key=lambda x: x.get("priority", 0), reverse=True)
        
        for action in sorted_actions[:10]:
            print(f"\n   [{action['type'].upper()}]")
            print(f"      {action['source']} → {action['target']}")
    
    def get_health_report(self) -> Dict:
        """건강 보고서"""
        report = {
            "awakened": self.is_awakened,
            "categories": {},
            "total_loaded": 0,
            "total_failed": 0
        }
        
        for category, engines in self.engines.items():
            loaded = sum(1 for s in engines.values() if s.loaded)
            failed = sum(1 for s in engines.values() if not s.loaded)
            
            report["categories"][category] = {
                "loaded": loaded,
                "failed": failed,
                "engines": list(engines.keys())
            }
            report["total_loaded"] += loaded
            report["total_failed"] += failed
        
        return report
    
    def activate_dormant(self, engine_name: str) -> bool:
        """휴면 엔진 활성화"""
        for category, engines in self.engines.items():
            if engine_name in engines:
                status = engines[engine_name]
                if not status.loaded:
                    self._load_engine(engine_name, status.path, category)
                    return engines[engine_name].loaded
        return False
    
    def connect_modules(self, source: str, target: str) -> bool:
        """모듈 연결 (논리적)"""
        # 실제 코드 수정 없이 연결 관계만 기록
        connection_log = self.root / "data" / "module_connections.json"
        
        connections = []
        if connection_log.exists():
            with open(connection_log, 'r', encoding='utf-8') as f:
                connections = json.load(f)
        
        connections.append({
            "source": source,
            "target": target,
            "timestamp": str(Path(__file__).stat().st_mtime)
        })
        
        with open(connection_log, 'w', encoding='utf-8') as f:
            json.dump(connections, f, indent=2)
        
        return True
    
    def status_summary(self):
        """상태 요약 출력"""
        print("\n" + "=" * 70)
        print("🧠 UNIFIED CORTEX STATUS")
        print("=" * 70)
        
        report = self.get_health_report()
        
        print(f"\n   Awakened: {'✅ Yes' if report['awakened'] else '❌ No'}")
        print(f"   Total Loaded: {report['total_loaded']}")
        print(f"   Total Failed: {report['total_failed']}")
        
        print("\n   📂 BY CATEGORY:")
        for cat, info in report["categories"].items():
            status = "✅" if info["loaded"] > 0 else "❌"
            print(f"      {status} {cat}: {info['loaded']}/{info['loaded'] + info['failed']}")
        
        print("\n" + "=" * 70)


def main():
    print("\n" + "🧠" * 35)
    print("UNIFIED CORTEX ACTIVATION")
    print("🧠" * 35 + "\n")
    
    cortex = UnifiedCortex()
    cortex.awaken()
    cortex.status_summary()
    
    # 사용 가능한 엔진 표시
    available = cortex.list_available()
    print("\n📋 AVAILABLE ENGINES:")
    for category, engines in available.items():
        if engines:
            print(f"   {category}: {', '.join(engines)}")


if __name__ == "__main__":
    main()
