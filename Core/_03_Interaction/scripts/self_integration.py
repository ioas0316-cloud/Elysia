"""
Self-Integration System (자가 통합 시스템)
==========================================

"엘리시아가 스스로 자신의 구조를 파악하고 재정렬한다"

[핵심 원칙]
1. 엘리시아가 직접 자신의 모듈 구조를 이해
2. 고아 모듈을 중앙 허브에 연결
3. 미사용 엔진을 활성화
4. 구조를 최적화하고 재정렬
5. 파동 공명으로 자연스러운 클러스터링

[자가 능력]
- 스스로 모듈 스캔
- 스스로 연결 상태 파악
- 스스로 통합 제안
- 스스로 구조 재정렬
"""

import os
import sys
import ast
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Any, Optional, Tuple
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 기존 시스템 import 시도
try:
    from Core._01_Foundation._05_Governance.Foundation.resonance_field import ResonanceField
    RESONANCE_AVAILABLE = True
except ImportError:
    RESONANCE_AVAILABLE = False

try:
    from Core._01_Foundation._02_Logic.hyper_quaternion import HyperQuaternion
    QUATERNION_AVAILABLE = True
except ImportError:
    QUATERNION_AVAILABLE = False


@dataclass
class ModuleNode:
    """모듈 노드 - 코드베이스의 하나의 세포"""
    path: str
    name: str
    lines: int
    classes: List[str] = field(default_factory=list)
    imports: Set[str] = field(default_factory=set)
    imported_by: Set[str] = field(default_factory=set)
    frequency: float = 440.0  # 파동 주파수
    category: str = "unknown"
    is_active: bool = True
    is_hub: bool = False
    is_orphan: bool = False


@dataclass
class IntegrationAction:
    """통합 작업"""
    action_type: str  # "connect", "activate", "reorganize"
    source: str
    target: str
    reason: str
    priority: int = 5


class SelfIntegrationSystem:
    """
    자가 통합 시스템
    
    엘리시아가 스스로 자신의 구조를 이해하고 재정렬합니다.
    """
    
    # 핵심 허브 모듈 (모든 것이 연결되어야 함)
    CORE_HUBS = [
        "hippocampus",       # 기억 중심
        "resonance_field",   # 파동장
        "reasoning_engine",  # 추론
        "hyper_quaternion",  # 4D 수학
        "emotional_engine",  # 감정
    ]
    
    # 범주별 주파수 (Hz) - wave_organizer와 동기화
    CATEGORY_FREQUENCIES = {
        "language": 440,       # 언어 - 라(A) 주파수
        "memory": 396,         # 기억 - 솔(G) 주파수
        "reasoning": 528,      # 추론 - 도(C) 치유 주파수
        "emotion": 639,        # 감정 - 미(E)
        "consciousness": 741,  # 의식 - 파(F#)
        "evolution": 852,      # 진화 - 라(A) 상위
        "physics": 963,        # 물리 - 시(B)
        "interface": 417,      # 인터페이스
        "creativity": 693,     # 창의성
        "ethics": 432,         # 윤리 - 우주 주파수
        "sensation": 639,      # 감각 (emotion과 공명)
        "transcendence": 963,  # 초월 (physics와 공명)
    }
    
    # 파동 기반 분류 키워드 - wave_organizer와 동기화
    PURPOSE_KEYWORDS = {
        "language": ["language", "grammar", "syntax", "hangul", "babel", "speech", "word", "syllable", "dialogue", "utterance"],
        "memory": ["memory", "hippocampus", "remember", "store", "recall", "cache", "history", "archive"],
        "reasoning": ["reason", "logic", "think", "causal", "infer", "deduc", "analysis", "cognitive", "goal", "plan"],
        "emotion": ["emotion", "feel", "empathy", "sentiment", "affect", "mood", "heart"],
        "consciousness": ["conscious", "aware", "soul", "spirit", "self", "mind", "identity", "attention"],
        "evolution": ["evolve", "mutate", "adapt", "grow", "learn", "train", "improve", "genesis"],
        "physics": ["physics", "wave", "quaternion", "gravity", "magnetic", "tensor", "field", "resonance", "ether", "hyper"],
        "interface": ["interface", "api", "server", "web", "chat", "user", "http", "dashboard", "envoy"],
        "creativity": ["dream", "imagine", "create", "art", "story", "saga", "poem", "visualize", "cosmic", "studio"],
        "ethics": ["ethics", "moral", "value", "protect", "guard", "safe", "law", "immune", "security"],
    }
    
    EXCLUDE = ["__pycache__", "node_modules", ".godot", ".venv", "Legacy"]
    
    def __init__(self):
        self.root = PROJECT_ROOT
        self.modules: Dict[str, ModuleNode] = {}
        self.integration_queue: List[IntegrationAction] = []
        
        print("=" * 80)
        print("🔄 SELF-INTEGRATION SYSTEM")
        print("엘리시아가 스스로 구조를 파악하고 재정렬합니다")
        print("=" * 80)
    
    def perceive_self(self):
        """1단계: 자기 인식 - 모든 모듈 스캔"""
        print("\n👁️ PHASE 1: SELF-PERCEPTION (자기 인식)")
        print("-" * 60)
        
        for py_file in self.root.rglob("*.py"):
            if any(p in str(py_file) for p in self.EXCLUDE):
                continue
            if py_file.stat().st_size < 100:
                continue
            
            rel_path = str(py_file.relative_to(self.root)).replace("\\", "/")
            
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                lines = len(content.split('\n'))
                classes = []
                imports = set()
                
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            classes.append(node.name)
                        elif isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.add(alias.name.split(".")[-1])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.add(node.module.split(".")[-1])
                except SyntaxError:
                    pass
                
                # 범주 및 주파수 결정
                category = self._categorize_module(py_file.stem, classes, content)
                frequency = self.CATEGORY_FREQUENCIES.get(category, 440.0)
                
                self.modules[rel_path] = ModuleNode(
                    path=rel_path,
                    name=py_file.stem,
                    lines=lines,
                    classes=classes,
                    imports=imports,
                    frequency=frequency,
                    category=category
                )
                
            except Exception:
                pass
        
        print(f"   Perceived {len(self.modules)} modules")
    
    def analyze_connections(self):
        """2단계: 연결 분석"""
        print("\n🔗 PHASE 2: CONNECTION ANALYSIS (연결 분석)")
        print("-" * 60)
        
        # 모듈 이름 → 경로 매핑
        name_to_path = {m.name: path for path, m in self.modules.items()}
        
        # imported_by 채우기
        for path, module in self.modules.items():
            for imp in module.imports:
                if imp in name_to_path:
                    target_path = name_to_path[imp]
                    if target_path in self.modules:
                        self.modules[target_path].imported_by.add(path)
        
        # 허브와 고아 식별
        hub_count = 0
        orphan_count = 0
        
        for path, module in self.modules.items():
            # 허브 = 5개 이상이 import
            if len(module.imported_by) >= 5:
                module.is_hub = True
                hub_count += 1
            
            # 고아 = 아무도 import하지 않고 스크립트도 아님
            if len(module.imported_by) == 0:
                if not any(x in path for x in ["scripts/", "tests/", "__main__"]):
                    module.is_orphan = True
                    orphan_count += 1
        
        print(f"   Found {hub_count} hub modules")
        print(f"   Found {orphan_count} orphan modules")
    
    def plan_integration(self):
        """3단계: 통합 계획 수립"""
        print("\n📋 PHASE 3: INTEGRATION PLANNING (통합 계획)")
        print("-" * 60)
        
        # 대형 고아 모듈 찾기
        large_orphans = [
            m for m in self.modules.values()
            if m.is_orphan and m.lines > 200
        ]
        
        print(f"   Large orphan modules to integrate: {len(large_orphans)}")
        
        for orphan in sorted(large_orphans, key=lambda x: x.lines, reverse=True)[:20]:
            # 가장 적합한 허브 찾기
            best_hub = self._find_best_hub(orphan)
            
            if best_hub:
                self.integration_queue.append(IntegrationAction(
                    action_type="connect",
                    source=orphan.path,
                    target=best_hub.path,
                    reason=f"Orphan module ({orphan.lines} lines) should connect to {best_hub.name}",
                    priority=min(10, orphan.lines // 100)
                ))
        
        # 미사용 엔진 활성화
        for path, module in self.modules.items():
            for cls in module.classes:
                if "Engine" in cls and module.is_orphan:
                    self.integration_queue.append(IntegrationAction(
                        action_type="activate",
                        source=path,
                        target="Core/Foundation/reasoning_engine.py",
                        reason=f"Dormant engine {cls} should be activated",
                        priority=7
                    ))
        
        print(f"   Planned {len(self.integration_queue)} integration actions")
    
    def _find_best_hub(self, orphan: ModuleNode) -> Optional[ModuleNode]:
        """고아 모듈에 가장 적합한 허브 찾기"""
        # 같은 범주의 허브 찾기
        for path, module in self.modules.items():
            if module.is_hub and module.category == orphan.category:
                return module
        
        # 없으면 기본 허브 (reasoning_engine)
        for path, module in self.modules.items():
            if "reasoning_engine" in path:
                return module
        
        return None
    
    def _categorize_module(self, name: str, classes: List[str], content: str) -> str:
        """
        모듈 범주 결정 - 파동 기반 분류
        
        wave_organizer와 동일한 로직: 키워드 매칭 + 컨텐츠 분석
        """
        content_lower = content.lower()
        name_lower = name.lower()
        classes_str = " ".join(classes).lower()
        
        # 전체 텍스트 결합 (이름 + 클래스 + 내용)
        combined = name_lower + " " + classes_str + " " + content_lower
        
        # 각 목적별 키워드 매칭 점수 계산
        best_category = ("unknown", 0)
        
        for purpose, keywords in self.PURPOSE_KEYWORDS.items():
            score = sum(combined.count(kw) for kw in keywords)
            # 이름에 키워드가 있으면 가중치 추가
            name_bonus = sum(3 for kw in keywords if kw in name_lower)
            # 클래스명에 키워드가 있으면 가중치 추가
            class_bonus = sum(2 for kw in keywords if kw in classes_str)
            total_score = score + name_bonus + class_bonus
            
            if total_score > best_category[1]:
                best_category = (purpose, total_score)
        
        return best_category[0]
    
    def execute_integration(self, auto_write: bool = False):
        """4단계: 통합 실행"""
        print("\n⚡ PHASE 4: INTEGRATION EXECUTION (통합 실행)")
        print("-" * 60)
        
        # 우선순위로 정렬
        sorted_actions = sorted(self.integration_queue, key=lambda x: x.priority, reverse=True)
        
        for action in sorted_actions[:15]:  # 상위 15개만
            print(f"\n   [{action.action_type.upper()}]")
            print(f"   Source: {action.source}")
            print(f"   Target: {action.target}")
            print(f"   Reason: {action.reason}")
        
        if not auto_write:
            print("\n   ⚠️ Dry run mode - no changes made")
            print("   Set auto_write=True to apply changes")
    
    def generate_structure_map(self) -> Dict:
        """5단계: 구조 맵 생성 (엘리시아가 참조할 수 있음)"""
        print("\n🗺️ PHASE 5: STRUCTURE MAP GENERATION")
        print("-" * 60)
        
        structure = {
            "total_modules": len(self.modules),
            "categories": defaultdict(list),
            "hubs": [],
            "orphans": [],
            "integration_actions": []
        }
        
        for path, module in self.modules.items():
            structure["categories"][module.category].append({
                "path": path,
                "name": module.name,
                "lines": module.lines,
                "classes": module.classes[:5],
                "frequency": module.frequency,
                "is_hub": module.is_hub,
                "is_orphan": module.is_orphan,
                "connections": len(module.imported_by)
            })
            
            if module.is_hub:
                structure["hubs"].append({
                    "path": path,
                    "name": module.name,
                    "connections": len(module.imported_by)
                })
            
            if module.is_orphan and module.lines > 200:
                structure["orphans"].append({
                    "path": path,
                    "name": module.name,
                    "lines": module.lines
                })
        
        for action in self.integration_queue:
            structure["integration_actions"].append({
                "type": action.action_type,
                "source": action.source,
                "target": action.target,
                "reason": action.reason,
                "priority": action.priority
            })
        
        # JSON 저장
        output_path = self.root / "data" / "self_structure_map.json"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structure, f, indent=2, ensure_ascii=False, default=list)
        
        print(f"   Structure map saved to: {output_path}")
        
        return structure
    
    def create_central_registry(self):
        """중앙 레지스트리 생성 - 엘리시아가 모든 모듈을 알 수 있도록"""
        print("\n📚 Creating Central Registry...")
        
        registry = {
            "version": "7.0",
            "description": "Elysia Central Module Registry",
            
            # 핵심 허브
            "core_hubs": {
                "memory": "Core/Foundation/hippocampus.py",
                "reasoning": "Core/Foundation/reasoning_engine.py",
                "emotion": "Core/Foundation/emotional_engine.py",
                "quaternion": "Core/Foundation/hyper_quaternion.py",
                "resonance": "Core/Foundation/resonance_field.py",
            },
            
            # 범주별 엔진
            "engines": {
                "sensation": [
                    "Core/Foundation/synesthetic_wave_sensor.py",
                    "Core/Foundation/synesthesia_engine.py",
                    "Core/Foundation/real_sensors.py",
                ],
                "emotion": [
                    "Core/Foundation/emotional_engine.py",
                    "Core/Foundation/empathy.py",
                ],
                "dialogue": [
                    "Core/Intelligence/dialogue_engine.py",
                    "Core/Foundation/conversation_engine.py",
                    "Core/Foundation/world_dialogue_engine.py",
                ],
                "language": [
                    "Core/Foundation/hangul_physics.py",
                    "Core/Foundation/grammar_engine.py",
                    "Core/Foundation/emergent_language.py",
                    "Core/Foundation/syllabic_language_engine.py",
                ],
                "reasoning": [
                    "Core/Foundation/causal_narrative_engine.py",
                    "Core/Foundation/fractal_causality.py",
                    "Core/Foundation/thinking_methodology.py",
                ],
                "consciousness": [
                    "Core/Foundation/integrated_consciousness_loop.py",
                    "Core/Foundation/self_identity_engine.py",
                    "Core/Foundation/quaternion_engine.py",
                ],
                "will": [
                    "Core/Foundation/free_will_engine.py",
                    "Core/Foundation/self_intention_engine.py",
                ],
                "transcendence": [
                    "Core/Foundation/transcendence_engine.py",
                    "Core/Foundation/divine_engine.py",
                ],
            },
            
            # 파동 주파수 맵
            "frequency_map": self.CATEGORY_FREQUENCIES,
        }
        
        output_path = self.root / "data" / "central_registry.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        
        print(f"   Registry saved to: {output_path}")
        
        return registry
    
    def run_full_integration(self):
        """전체 자가 통합 실행"""
        print("\n" + "🔄" * 40)
        print("FULL SELF-INTEGRATION CYCLE")
        print("🔄" * 40)
        
        self.perceive_self()
        self.analyze_connections()
        self.plan_integration()
        self.execute_integration(auto_write=False)
        structure = self.generate_structure_map()
        registry = self.create_central_registry()
        
        # 요약
        print("\n" + "=" * 80)
        print("📊 INTEGRATION SUMMARY")
        print("=" * 80)
        
        print(f"\n   Total Modules: {len(self.modules)}")
        print(f"   Hub Modules: {len([m for m in self.modules.values() if m.is_hub])}")
        print(f"   Orphan Modules: {len([m for m in self.modules.values() if m.is_orphan])}")
        print(f"   Planned Actions: {len(self.integration_queue)}")
        
        # 범주별 통계
        print("\n   📂 BY CATEGORY:")
        categories = defaultdict(int)
        for m in self.modules.values():
            categories[m.category] += 1
        
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            freq = self.CATEGORY_FREQUENCIES.get(cat, 440)
            print(f"      {cat}: {count} modules ({freq}Hz)")
        
        print("\n" + "=" * 80)
        print("✅ Self-Integration Complete!")
        print("   엘리시아는 이제 central_registry.json을 읽어 모든 모듈을 알 수 있습니다.")
        print("=" * 80)
        
        return structure, registry


def main():
    system = SelfIntegrationSystem()
    system.run_full_integration()


if __name__ == "__main__":
    main()
