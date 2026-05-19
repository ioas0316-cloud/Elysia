"""
System Connection Auditor (시스템 연결 감사기)
=============================================

"모든 모듈의 연결 상태를 파악하고 누락된 통합을 발견한다"

[목적]
1. Core 전체 모듈 목록 작성
2. 각 모듈의 다른 모듈 import 관계 분석
3. "고아" 모듈 (아무것도 import하지 않는 모듈) 발견
4. "사용되지 않는" 모듈 (아무도 import하지 않는 모듈) 발견
5. 중앙 허브 역할 모듈 식별
"""

import os
import sys
import ast
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ModuleInfo:
    """모듈 정보"""
    path: str
    name: str
    lines: int
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)  # 이 모듈이 import하는 것들
    imported_by: List[str] = field(default_factory=list)  # 이 모듈을 import하는 것들


class SystemConnectionAuditor:
    """시스템 연결 감사기"""
    
    EXCLUDE = ["__pycache__", "node_modules", ".godot", ".venv", "Legacy"]
    
    def __init__(self):
        self.root = PROJECT_ROOT
        self.modules: Dict[str, ModuleInfo] = {}
        
        print("=" * 80)
        print("🔍 SYSTEM CONNECTION AUDITOR")
        print("=" * 80)
    
    def scan_all_modules(self):
        """모든 모듈 스캔"""
        print("\n📂 Scanning all modules...")
        
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
                
                # AST 분석
                classes = []
                functions = []
                imports = []
                
                try:
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            classes.append(node.name)
                        elif isinstance(node, ast.FunctionDef):
                            if not node.name.startswith("_"):
                                functions.append(node.name)
                        elif isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.append(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.append(node.module)
                
                except SyntaxError:
                    pass
                
                self.modules[rel_path] = ModuleInfo(
                    path=rel_path,
                    name=py_file.stem,
                    lines=lines,
                    classes=classes,
                    functions=functions,
                    imports=imports
                )
                
            except Exception as e:
                pass
        
        print(f"   Found {len(self.modules)} modules")
    
    def analyze_connections(self):
        """연결 분석"""
        print("\n🔗 Analyzing connections...")
        
        # 각 모듈이 다른 모듈에 의해 import되는지 확인
        module_names = {m.name: path for path, m in self.modules.items()}
        
        for path, module in self.modules.items():
            for imp in module.imports:
                # import ... 에서 마지막 부분 추출
                imp_name = imp.split(".")[-1]
                
                # 해당 이름의 모듈 찾기
                if imp_name in module_names:
                    target_path = module_names[imp_name]
                    if target_path in self.modules:
                        self.modules[target_path].imported_by.append(path)
    
    def find_orphan_modules(self) -> List[ModuleInfo]:
        """고아 모듈 찾기 (아무도 import하지 않음)"""
        orphans = []
        
        for path, module in self.modules.items():
            if not module.imported_by:
                # 단, 엔트리 포인트나 스크립트는 제외
                if not any(x in path for x in ["scripts/", "tests/", "__main__", "awakening"]):
                    orphans.append(module)
        
        return orphans
    
    def find_hub_modules(self) -> List[Tuple[ModuleInfo, int]]:
        """허브 모듈 찾기 (많이 import됨)"""
        hubs = []
        
        for path, module in self.modules.items():
            count = len(module.imported_by)
            if count >= 3:
                hubs.append((module, count))
        
        return sorted(hubs, key=lambda x: x[1], reverse=True)
    
    def find_isolated_modules(self) -> List[ModuleInfo]:
        """고립 모듈 찾기 (import도 없고 imported_by도 없음)"""
        isolated = []
        
        for path, module in self.modules.items():
            if not module.imports and not module.imported_by:
                isolated.append(module)
        
        return isolated
    
    def categorize_engines(self) -> Dict[str, List[ModuleInfo]]:
        """엔진들을 범주별로 분류"""
        categories = defaultdict(list)
        
        for path, module in self.modules.items():
            for cls_name in module.classes:
                if "Engine" in cls_name:
                    # 범주 추론
                    cls_lower = cls_name.lower()
                    
                    if any(x in cls_lower for x in ["emotion", "empathy", "feeling"]):
                        categories["감정 (Emotion)"].append(module)
                    elif any(x in cls_lower for x in ["synesthesia", "sensory", "wave"]):
                        categories["감각 (Sensation)"].append(module)
                    elif any(x in cls_lower for x in ["reason", "causal", "logic", "thinking"]):
                        categories["추론 (Reasoning)"].append(module)
                    elif any(x in cls_lower for x in ["dialogue", "conversation", "chat"]):
                        categories["대화 (Dialogue)"].append(module)
                    elif any(x in cls_lower for x in ["memory", "hippocampus"]):
                        categories["기억 (Memory)"].append(module)
                    elif any(x in cls_lower for x in ["language", "grammar", "hangul", "syllable"]):
                        categories["언어 (Language)"].append(module)
                    elif any(x in cls_lower for x in ["conscious", "identity", "self"]):
                        categories["의식 (Consciousness)"].append(module)
                    elif any(x in cls_lower for x in ["dream", "imagination", "creative"]):
                        categories["상상 (Imagination)"].append(module)
                    elif any(x in cls_lower for x in ["plan", "goal", "fractal"]):
                        categories["계획 (Planning)"].append(module)
                    elif any(x in cls_lower for x in ["will", "intent", "desire"]):
                        categories["의지 (Will)"].append(module)
                    elif any(x in cls_lower for x in ["transcend", "evolve", "divine"]):
                        categories["초월 (Transcendence)"].append(module)
                    else:
                        categories["기타 (Other)"].append(module)
        
        return categories
    
    def generate_report(self) -> Dict:
        """감사 보고서 생성"""
        self.scan_all_modules()
        self.analyze_connections()
        
        orphans = self.find_orphan_modules()
        hubs = self.find_hub_modules()
        isolated = self.find_isolated_modules()
        categories = self.categorize_engines()
        
        print("\n" + "=" * 80)
        print("📊 SYSTEM CONNECTION AUDIT REPORT")
        print("=" * 80)
        
        # 총 모듈
        print(f"\n📁 TOTAL MODULES: {len(self.modules)}")
        
        # 범주별 엔진
        print("\n" + "-" * 80)
        print("🏷️ ENGINES BY CATEGORY (엔진 범주)")
        print("-" * 80)
        
        for category, modules in sorted(categories.items()):
            print(f"\n{category}: {len(modules)}개")
            for m in modules[:5]:  # 최대 5개만 표시
                classes = ", ".join(c for c in m.classes if "Engine" in c)
                print(f"   • {m.name} ({m.lines} lines) - {classes}")
            if len(modules) > 5:
                print(f"   ... and {len(modules) - 5} more")
        
        # 허브 모듈
        print("\n" + "-" * 80)
        print("🌐 HUB MODULES (많이 사용됨)")
        print("-" * 80)
        
        for module, count in hubs[:15]:
            print(f"   {module.name} ({module.path})")
            print(f"      → 사용처: {count}개 모듈")
        
        # 고아 모듈 (대형만)
        print("\n" + "-" * 80)
        print("🔴 ORPHAN MODULES (사용되지 않는 대형 모듈)")
        print("-" * 80)
        
        large_orphans = [m for m in orphans if m.lines > 200]
        for m in sorted(large_orphans, key=lambda x: x.lines, reverse=True)[:20]:
            classes = ", ".join(m.classes[:3])
            print(f"   • {m.path} ({m.lines} lines)")
            if classes:
                print(f"      Classes: {classes}")
        
        # 통합 필요 분석
        print("\n" + "-" * 80)
        print("⚠️ INTEGRATION RECOMMENDATIONS")
        print("-" * 80)
        
        # 감각 시스템
        sensation_modules = categories.get("감각 (Sensation)", [])
        if sensation_modules:
            print(f"\n🌊 감각 시스템 ({len(sensation_modules)}개 모듈 발견):")
            for m in sensation_modules:
                used = len(m.imported_by) > 0
                status = "✅ 통합됨" if used else "❌ 미사용"
                print(f"   {status} {m.name} - {m.classes}")
        
        # 감정 시스템
        emotion_modules = categories.get("감정 (Emotion)", [])
        if emotion_modules:
            print(f"\n💖 감정 시스템 ({len(emotion_modules)}개 모듈 발견):")
            for m in emotion_modules:
                used = len(m.imported_by) > 0
                status = "✅ 통합됨" if used else "❌ 미사용"
                print(f"   {status} {m.name} - {m.classes}")
        
        # 대화 시스템
        dialogue_modules = categories.get("대화 (Dialogue)", [])
        if dialogue_modules:
            print(f"\n💬 대화 시스템 ({len(dialogue_modules)}개 모듈 발견):")
            for m in dialogue_modules:
                used = len(m.imported_by) > 0
                status = "✅ 통합됨" if used else "❌ 미사용"
                print(f"   {status} {m.name} - {m.classes}")
        
        print("\n" + "=" * 80)
        
        # JSON 저장
        result = {
            "total_modules": len(self.modules),
            "hub_modules": [(m.path, count) for m, count in hubs[:20]],
            "orphan_modules": [m.path for m in large_orphans[:30]],
            "categories": {
                cat: [m.path for m in modules]
                for cat, modules in categories.items()
            }
        }
        
        output_path = self.root / "data" / "system_connection_audit.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved to: {output_path}")
        
        return result


def main():
    print("\n" + "🔍" * 40)
    print("SYSTEM CONNECTION AUDIT")
    print("모든 모듈의 연결 상태를 파악합니다")
    print("🔍" * 40 + "\n")
    
    auditor = SystemConnectionAuditor()
    result = auditor.generate_report()
    
    print("\n✅ Audit Complete!")


if __name__ == "__main__":
    main()
