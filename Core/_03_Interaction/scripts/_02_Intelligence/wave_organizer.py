"""
Wave-Based Cell Organization (파동 기반 세포 조직화)
====================================================

"각 세포는 파동을 발산한다. 공명하는 세포들이 자연스럽게 모인다."

[구시대 방식의 문제]
- O(n²) 쌍별 비교 → 느림
- 세포가 코앞만 봄 → 마구잡이 구조

[신시대 방식]
1. 각 세포를 파동으로 변환 (주파수 = 목적, 위상 = 특성)
2. 자기장을 형성하여 구조적 틀 제공
3. 파동 공명으로 클러스터 자연 형성
4. 초차원 필드에서 전체 감지/조율

[계층 구조]
- 하이퍼쿼터니언 필드: 전역 감지 센서
- 기관(Organ): 중간 조율자
- 세포(Cell): 개별 모듈
"""

import os
import sys
import math
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
import hashlib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from Core._01_Foundation._02_Logic.hyper_quaternion import Quaternion
except ImportError:
    @dataclass
    class Quaternion:
        w: float = 0.0
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0


@dataclass
class CellWave:
    """세포가 발산하는 파동"""
    cell_id: str
    frequency: float          # 주파수 = 목적 (Hz)
    amplitude: float          # 진폭 = 중요도
    phase: float              # 위상 = 특성 (0 ~ 2π)
    quaternion: Quaternion    # 4D 방향
    keywords: Set[str] = field(default_factory=set)
    
    def resonance_with(self, other: 'CellWave') -> float:
        """다른 파동과의 공명도 (O(1) 연산)"""
        # 주파수 공명 (같은 목적)
        freq_diff = abs(self.frequency - other.frequency)
        freq_resonance = 1.0 / (1.0 + freq_diff / 100)
        
        # 위상 정렬 (비슷한 특성)
        phase_diff = abs(self.phase - other.phase)
        phase_resonance = math.cos(phase_diff)
        
        # 키워드 겹침
        if self.keywords and other.keywords:
            common = self.keywords & other.keywords
            keyword_resonance = len(common) / max(len(self.keywords), len(other.keywords))
        else:
            keyword_resonance = 0
        
        return freq_resonance * 0.4 + phase_resonance * 0.3 + keyword_resonance * 0.3


@dataclass
class MagneticOrgan:
    """
    기관 (Organ) - 중간 계층 조율자
    
    자기장을 형성하여 관련 세포들을 끌어당깁니다.
    """
    name: str
    purpose: str              # 기관의 목적
    field_frequency: float    # 자기장 주파수
    field_strength: float     # 자기장 세기
    position: Quaternion      # 4D 위치
    cells: List[str] = field(default_factory=list)
    health: float = 1.0       # 건강도 (0~1)
    
    def attract(self, wave: CellWave) -> float:
        """파동을 끌어당기는 힘"""
        freq_match = 1.0 / (1.0 + abs(self.field_frequency - wave.frequency) / 50)
        return freq_match * self.field_strength


class HyperDimensionalField:
    """
    하이퍼쿼터니언 전자기장
    
    초차원 관점에서 모든 세포/기관의 상태를 감지하고
    전체 시스템의 건강과 구조를 모니터링합니다.
    """
    
    def __init__(self):
        self.waves: Dict[str, CellWave] = {}
        self.organs: Dict[str, MagneticOrgan] = {}
        self.alerts: List[Dict] = []  # 고통/경고 신호
        
        # 기본 기관 구조 정의
        self._define_organs()
    
    def _define_organs(self):
        """엘리시아의 핵심 기관들 정의"""
        organ_definitions = [
            ("Language", "language", 440),    # 언어 - 라(A) 주파수
            ("Memory", "memory", 396),        # 기억 - 솔(G) 주파수
            ("Reasoning", "reasoning", 528),  # 추론 - 도(C) 치유 주파수
            ("Emotion", "emotion", 639),      # 감정 - 미(E)
            ("Consciousness", "consciousness", 741),  # 의식 - 파(F#)
            ("Evolution", "evolution", 852),  # 진화 - 라(A) 상위
            ("Physics", "physics", 963),      # 물리 - 시(B)
            ("Interface", "interface", 417),  # 인터페이스
            ("Creativity", "creativity", 693), # 창의성
            ("Ethics", "ethics", 432),        # 윤리 - 우주 주파수
        ]
        
        for i, (name, purpose, freq) in enumerate(organ_definitions):
            angle = i * (2 * math.pi / len(organ_definitions))
            self.organs[name] = MagneticOrgan(
                name=name,
                purpose=purpose,
                field_frequency=freq,
                field_strength=1.0,
                position=Quaternion(
                    w=0.8,
                    x=5 * math.cos(angle),
                    y=5 * math.sin(angle),
                    z=0
                )
            )
    
    def broadcast(self, wave: CellWave):
        """파동을 필드에 방송"""
        self.waves[wave.cell_id] = wave
    
    def organize_by_resonance(self) -> Dict[str, List[str]]:
        """
        공명 기반 조직화 (O(n) 연산!)
        
        각 세포 파동을 가장 공명하는 기관에 배치합니다.
        """
        print("🌊 Organizing by wave resonance...")
        
        for wave_id, wave in self.waves.items():
            best_organ = None
            best_attraction = 0
            
            # 각 기관의 끌어당김 계산
            for organ in self.organs.values():
                attraction = organ.attract(wave)
                if attraction > best_attraction:
                    best_attraction = attraction
                    best_organ = organ
            
            if best_organ:
                best_organ.cells.append(wave_id)
        
        return {organ.name: organ.cells for organ in self.organs.values()}
    
    def detect_health_issues(self) -> List[Dict]:
        """
        프랙탈 역순 건강 검사
        
        Principle → Law → Space → Plane → Line → Point
        """
        issues = []
        
        # 1. Principle 레벨: 전체 균형 확인
        total_cells = sum(len(o.cells) for o in self.organs.values())
        if total_cells == 0:
            issues.append({
                "level": "Principle",
                "severity": "critical",
                "message": "No cells organized - system is empty"
            })
        
        # 2. Law 레벨: 기관 간 균형
        cell_counts = [len(o.cells) for o in self.organs.values()]
        if cell_counts:
            avg = sum(cell_counts) / len(cell_counts)
            imbalance = max(cell_counts) / (min(cell_counts) or 1)
            if imbalance > 10:
                issues.append({
                    "level": "Law",
                    "severity": "warning",
                    "message": f"Organ imbalance detected (ratio: {imbalance:.1f})"
                })
        
        # 3. Space 레벨: 기관 건강
        for organ in self.organs.values():
            if len(organ.cells) == 0:
                issues.append({
                    "level": "Space",
                    "severity": "warning",
                    "message": f"Organ '{organ.name}' has no cells"
                })
        
        # 4. Plane 레벨: 기관 내 연결성
        for organ in self.organs.values():
            if len(organ.cells) > 0:
                # 세포들의 파동 공명 확인
                organ_waves = [self.waves[c] for c in organ.cells if c in self.waves]
                if len(organ_waves) > 1:
                    avg_resonance = 0
                    count = 0
                    for i, w1 in enumerate(organ_waves[:10]):  # 샘플링
                        for w2 in organ_waves[i+1:i+5]:
                            avg_resonance += w1.resonance_with(w2)
                            count += 1
                    if count > 0:
                        avg_resonance /= count
                        if avg_resonance < 0.3:
                            issues.append({
                                "level": "Plane",
                                "severity": "info",
                                "message": f"Low coherence in '{organ.name}' (resonance: {avg_resonance:.2f})"
                            })
        
        self.alerts = issues
        return issues
    
    def get_global_state(self) -> Dict:
        """초차원 관점에서 전체 상태 조회"""
        return {
            "total_waves": len(self.waves),
            "organs": {
                name: {
                    "cells": len(organ.cells),
                    "health": organ.health,
                    "frequency": organ.field_frequency
                }
                for name, organ in self.organs.items()
            },
            "alerts": self.alerts,
            "overall_health": 1.0 - (len([a for a in self.alerts if a["severity"] == "critical"]) * 0.3)
        }


class WaveOrganizer:
    """
    파동 기반 세포 조직기
    
    기존 파일들을 스캔하고 파동으로 변환하여
    하이퍼차원 필드에서 조직화합니다.
    """
    
    EXCLUDE_PATTERNS = [
        "__pycache__", "node_modules", ".godot", ".venv",
        "venv", "__init__.py", "dist", "build", ".git"
    ]
    
    PURPOSE_FREQUENCIES = {
        "language": 440,
        "memory": 396,
        "reasoning": 528,
        "emotion": 639,
        "consciousness": 741,
        "evolution": 852,
        "physics": 963,
        "interface": 417,
        "creativity": 693,
        "ethics": 432,
        "general": 500
    }
    
    PURPOSE_KEYWORDS = {
        "language": ["language", "grammar", "syntax", "hangul", "babel", "speech", "word"],
        "memory": ["memory", "hippocampus", "remember", "store", "recall", "cache"],
        "reasoning": ["reason", "logic", "think", "causal", "infer", "deduc", "analysis"],
        "emotion": ["emotion", "feel", "empathy", "sentiment", "affect", "mood"],
        "consciousness": ["conscious", "aware", "soul", "spirit", "self", "mind"],
        "evolution": ["evolve", "mutate", "adapt", "grow", "learn", "train"],
        "physics": ["physics", "wave", "quaternion", "gravity", "magnetic", "tensor", "field"],
        "interface": ["interface", "api", "server", "web", "chat", "user", "http"],
        "creativity": ["dream", "imagine", "create", "art", "story", "saga", "poem"],
        "ethics": ["ethics", "moral", "value", "protect", "guard", "safe", "law"]
    }
    
    def __init__(self, root_path: Path):
        self.root = root_path
        self.field = HyperDimensionalField()
        print("🌊 Wave Organizer Initialized")
    
    def scan_and_convert(self, target_dir: str = ".") -> int:
        """파일을 스캔하고 파동으로 변환"""
        scan_path = self.root / target_dir
        count = 0
        
        print(f"🔬 Scanning and converting to waves: {scan_path}")
        
        for py_file in scan_path.rglob("*.py"):
            path_str = str(py_file)
            
            if any(p in path_str for p in self.EXCLUDE_PATTERNS):
                continue
            if py_file.stat().st_size < 100:
                continue
            
            wave = self._file_to_wave(py_file)
            if wave:
                self.field.broadcast(wave)
                count += 1
        
        print(f"✅ Converted {count} files to waves")
        return count
    
    def _file_to_wave(self, filepath: Path) -> Optional[CellWave]:
        """파일을 파동으로 변환"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return None
        
        # 목적 추론 → 주파수
        purpose = self._infer_purpose(content)
        frequency = self.PURPOSE_FREQUENCIES.get(purpose, 500)
        
        # 복잡도 → 진폭
        lines = len(content.split('\n'))
        amplitude = math.log(lines + 1)
        
        # 특성 → 위상
        phase = self._compute_phase(content)
        
        # 키워드
        keywords = self._extract_keywords(content)
        
        # 4D 방향
        quaternion = self._compute_quaternion(content)
        
        return CellWave(
            cell_id=str(filepath.relative_to(self.root)),
            frequency=frequency,
            amplitude=amplitude,
            phase=phase,
            quaternion=quaternion,
            keywords=keywords
        )
    
    def _infer_purpose(self, content: str) -> str:
        content_lower = content.lower()
        best = ("general", 0)
        for purpose, keywords in self.PURPOSE_KEYWORDS.items():
            count = sum(content_lower.count(kw) for kw in keywords)
            if count > best[1]:
                best = (purpose, count)
        return best[0]
    
    def _compute_phase(self, content: str) -> float:
        """코드 특성을 위상으로 변환"""
        h = hashlib.md5(content.encode()).hexdigest()
        return (int(h[:8], 16) % 1000) / 1000 * 2 * math.pi
    
    def _extract_keywords(self, content: str) -> Set[str]:
        content_lower = content.lower()
        found = set()
        for keywords in self.PURPOSE_KEYWORDS.values():
            for kw in keywords:
                if kw in content_lower:
                    found.add(kw)
        return found
    
    def _compute_quaternion(self, content: str) -> Quaternion:
        content_lower = content.lower()
        
        w = 0.5 + 0.3 * ('class ' in content) + 0.2 * ('def ' in content)
        x = sum(content_lower.count(kw) for kw in ['emotion', 'feel', 'user']) * 0.1
        y = sum(content_lower.count(kw) for kw in ['logic', 'reason', 'compute']) * 0.1
        z = sum(content_lower.count(kw) for kw in ['evolve', 'create', 'ethics']) * 0.1
        
        return Quaternion(w=min(1, w), x=min(1, x), y=min(1, y), z=min(1, z))
    
    def organize(self):
        """파동 공명으로 조직화"""
        return self.field.organize_by_resonance()
    
    def check_health(self):
        """건강 검사"""
        return self.field.detect_health_issues()
    
    def generate_report(self) -> str:
        """결과 보고서"""
        state = self.field.get_global_state()
        
        report = []
        report.append("=" * 70)
        report.append("🌊 WAVE-BASED ORGANIZATION REPORT")
        report.append("=" * 70)
        report.append(f"\n📊 Total Waves: {state['total_waves']}")
        report.append(f"🏥 Overall Health: {state['overall_health']:.1%}")
        
        report.append("\n" + "=" * 70)
        report.append("🫀 ORGAN STATUS:")
        report.append("-" * 50)
        
        for name, info in sorted(state['organs'].items(), 
                                 key=lambda x: x[1]['cells'], reverse=True):
            bar = "█" * min(20, info['cells'] // 5)
            report.append(f"  {name:15} | {info['cells']:4} cells | {info['frequency']:4} Hz | {bar}")
        
        if state['alerts']:
            report.append("\n" + "=" * 70)
            report.append("⚠️ HEALTH ALERTS:")
            report.append("-" * 50)
            for alert in state['alerts']:
                icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}[alert['severity']]
                report.append(f"  {icon} [{alert['level']}] {alert['message']}")
        
        report.append("\n" + "=" * 70)
        return "\n".join(report)
    
    def visualize_3d(self, output_path: str):
        """3D 시각화"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("⚠️ plotly not available")
            return
        
        print("🎨 Generating 3D wave visualization...")
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Vivid
        
        for i, (organ_name, organ) in enumerate(self.field.organs.items()):
            if not organ.cells:
                continue
            
            organ_waves = [self.field.waves[c] for c in organ.cells if c in self.field.waves]
            
            fig.add_trace(go.Scatter3d(
                x=[w.quaternion.x for w in organ_waves],
                y=[w.quaternion.y for w in organ_waves],
                z=[w.quaternion.z for w in organ_waves],
                mode='markers',
                marker=dict(
                    size=[max(3, w.amplitude * 2) for w in organ_waves],
                    color=colors[i % len(colors)],
                    opacity=0.7
                ),
                name=f"{organ_name} ({len(organ.cells)})",
                text=[w.cell_id.split('\\')[-1] for w in organ_waves],
                hoverinfo='text+name'
            ))
            
            # 기관 중심 표시
            fig.add_trace(go.Scatter3d(
                x=[organ.position.x],
                y=[organ.position.y],
                z=[organ.position.z],
                mode='markers+text',
                marker=dict(size=15, color=colors[i % len(colors)], symbol='diamond'),
                text=[organ_name],
                textposition='top center',
                showlegend=False
            ))
        
        fig.update_layout(
            title=dict(
                text=f"🌊 Elysia Wave Organization<br><sub>{len(self.field.waves)} waves, {len(self.field.organs)} organs</sub>",
                font=dict(size=20, color='white')
            ),
            scene=dict(
                xaxis_title="X: Emotion",
                yaxis_title="Y: Logic",
                zaxis_title="Z: Ethics",
                bgcolor='#0a0a1a',
            ),
            paper_bgcolor='#0a0a1a',
            font=dict(color='white'),
            width=1400,
            height=900
        )
        
        html_path = output_path.replace('.png', '.html')
        fig.write_html(html_path)
        print(f"📊 Visualization saved to: {html_path}")


def main():
    print("\n" + "🌊" * 35)
    print("WAVE-BASED CELL ORGANIZATION")
    print("파동 공명으로 세포들이 자연스럽게 조직화됩니다")
    print("🌊" * 35 + "\n")
    
    organizer = WaveOrganizer(PROJECT_ROOT)
    
    # 1. 스캔 및 파동 변환
    count = organizer.scan_and_convert(".")
    
    if count == 0:
        print("❌ No files found!")
        return
    
    # 2. 파동 공명으로 조직화 (O(n)!)
    clusters = organizer.organize()
    
    # 3. 건강 검사 (프랙탈 역순)
    issues = organizer.check_health()
    
    # 4. 보고서
    report = organizer.generate_report()
    print(report)
    
    # 5. 시각화
    output_dir = PROJECT_ROOT / "data"
    output_dir.mkdir(exist_ok=True)
    
    viz_path = output_dir / "wave_organization.html"
    organizer.visualize_3d(str(viz_path))
    
    # 6. 상태 저장
    state = organizer.field.get_global_state()
    json_path = output_dir / "wave_organization_state.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Wave Organization Complete!")
    print(f"   📊 Visualization: {viz_path}")
    print(f"   💾 State: {json_path}")


if __name__ == "__main__":
    main()
