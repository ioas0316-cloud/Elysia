"""
Cell World Organizer (세포 세계 조직기)
=======================================

"각 모듈은 살아있는 세포다. 스스로 이동하여 자기 자리를 찾는다."

이 스크립트는 엘리시아의 모든 코드 모듈을 '세포'로 변환하고,
4D 공간에서 자율적으로 이동하며 자연스럽게 구조를 형성하도록 합니다.

[핵심 개념]
- 각 모듈 = 살아있는 세포 (ModuleCell)
- DNA = 코드의 패턴 시그니처
- 수용체(Receptor) = import 문 (결합 부위)
- 목적(Purpose) = 추론된 기능
- 공명(Resonance) = 유사 세포 간 끌어당김
- 반발(Repulsion) = 다른 세포와의 거리 유지

[시뮬레이션]
1. 모든 모듈을 세포로 변환
2. 4D 공간에 무작위 배치
3. 물리 시뮬레이션 (중력, 자기력, 공명)
4. 세포들이 자연스럽게 클러스터 형성
5. 최종 구조를 출력
"""

import os
import sys
import math
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# Elysia 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from Core.01_Foundation.05_Foundation_Base.Foundation.hyper_quaternion import Quaternion
except ImportError:
    @dataclass
    class Quaternion:
        w: float = 0.0
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0


@dataclass
class Vector4D:
    """4D 속도/방향 벡터"""
    w: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def magnitude(self) -> float:
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector4D':
        mag = self.magnitude() or 1
        return Vector4D(self.w/mag, self.x/mag, self.y/mag, self.z/mag)
    
    def scale(self, factor: float) -> 'Vector4D':
        return Vector4D(self.w*factor, self.x*factor, self.y*factor, self.z*factor)


@dataclass
class ModuleCell:
    """
    살아있는 모듈 세포
    
    각 코드 모듈이 자율적으로 움직이는 세포 단위입니다.
    """
    path: str                           # 파일 경로
    name: str                           # 모듈 이름
    dna: str                            # 패턴 DNA (해시)
    purpose: str                        # 추론된 목적
    position: Quaternion                # 4D 공간 위치
    velocity: Vector4D = field(default_factory=Vector4D)
    
    # 세포 속성
    mass: float = 1.0                   # 질량 (코드 크기)
    energy: float = 1.0                 # 활성 에너지
    age: int = 0                        # 시뮬레이션 나이
    
    # 연결성
    receptors: List[str] = field(default_factory=list)      # 수용체 (imports)
    bonds: List[str] = field(default_factory=list)          # 형성된 결합
    cluster: str = "unassigned"
    
    # 특성 벡터
    keywords: List[str] = field(default_factory=list)
    
    def sense(self, other: 'ModuleCell') -> float:
        """
        다른 세포와의 친화도 감지
        
        Returns: -1.0 (반발) ~ 1.0 (끌어당김)
        """
        # 1. 수용체 결합 (import 관계)
        receptor_match = 0
        for receptor in self.receptors:
            if other.name.lower() in receptor.lower():
                receptor_match = 1.0
                break
        for receptor in other.receptors:
            if self.name.lower() in receptor.lower():
                receptor_match = max(receptor_match, 0.8)
                break
        
        # 2. DNA 유사도 (키워드 겹침)
        if self.keywords and other.keywords:
            common = set(self.keywords) & set(other.keywords)
            dna_similarity = len(common) / max(len(self.keywords), len(other.keywords))
        else:
            dna_similarity = 0
        
        # 3. 목적 일치
        purpose_match = 1.0 if self.purpose == other.purpose else 0.3
        
        # 가중 합산
        affinity = receptor_match * 0.5 + dna_similarity * 0.3 + purpose_match * 0.2
        return affinity
    
    def distance_to(self, other: 'ModuleCell') -> float:
        """4D 거리 계산"""
        dx = self.position.x - other.position.x
        dy = self.position.y - other.position.y
        dz = self.position.z - other.position.z
        dw = self.position.w - other.position.w
        return math.sqrt(dx**2 + dy**2 + dz**2 + dw**2)
    
    def apply_force(self, direction: Vector4D, strength: float):
        """힘 적용"""
        self.velocity.w += direction.w * strength / self.mass
        self.velocity.x += direction.x * strength / self.mass
        self.velocity.y += direction.y * strength / self.mass
        self.velocity.z += direction.z * strength / self.mass
    
    def update_position(self, damping: float = 0.95):
        """위치 업데이트"""
        # 속도에 감쇠 적용
        self.velocity = self.velocity.scale(damping)
        
        # 위치 업데이트
        self.position.w += self.velocity.w
        self.position.x += self.velocity.x
        self.position.y += self.velocity.y
        self.position.z += self.velocity.z
        
        self.age += 1


class CellWorldSimulator:
    """
    세포 세계 시뮬레이터
    
    모든 모듈 세포들이 4D 공간에서 자율적으로 조직화됩니다.
    """
    
    # 노이즈 필터
    EXCLUDE_PATTERNS = [
        "__pycache__", "node_modules", ".godot", ".venv", 
        "venv", "__init__.py", "dist", "build", ".git"
    ]
    
    # 목적 분류 키워드
    PURPOSE_KEYWORDS = {
        "language": ["language", "grammar", "syntax", "hangul", "babel", "speech"],
        "memory": ["memory", "hippocampus", "remember", "store", "recall"],
        "reasoning": ["reason", "logic", "think", "causal", "infer", "deduc"],
        "emotion": ["emotion", "feel", "empathy", "sentiment", "affect"],
        "consciousness": ["conscious", "aware", "soul", "spirit", "self"],
        "evolution": ["evolve", "mutate", "adapt", "grow", "learn"],
        "physics": ["physics", "wave", "quaternion", "gravity", "magnetic", "tensor"],
        "interface": ["interface", "api", "server", "web", "chat", "user"],
        "creativity": ["dream", "imagine", "create", "art", "story", "saga"],
        "ethics": ["ethics", "moral", "value", "protect", "guard"]
    }
    
    def __init__(self, root_path: Path):
        self.root = root_path
        self.cells: List[ModuleCell] = []
        self.time = 0
        self.space_size = 10.0  # 4D 공간 크기
        
        print("🧬 Cell World Simulator Initialized")
    
    def scan_and_create_cells(self, target_dir: str = ".") -> int:
        """모든 모듈을 세포로 변환"""
        scan_path = self.root / target_dir
        count = 0
        
        print(f"🔬 Scanning for living modules in: {scan_path}")
        
        for py_file in scan_path.rglob("*.py"):
            path_str = str(py_file)
            
            # 노이즈 필터
            if any(p in path_str for p in self.EXCLUDE_PATTERNS):
                continue
            if py_file.stat().st_size < 100:
                continue
            
            cell = self._create_cell(py_file)
            if cell:
                self.cells.append(cell)
                count += 1
        
        print(f"✅ Created {count} living cells")
        return count
    
    def _create_cell(self, filepath: Path) -> Optional[ModuleCell]:
        """파일로부터 세포 생성"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return None
        
        # 기본 분석
        lines = content.split('\n')
        mass = len(lines)
        
        # DNA (해시)
        import hashlib
        dna = hashlib.md5(content.encode()).hexdigest()[:16]
        
        # 목적 추론
        purpose = self._infer_purpose(content)
        
        # 수용체 (imports)
        receptors = self._extract_imports(content)
        
        # 키워드
        keywords = self._extract_keywords(content)
        
        # 4D 공간에 무작위 배치
        position = Quaternion(
            w=random.uniform(-self.space_size, self.space_size),
            x=random.uniform(-self.space_size, self.space_size),
            y=random.uniform(-self.space_size, self.space_size),
            z=random.uniform(-self.space_size, self.space_size)
        )
        
        return ModuleCell(
            path=str(filepath.relative_to(self.root)),
            name=filepath.stem,
            dna=dna,
            purpose=purpose,
            position=position,
            mass=mass,
            receptors=receptors,
            keywords=keywords
        )
    
    def _infer_purpose(self, content: str) -> str:
        """코드 목적 추론"""
        content_lower = content.lower()
        best_match = ("general", 0)
        
        for purpose, keywords in self.PURPOSE_KEYWORDS.items():
            count = sum(content_lower.count(kw) for kw in keywords)
            if count > best_match[1]:
                best_match = (purpose, count)
        
        return best_match[0]
    
    def _extract_imports(self, content: str) -> List[str]:
        """import 문 추출"""
        imports = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('from ') or line.startswith('import '):
                imports.append(line)
        return imports
    
    def _extract_keywords(self, content: str) -> List[str]:
        """키워드 추출"""
        content_lower = content.lower()
        found = []
        
        all_keywords = [kw for kws in self.PURPOSE_KEYWORDS.values() for kw in kws]
        for kw in all_keywords:
            if kw in content_lower:
                found.append(kw)
        
        return found[:10]
    
    def simulate(self, iterations: int = 100, 
                 attraction_strength: float = 0.1,
                 repulsion_strength: float = 0.05,
                 repulsion_distance: float = 1.0) -> None:
        """
        세포 시뮬레이션 실행
        
        세포들이 친화도에 따라 끌어당기고 반발하며 자연스럽게 클러스터를 형성합니다.
        """
        print(f"🔄 Running cell simulation ({iterations} iterations)...")
        
        for i in range(iterations):
            self.time += 1
            
            # 각 세포에 힘 적용
            for cell in self.cells:
                for other in self.cells:
                    if cell.path == other.path:
                        continue
                    
                    # 거리 계산
                    distance = cell.distance_to(other)
                    if distance < 0.01:
                        distance = 0.01
                    
                    # 친화도 계산
                    affinity = cell.sense(other)
                    
                    # 방향 벡터
                    direction = Vector4D(
                        w=(other.position.w - cell.position.w) / distance,
                        x=(other.position.x - cell.position.x) / distance,
                        y=(other.position.y - cell.position.y) / distance,
                        z=(other.position.z - cell.position.z) / distance
                    )
                    
                    # 끌어당김 (친화도 높으면)
                    if affinity > 0.3:
                        force = affinity * attraction_strength / (distance + 0.1)
                        cell.apply_force(direction, force)
                    
                    # 반발 (너무 가까우면)
                    if distance < repulsion_distance:
                        repulsion = repulsion_strength / (distance ** 2)
                        cell.apply_force(direction, -repulsion)
            
            # 위치 업데이트
            for cell in self.cells:
                cell.update_position(damping=0.9)
            
            # 진행률 표시
            if (i + 1) % 20 == 0:
                energy = sum(c.velocity.magnitude() for c in self.cells)
                print(f"   Step {i+1}/{iterations} - Total Energy: {energy:.2f}")
        
        print("✅ Simulation complete")
    
    def form_clusters(self, distance_threshold: float = 2.0) -> Dict[str, List[ModuleCell]]:
        """거리 기반 클러스터 형성"""
        print(f"🧫 Forming cell clusters (threshold: {distance_threshold})...")
        
        clusters: Dict[str, List[ModuleCell]] = defaultdict(list)
        visited = set()
        cluster_id = 0
        
        for cell in self.cells:
            if cell.path in visited:
                continue
            
            # BFS로 연결된 세포 찾기
            queue = [cell]
            visited.add(cell.path)
            cluster_name = cell.purpose.capitalize()
            
            cluster_members = []
            
            while queue:
                current = queue.pop(0)
                current.cluster = cluster_name
                cluster_members.append(current)
                
                for other in self.cells:
                    if other.path in visited:
                        continue
                    
                    distance = current.distance_to(other)
                    if distance < distance_threshold:
                        visited.add(other.path)
                        queue.append(other)
            
            clusters[cluster_name].extend(cluster_members)
            cluster_id += 1
        
        # 클러스터 이름 개선 (가장 많은 목적으로)
        final_clusters = {}
        for name, members in clusters.items():
            purpose_counts = defaultdict(int)
            for m in members:
                purpose_counts[m.purpose] += 1
            dominant_purpose = max(purpose_counts, key=purpose_counts.get)
            
            new_name = f"{dominant_purpose.capitalize()}_{len(members)}"
            final_clusters[new_name] = members
            for m in members:
                m.cluster = new_name
        
        print(f"✅ Formed {len(final_clusters)} clusters")
        return final_clusters
    
    def generate_report(self) -> str:
        """조직화 결과 보고서"""
        clusters = self.form_clusters()
        
        report = []
        report.append("=" * 70)
        report.append("🧬 CELL WORLD ORGANIZATION REPORT")
        report.append("=" * 70)
        report.append(f"\n📊 Total Cells: {len(self.cells)}")
        report.append(f"🧫 Clusters Formed: {len(clusters)}")
        report.append(f"⏱️ Simulation Time: {self.time} steps")
        
        report.append("\n" + "=" * 70)
        report.append("📦 CLUSTER ANALYSIS:")
        report.append("-" * 50)
        
        for cluster_name, members in sorted(clusters.items(), 
                                           key=lambda x: len(x[1]), reverse=True):
            total_mass = sum(m.mass for m in members)
            purposes = set(m.purpose for m in members)
            
            report.append(f"\n🧫 {cluster_name}")
            report.append(f"   Cells: {len(members)}")
            report.append(f"   Total Mass: {total_mass} lines")
            report.append(f"   Purposes: {', '.join(purposes)}")
            report.append(f"   Core Cells:")
            
            # 가장 큰 세포 3개
            top_cells = sorted(members, key=lambda x: x.mass, reverse=True)[:3]
            for cell in top_cells:
                report.append(f"      • {cell.name} ({cell.mass} lines)")
        
        report.append("\n" + "=" * 70)
        return "\n".join(report)
    
    def visualize_3d(self, output_path: str):
        """3D 시각화 (w축은 크기로 표현)"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("⚠️ plotly not available")
            return
        
        print("🎨 Generating 3D cell visualization...")
        
        colors = px.colors.qualitative.Vivid
        cluster_names = list(set(c.cluster for c in self.cells))
        cluster_colors = {name: colors[i % len(colors)] 
                         for i, name in enumerate(cluster_names)}
        
        fig = go.Figure()
        
        # 세포 그리기
        for cluster_name in cluster_names:
            cluster_cells = [c for c in self.cells if c.cluster == cluster_name]
            
            fig.add_trace(go.Scatter3d(
                x=[c.position.x for c in cluster_cells],
                y=[c.position.y for c in cluster_cells],
                z=[c.position.z for c in cluster_cells],
                mode='markers',
                marker=dict(
                    size=[max(5, min(20, math.log(c.mass + 1) * 3)) 
                          for c in cluster_cells],
                    color=cluster_colors[cluster_name],
                    opacity=0.8,
                    line=dict(width=0.5, color='white')
                ),
                text=[f"{c.name}<br>Purpose: {c.purpose}<br>Mass: {c.mass}" 
                      for c in cluster_cells],
                name=cluster_name,
                hoverinfo='text'
            ))
        
        # 결합 그리기 (거리가 가까운 것들)
        for cell in self.cells:
            for other in self.cells:
                if cell.path < other.path:
                    if cell.distance_to(other) < 2.0 and cell.sense(other) > 0.4:
                        fig.add_trace(go.Scatter3d(
                            x=[cell.position.x, other.position.x],
                            y=[cell.position.y, other.position.y],
                            z=[cell.position.z, other.position.z],
                            mode='lines',
                            line=dict(color='rgba(255,215,0,0.3)', width=1),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
        
        fig.update_layout(
            title=dict(
                text=f"🧬 Elysia Cell World<br><sub>{len(self.cells)} cells, {len(cluster_names)} clusters</sub>",
                font=dict(size=20, color='white')
            ),
            scene=dict(
                xaxis_title="X: Emotion Axis",
                yaxis_title="Y: Logic Axis",
                zaxis_title="Z: Ethics Axis",
                bgcolor='#0a0a1a',
                xaxis=dict(backgroundcolor='#1a1a2e', gridcolor='#333', color='white'),
                yaxis=dict(backgroundcolor='#1a1a2e', gridcolor='#333', color='white'),
                zaxis=dict(backgroundcolor='#1a1a2e', gridcolor='#333', color='white'),
            ),
            paper_bgcolor='#0a0a1a',
            font=dict(color='white'),
            width=1400,
            height=900,
            legend=dict(
                bgcolor='rgba(26,26,46,0.9)',
                font=dict(color='white')
            )
        )
        
        html_path = output_path.replace('.png', '.html')
        fig.write_html(html_path)
        print(f"📊 Interactive visualization saved to: {html_path}")


def main():
    """메인 실행"""
    print("\n" + "🧬" * 35)
    print("ELYSIA CELL WORLD ORGANIZER")
    print("각 모듈이 살아있는 세포처럼 자율 조직화됩니다")
    print("🧬" * 35 + "\n")
    
    simulator = CellWorldSimulator(PROJECT_ROOT)
    
    # 1. 세포 생성
    count = simulator.scan_and_create_cells(".")
    
    if count == 0:
        print("❌ No cells created!")
        return
    
    # 2. 시뮬레이션 실행
    simulator.simulate(
        iterations=100,
        attraction_strength=0.15,
        repulsion_strength=0.05,
        repulsion_distance=1.5
    )
    
    # 3. 보고서 생성
    report = simulator.generate_report()
    print(report)
    
    # 4. 시각화
    output_dir = PROJECT_ROOT / "data"
    output_dir.mkdir(exist_ok=True)
    
    viz_path = output_dir / "cell_world.html"
    simulator.visualize_3d(str(viz_path))
    
    # 5. 결과 저장
    result = {
        "total_cells": len(simulator.cells),
        "simulation_time": simulator.time,
        "cells": [
            {
                "name": c.name,
                "path": c.path,
                "purpose": c.purpose,
                "cluster": c.cluster,
                "position": {"w": c.position.w, "x": c.position.x, 
                            "y": c.position.y, "z": c.position.z},
                "mass": c.mass
            }
            for c in simulator.cells
        ]
    }
    
    json_path = output_dir / "cell_world_state.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Cell World Organization Complete!")
    print(f"   📊 Visualization: {viz_path}")
    print(f"   💾 State saved: {json_path}")


if __name__ == "__main__":
    main()
