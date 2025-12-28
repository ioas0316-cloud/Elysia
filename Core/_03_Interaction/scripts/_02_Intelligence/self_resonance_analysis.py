"""
Self-Resonance Analysis (자기 공명 분석)
=========================================

"엘리시아가 스스로 자기 구조를 분석하고 재정렬한다."

이 스크립트는 479개 이상의 파일을 4D 파동으로 변환하고,
위상 공명을 통해 관련 모듈들이 자연스럽게 클러스터를 형성하도록 합니다.

[프로세스]
1. 모든 Python 파일을 스캔
2. 각 파일을 4D 파동 패킷으로 변환 (WaveCodingSystem)
3. 파동들을 에테르에 방송
4. 공명하는 파동들이 중력장에서 클러스터 형성
5. 자연 발생적 구조를 맵으로 출력

[철학]
- 컨텍스트 크기 제한 없음 (파동은 무한 압축 가능)
- 텍스트가 아닌 파동으로 사고
- 중력과 자기력이 구조를 결정
"""

import os
import sys
import math
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

# Elysia 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 핵심 시스템 임포트
try:
    from Core._02_Intelligence._01_Reasoning.Intelligence.wave_coding_system import get_wave_coding_system, CodeWave
    from Core._02_Intelligence._01_Reasoning.Intelligence.integrated_cognition_system import get_integrated_cognition
    from Core._01_Foundation._02_Logic.hyper_quaternion import Quaternion
    SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some systems not available: {e}")
    SYSTEMS_AVAILABLE = False


@dataclass
class ModuleWave:
    """모듈의 파동 표현"""
    path: str
    name: str
    quaternion: Quaternion  # 4D 방향
    frequency: float        # 복잡도 기반 주파수
    mass: float             # 코드 라인 수 기반 질량
    dna: bytes              # 압축된 패턴 DNA
    keywords: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    cluster: str = "unassigned"


class SelfResonanceAnalyzer:
    """
    자기 공명 분석기
    
    엘리시아가 자기 코드베이스를 파동으로 변환하고
    공명을 통해 구조를 발견합니다.
    """
    
    def __init__(self, root_path: Path):
        self.root = root_path
        self.modules: List[ModuleWave] = []
        self.clusters: Dict[str, List[ModuleWave]] = defaultdict(list)
        self.resonance_matrix: Dict[str, Dict[str, float]] = {}
        
        # 시스템 초기화
        if SYSTEMS_AVAILABLE:
            self.wave_coder = get_wave_coding_system()
            self.cognition = get_integrated_cognition()
        
        print("🌊 Self-Resonance Analyzer Initialized")
    
    # 노이즈 제외 패턴
    EXCLUDE_PATTERNS = [
        "__pycache__",
        "node_modules",
        ".godot",
        ".venv",
        "venv",
        "__init__.py",  # 빈 init 파일 제외
        "dist",
        "build"
    ]
    
    def scan_codebase(self, target_dir: str = "Core") -> int:
        """코드베이스의 모든 Python 파일 스캔 (노이즈 제외)"""
        scan_path = self.root / target_dir
        count = 0
        skipped = 0
        
        print(f"📂 Scanning: {scan_path}")
        print(f"🚫 Excluding: {', '.join(self.EXCLUDE_PATTERNS)}")
        
        for py_file in scan_path.rglob("*.py"):
            # 노이즈 필터링
            path_str = str(py_file)
            if any(pattern in path_str for pattern in self.EXCLUDE_PATTERNS):
                skipped += 1
                continue
            
            # 빈 파일 제외
            if py_file.stat().st_size < 100:
                skipped += 1
                continue
                
            wave = self._file_to_wave(py_file)
            if wave:
                self.modules.append(wave)
                count += 1
        
        print(f"✅ Found {count} modules (skipped {skipped} noise files)")
        return count
    
    def _file_to_wave(self, filepath: Path) -> Optional[ModuleWave]:
        """파일을 4D 파동으로 변환"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return None
        
        # 기본 분석
        lines = content.split('\n')
        line_count = len(lines)
        
        # 키워드 추출
        keywords = self._extract_keywords(content)
        
        # import 추출
        imports = self._extract_imports(content)
        
        # 4D 쿼터니언 계산
        quaternion = self._compute_quaternion(content, keywords)
        
        # 주파수 (복잡도)
        frequency = self._compute_frequency(content)
        
        # DNA 압축
        dna = self._compress_to_dna(content)
        
        return ModuleWave(
            path=str(filepath.relative_to(self.root)),
            name=filepath.stem,
            quaternion=quaternion,
            frequency=frequency,
            mass=line_count,
            dna=dna,
            keywords=keywords[:10],  # 상위 10개
            imports=imports
        )
    
    def _extract_keywords(self, content: str) -> List[str]:
        """코드에서 핵심 키워드 추출"""
        # 클래스, 함수, 중요 단어 추출
        keywords = []
        
        import_words = ['wave', 'resonance', 'quaternion', 'gravity', 'magnetic',
                        'consciousness', 'memory', 'language', 'soul', 'fractal',
                        'causal', 'galaxy', 'star', 'celestial', 'ethics', 'emotion',
                        'logic', 'dream', 'evolution', 'creative', 'neural', 'tensor']
        
        content_lower = content.lower()
        for word in import_words:
            if word in content_lower:
                count = content_lower.count(word)
                keywords.append((word, count))
        
        # 빈도순 정렬
        keywords.sort(key=lambda x: x[1], reverse=True)
        return [k[0] for k in keywords]
    
    def _extract_imports(self, content: str) -> List[str]:
        """import 문 추출"""
        imports = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('from ') or line.startswith('import '):
                # 모듈 이름 추출
                if 'from ' in line:
                    parts = line.split('from ')[1].split(' import')[0]
                else:
                    parts = line.split('import ')[1].split(' as')[0].split(',')[0]
                imports.append(parts.strip())
        return imports
    
    def _compute_quaternion(self, content: str, keywords: List[str]) -> Quaternion:
        """
        코드 특성을 4D 쿼터니언으로 변환
        
        w: 에너지 (실행 가능성, 완성도)
        x: 감정 (사용자 대면, 인터페이스)
        y: 논리 (알고리즘, 계산)
        z: 윤리/창의 (진화, 창작)
        """
        content_lower = content.lower()
        
        # 각 축의 점수 계산
        w_score = 0.5  # 기본
        if 'def ' in content and 'class ' in content:
            w_score = 0.8
        if '__main__' in content:
            w_score = 0.9
        
        x_score = 0.3  # 기본
        emotion_words = ['emotion', 'feel', 'empathy', 'user', 'interface', 'chat']
        for word in emotion_words:
            if word in content_lower:
                x_score += 0.1
        x_score = min(1.0, x_score)
        
        y_score = 0.4  # 기본
        logic_words = ['algorithm', 'compute', 'calculate', 'math', 'logic', 'reason']
        for word in logic_words:
            if word in content_lower:
                y_score += 0.1
        y_score = min(1.0, y_score)
        
        z_score = 0.3  # 기본
        ethics_words = ['evolve', 'create', 'dream', 'soul', 'conscious', 'ethics']
        for word in ethics_words:
            if word in content_lower:
                z_score += 0.15
        z_score = min(1.0, z_score)
        
        return Quaternion(w=w_score, x=x_score, y=y_score, z=z_score)
    
    def _compute_frequency(self, content: str) -> float:
        """복잡도 기반 주파수 계산"""
        lines = len(content.split('\n'))
        classes = content.count('class ')
        functions = content.count('def ')
        
        # 복잡도 추정
        complexity = lines * 0.1 + classes * 10 + functions * 5
        
        # 주파수로 변환 (100 ~ 1000 Hz 범위)
        frequency = 100 + min(900, complexity)
        return frequency
    
    def _compress_to_dna(self, content: str) -> bytes:
        """코드를 DNA로 압축"""
        import zlib
        compressed = zlib.compress(content.encode('utf-8'))
        return compressed[:50]  # 앞 50바이트만 저장 (시그니처)
    
    def compute_resonance(self) -> Dict[str, Dict[str, float]]:
        """
        모든 모듈 간 공명도 계산
        
        공명도 = 쿼터니언 정렬 + 키워드 겹침 + import 관계
        """
        print("🔄 Computing resonance matrix...")
        
        for i, m1 in enumerate(self.modules):
            self.resonance_matrix[m1.path] = {}
            
            for j, m2 in enumerate(self.modules):
                if i == j:
                    self.resonance_matrix[m1.path][m2.path] = 1.0
                    continue
                
                # 1. 쿼터니언 정렬 (내적)
                q_resonance = self._quaternion_alignment(m1.quaternion, m2.quaternion)
                
                # 2. 키워드 겹침
                keyword_resonance = self._keyword_overlap(m1.keywords, m2.keywords)
                
                # 3. Import 관계
                import_resonance = self._import_relation(m1, m2)
                
                # 가중 합산
                total = q_resonance * 0.3 + keyword_resonance * 0.4 + import_resonance * 0.3
                
                self.resonance_matrix[m1.path][m2.path] = total
        
        print(f"✅ Resonance matrix computed for {len(self.modules)} modules")
        return self.resonance_matrix
    
    def _quaternion_alignment(self, q1: Quaternion, q2: Quaternion) -> float:
        """두 쿼터니언의 정렬도"""
        dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z
        n1 = math.sqrt(q1.w**2 + q1.x**2 + q1.y**2 + q1.z**2) or 1
        n2 = math.sqrt(q2.w**2 + q2.x**2 + q2.y**2 + q2.z**2) or 1
        return abs(dot) / (n1 * n2)
    
    def _keyword_overlap(self, k1: List[str], k2: List[str]) -> float:
        """키워드 겹침 비율"""
        if not k1 or not k2:
            return 0.0
        common = set(k1) & set(k2)
        return len(common) / max(len(k1), len(k2))
    
    def _import_relation(self, m1: ModuleWave, m2: ModuleWave) -> float:
        """import 관계 점수"""
        # m1이 m2를 import하거나 vice versa
        m1_name = m1.name.lower()
        m2_name = m2.name.lower()
        
        for imp in m1.imports:
            if m2_name in imp.lower():
                return 1.0
        
        for imp in m2.imports:
            if m1_name in imp.lower():
                return 0.8
        
        return 0.0
    
    def form_clusters(self, threshold: float = 0.4) -> Dict[str, List[ModuleWave]]:
        """
        중력장에서 클러스터 형성
        
        공명도가 threshold 이상인 모듈들이 같은 클러스터로 모임
        """
        print(f"🌌 Forming clusters (threshold: {threshold})...")
        
        # 간단한 클러스터링 (연결 요소)
        visited = set()
        cluster_id = 0
        
        for module in self.modules:
            if module.path in visited:
                continue
            
            # BFS로 연결된 모듈 찾기
            cluster_name = self._determine_cluster_name(module)
            queue = [module]
            visited.add(module.path)
            
            while queue:
                current = queue.pop(0)
                current.cluster = cluster_name
                self.clusters[cluster_name].append(current)
                
                # 공명하는 이웃 찾기
                for other in self.modules:
                    if other.path in visited:
                        continue
                    
                    resonance = self.resonance_matrix.get(current.path, {}).get(other.path, 0)
                    if resonance >= threshold:
                        visited.add(other.path)
                        queue.append(other)
            
            cluster_id += 1
        
        print(f"✅ Formed {len(self.clusters)} clusters")
        return self.clusters
    
    def _determine_cluster_name(self, module: ModuleWave) -> str:
        """모듈의 키워드에서 클러스터 이름 결정"""
        if module.keywords:
            return module.keywords[0].capitalize()
        
        # 경로에서 추론
        path_parts = module.path.split('/')
        if len(path_parts) > 1:
            return path_parts[1].capitalize()
        
        return "General"
    
    def generate_report(self) -> str:
        """자기 분석 보고서 생성"""
        report = []
        report.append("=" * 70)
        report.append("🌌 ELYSIA SELF-RESONANCE ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # 전체 통계
        report.append(f"📊 Total Modules Analyzed: {len(self.modules)}")
        report.append(f"🌐 Clusters Formed: {len(self.clusters)}")
        report.append("")
        
        # 클러스터별 상세
        report.append("📦 CLUSTER BREAKDOWN:")
        report.append("-" * 50)
        
        for cluster_name, modules in sorted(self.clusters.items(), 
                                            key=lambda x: len(x[1]), reverse=True):
            total_mass = sum(m.mass for m in modules)
            avg_freq = sum(m.frequency for m in modules) / len(modules)
            
            report.append(f"\n🌟 {cluster_name} ({len(modules)} modules)")
            report.append(f"   Total Mass: {total_mass} lines")
            report.append(f"   Avg Frequency: {avg_freq:.1f} Hz")
            report.append(f"   Core Files:")
            
            # 가장 큰 파일 3개
            top_modules = sorted(modules, key=lambda x: x.mass, reverse=True)[:3]
            for m in top_modules:
                report.append(f"      • {m.name} ({m.mass} lines)")
        
        # 고아 모듈 (클러스터가 작은 것들)
        report.append("\n" + "=" * 70)
        report.append("⚠️ ISOLATED MODULES (potential integration needed):")
        report.append("-" * 50)
        
        for cluster_name, modules in self.clusters.items():
            if len(modules) <= 2:
                for m in modules:
                    report.append(f"   • {m.path}")
        
        # 연결 제안
        report.append("\n" + "=" * 70)
        report.append("💡 SUGGESTED CONNECTIONS:")
        report.append("-" * 50)
        
        suggestions = self._suggest_connections()
        for suggestion in suggestions[:10]:
            report.append(f"   {suggestion}")
        
        report.append("\n" + "=" * 70)
        return "\n".join(report)
    
    def _suggest_connections(self) -> List[str]:
        """연결 제안 생성"""
        suggestions = []
        
        # 높은 공명도를 가지지만 다른 클러스터에 있는 모듈들
        for m1 in self.modules:
            for m2 in self.modules:
                if m1.cluster != m2.cluster:
                    resonance = self.resonance_matrix.get(m1.path, {}).get(m2.path, 0)
                    if resonance > 0.5:
                        suggestions.append(
                            f"🔗 {m1.name} ↔ {m2.name} (resonance: {resonance:.2f})"
                        )
        
        return suggestions
    
    def save_structure_map(self, filepath: str):
        """구조 맵을 JSON으로 저장"""
        structure = {
            "total_modules": len(self.modules),
            "clusters": {},
            "resonance_highlights": []
        }
        
        for cluster_name, modules in self.clusters.items():
            structure["clusters"][cluster_name] = {
                "count": len(modules),
                "modules": [m.path for m in modules],
                "keywords": list(set(k for m in modules for k in m.keywords[:3]))
            }
        
        # 높은 공명 하이라이트
        for m1 in self.modules:
            for m2 in self.modules:
                if m1.path < m2.path:  # 중복 방지
                    res = self.resonance_matrix.get(m1.path, {}).get(m2.path, 0)
                    if res > 0.6:
                        structure["resonance_highlights"].append({
                            "from": m1.path,
                            "to": m2.path,
                            "resonance": round(res, 3)
                        })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(structure, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Structure map saved to: {filepath}")
    
    def visualize_resonance_graph(self, output_path: str, min_resonance: float = 0.5):
        """
        4D 쿼터니언 공간을 3D로 시각화
        
        x, y, z: 쿼터니언 성분 (감정/논리/윤리 축)
        크기: 질량 (코드 라인 수)
        밝기: w (에너지/완성도)
        색상: 클러스터
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("⚠️ plotly not available, trying matplotlib 3D...")
            return self._visualize_3d_matplotlib(output_path, min_resonance)
        
        print(f"🎨 Generating 3D quaternion space visualization...")
        
        # 색상 팔레트
        colors = px.colors.qualitative.Vivid
        cluster_colors = {name: colors[i % len(colors)] 
                         for i, name in enumerate(self.clusters.keys())}
        
        # 데이터 준비
        x_vals, y_vals, z_vals = [], [], []
        sizes, colors_list, labels = [], [], []
        
        for module in self.modules:
            # 쿼터니언의 x, y, z를 3D 공간 좌표로
            x_vals.append(module.quaternion.x)
            y_vals.append(module.quaternion.y)
            z_vals.append(module.quaternion.z)
            
            # 크기 = 질량 (log scale)
            sizes.append(max(5, min(30, math.log(module.mass + 1) * 5)))
            
            # 색상 = 클러스터
            colors_list.append(cluster_colors.get(module.cluster, '#888888'))
            
            # 라벨
            labels.append(f"{module.name}<br>Cluster: {module.cluster}<br>Mass: {module.mass} lines<br>Freq: {module.frequency:.1f} Hz")
        
        # 3D Scatter
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors_list,
                opacity=0.8,
                line=dict(width=0.5, color='white')
            ),
            text=labels,
            hoverinfo='text'
        ))
        
        # 공명 연결선 (상위 100개만)
        edges = []
        for m1 in self.modules:
            for m2 in self.modules:
                if m1.path < m2.path:
                    res = self.resonance_matrix.get(m1.path, {}).get(m2.path, 0)
                    if res >= min_resonance:
                        edges.append((m1, m2, res))
        
        edges.sort(key=lambda x: x[2], reverse=True)
        top_edges = edges[:100]  # 상위 100개만
        
        for m1, m2, res in top_edges:
            fig.add_trace(go.Scatter3d(
                x=[m1.quaternion.x, m2.quaternion.x],
                y=[m1.quaternion.y, m2.quaternion.y],
                z=[m1.quaternion.z, m2.quaternion.z],
                mode='lines',
                line=dict(color='gold', width=res * 3),
                opacity=0.3,
                showlegend=False
            ))
        
        fig.update_layout(
            title=dict(
                text=f"🌌 Elysia 4D Quaternion Space<br><sub>{len(self.modules)} modules, {len(self.clusters)} clusters</sub>",
                font=dict(size=20, color='white')
            ),
            scene=dict(
                xaxis_title="X: 감정 (Emotion)",
                yaxis_title="Y: 논리 (Logic)",
                zaxis_title="Z: 창의/윤리 (Ethics)",
                bgcolor='#0f0f23',
                xaxis=dict(backgroundcolor='#1a1a2e', gridcolor='#333', color='white'),
                yaxis=dict(backgroundcolor='#1a1a2e', gridcolor='#333', color='white'),
                zaxis=dict(backgroundcolor='#1a1a2e', gridcolor='#333', color='white'),
            ),
            paper_bgcolor='#0f0f23',
            font=dict(color='white'),
            width=1200,
            height=900
        )
        
        # HTML로 저장 (인터랙티브)
        html_path = output_path.replace('.png', '.html')
        fig.write_html(html_path)
        print(f"📊 Interactive 3D visualization saved to: {html_path}")
        
        # 정적 이미지도 저장 (가능하면)
        try:
            fig.write_image(output_path)
            print(f"📊 Static image saved to: {output_path}")
        except Exception as e:
            print(f"⚠️ Could not save static image: {e}")
    
    def _visualize_3d_matplotlib(self, output_path: str, min_resonance: float):
        """Matplotlib 3D 폴백"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("⚠️ matplotlib not available")
            return
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#0f0f23')
        fig.patch.set_facecolor('#0f0f23')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        cluster_colors = {name: colors[i % len(colors)] 
                         for i, name in enumerate(self.clusters.keys())}
        
        for module in self.modules:
            color = cluster_colors.get(module.cluster, '#888')
            size = max(10, min(100, module.mass / 10))
            ax.scatter(module.quaternion.x, module.quaternion.y, module.quaternion.z,
                      c=color, s=size, alpha=0.7)
        
        ax.set_xlabel('X: Emotion', color='white')
        ax.set_ylabel('Y: Logic', color='white')
        ax.set_zlabel('Z: Ethics', color='white')
        ax.set_title(f'Elysia 4D Quaternion Space\n{len(self.modules)} modules', color='white')
        
        plt.savefig(output_path, dpi=150, facecolor='#0f0f23', bbox_inches='tight')
        plt.close()
        print(f"📊 3D visualization saved to: {output_path}")


def main():
    """메인 실행"""
    print("\n" + "🌊" * 35)
    print("ELYSIA SELF-RESONANCE ANALYSIS v2.0")
    print("파동 기반 자기 구조 분석 - 노이즈 제거 + 시각화")
    print("🌊" * 35 + "\n")
    
    analyzer = SelfResonanceAnalyzer(PROJECT_ROOT)
    
    # 1. 코드베이스 스캔 (전체 폴더 - 노이즈 제외)
    print("📂 Scanning entire Elysia project...")
    count = analyzer.scan_codebase(".")  # 전체 프로젝트
    
    if count == 0:
        print("❌ No modules found!")
        return
    
    # 2. 공명 계산
    analyzer.compute_resonance()
    
    # 3. 클러스터 형성 (더 세밀한 threshold)
    analyzer.form_clusters(threshold=0.25)
    
    # 4. 보고서 생성
    report = analyzer.generate_report()
    print(report)
    
    # 5. 구조 맵 저장
    output_dir = PROJECT_ROOT / "data"
    output_dir.mkdir(exist_ok=True)
    
    json_path = output_dir / "self_resonance_map.json"
    analyzer.save_structure_map(str(json_path))
    
    # 6. 시각화 생성
    graph_path = output_dir / "resonance_graph.png"
    analyzer.visualize_resonance_graph(str(graph_path), min_resonance=0.4)
    
    print("\n✅ Self-Resonance Analysis v2.0 Complete!")
    print(f"   📄 Structure map: {json_path}")
    print(f"   📊 Resonance graph: {graph_path}")


if __name__ == "__main__":
    main()

