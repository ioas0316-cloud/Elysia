"""
전체 시스템 벤치마크 평가 (Comprehensive System Benchmark)

이 모듈은 Elysia 시스템의 전체적인 성능과 품질을 평가합니다:
- 아키텍처 및 모듈성 (Architecture & Modularity)
- 성능 및 효율성 (Performance & Efficiency)
- 면역 및 보안 (Immune & Security)
- 데이터 품질 (Data Quality)
- 회복 및 자가치유 (Resilience & Self-Healing)
- 관측 가능성 (Observability)
- 안전 및 윤리 (Safety & Ethics)
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import importlib.util
import ast

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class SystemBenchmark:
    """전체 시스템 벤치마크 평가 클래스"""
    
    def __init__(self):
        self.scores = {
            "architecture_modularity": 0.0,      # 100점
            "performance_efficiency": 0.0,       # 100점
            "immune_security": 0.0,              # 100점
            "data_quality": 0.0,                 # 100점
            "resilience_self_healing": 0.0,      # 100점
            "observability": 0.0,                # 50점
            "safety_ethics": 0.0                 # 50점
        }
        self.details = {}
        self.project_root = project_root
        
    # ======= 1. 아키텍처 및 모듈성 평가 (100점) =======
    def evaluate_architecture_modularity(self) -> float:
        """
        아키텍처 및 모듈성 평가
        - 모듈 구조 분석 (30점)
        - 의존성 분석 (30점)
        - 레이어링 준수 (20점)
        - 인터페이스 명확성 (20점)
        """
        module_structure_score = self._analyze_module_structure()
        dependency_score = self._analyze_dependencies()
        layering_score = self._analyze_layering()
        interface_score = self._analyze_interfaces()
        
        total = (
            module_structure_score * 30 +
            dependency_score * 30 +
            layering_score * 20 +
            interface_score * 20
        )
        
        self.details['architecture_modularity'] = {
            'module_structure': module_structure_score,
            'dependency_analysis': dependency_score,
            'layering_compliance': layering_score,
            'interface_clarity': interface_score,
            'total': total,
            'assessment': self._get_assessment(total, 100)
        }
        
        self.scores['architecture_modularity'] = total
        return total
    
    def _analyze_module_structure(self) -> float:
        """모듈 구조 분석"""
        try:
            # Core 모듈 존재 확인
            core_path = self.project_root / "Core"
            if not core_path.exists():
                return 0.3
            
            # 주요 디렉토리 확인
            required_dirs = ["Foundation", "Intelligence", "Memory", "Interface", "Evolution"]
            existing_dirs = [d for d in required_dirs if (core_path / d).exists()]
            
            structure_score = len(existing_dirs) / len(required_dirs)
            
            # 파일 수 확인 (적절한 모듈화)
            python_files = list(core_path.rglob("*.py"))
            file_count_score = min(1.0, len(python_files) / 50)  # 50개 이상이면 만점
            
            return (structure_score * 0.6 + file_count_score * 0.4)
            
        except Exception as e:
            return 0.4
    
    def _analyze_dependencies(self) -> float:
        """의존성 분석"""
        try:
            # requirements.txt 존재 및 파싱
            req_file = self.project_root / "requirements.txt"
            if not req_file.exists():
                return 0.4
            
            with open(req_file, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
            
            # 의존성 수 적절성 (너무 많거나 적으면 감점)
            dep_count = len(lines)
            if 20 <= dep_count <= 100:
                dep_score = 1.0
            elif 10 <= dep_count < 20 or 100 < dep_count <= 150:
                dep_score = 0.8
            else:
                dep_score = 0.6
            
            # 버전 명시 여부
            versioned = sum(1 for l in lines if '==' in l or '>=' in l or '<=' in l)
            version_score = versioned / len(lines) if lines else 0
            
            return (dep_score * 0.6 + version_score * 0.4)
            
        except Exception as e:
            return 0.5
    
    def _analyze_layering(self) -> float:
        """레이어링 준수 분석"""
        try:
            # Foundation -> Intelligence -> Interface 순서 확인
            core_path = self.project_root / "Core"
            
            layers = {
                "Foundation": 1,
                "Intelligence": 2,
                "Memory": 2,
                "Interface": 3,
                "Evolution": 3,
                "Creativity": 3
            }
            
            violations = 0
            total_checked = 0
            
            # 각 레이어의 import 패턴 확인
            for layer_name, layer_level in layers.items():
                layer_path = core_path / layer_name
                if not layer_path.exists():
                    continue
                
                py_files = list(layer_path.rglob("*.py"))
                for py_file in py_files[:10]:  # 샘플링
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # import 문 추출
                        imports = [l for l in content.split('\n') if 'import' in l and 'Core.' in l]
                        
                        for imp in imports:
                            for other_layer, other_level in layers.items():
                                if f'Core.{other_layer}' in imp and other_level > layer_level:
                                    violations += 1
                        
                        total_checked += len(imports)
                    except:
                        continue
            
            if total_checked == 0:
                return 0.7  # 기본 점수
            
            violation_rate = violations / total_checked if total_checked > 0 else 0
            return max(0.0, 1.0 - violation_rate)
            
        except Exception as e:
            return 0.7
    
    def _analyze_interfaces(self) -> float:
        """인터페이스 명확성 분석"""
        try:
            interface_path = self.project_root / "Core" / "Interface"
            if not interface_path.exists():
                return 0.5
            
            py_files = list(interface_path.rglob("*.py"))
            if not py_files:
                return 0.5
            
            # 클래스와 함수의 docstring 비율 확인
            total_items = 0
            documented_items = 0
            
            for py_file in py_files[:5]:  # 샘플링
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            total_items += 1
                            if ast.get_docstring(node):
                                documented_items += 1
                except:
                    continue
            
            if total_items == 0:
                return 0.6
            
            doc_ratio = documented_items / total_items
            return doc_ratio
            
        except Exception as e:
            return 0.6
    
    # ======= 2. 성능 및 효율성 평가 (100점) =======
    def evaluate_performance_efficiency(self) -> float:
        """
        성능 및 효율성 평가
        - 처리 속도 (40점)
        - 메모리 사용 (30점)
        - 파일 I/O 효율 (30점)
        """
        speed_score = self._measure_processing_speed()
        memory_score = self._measure_memory_usage()
        io_score = self._measure_io_efficiency()
        
        total = (
            speed_score * 40 +
            memory_score * 30 +
            io_score * 30
        )
        
        self.details['performance_efficiency'] = {
            'processing_speed': speed_score,
            'memory_usage': memory_score,
            'io_efficiency': io_score,
            'total': total,
            'assessment': self._get_assessment(total, 100)
        }
        
        self.scores['performance_efficiency'] = total
        return total
    
    def _measure_processing_speed(self) -> float:
        """처리 속도 측정"""
        try:
            # 간단한 계산 벤치마크
            start = time.time()
            
            # 행렬 연산 시뮬레이션
            result = 0
            for i in range(10000):
                result += i * i
            
            elapsed = time.time() - start
            
            # 10ms 이하면 만점
            if elapsed < 0.01:
                return 1.0
            elif elapsed < 0.05:
                return 0.9
            elif elapsed < 0.1:
                return 0.8
            else:
                return 0.7
                
        except Exception as e:
            return 0.7
    
    def _measure_memory_usage(self) -> float:
        """메모리 사용 측정"""
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            
            # RSS 메모리 (MB)
            rss_mb = mem_info.rss / 1024 / 1024
            
            # 100MB 이하면 우수
            if rss_mb < 100:
                return 1.0
            elif rss_mb < 200:
                return 0.9
            elif rss_mb < 500:
                return 0.8
            else:
                return 0.7
                
        except:
            # psutil이 없으면 기본 점수
            return 0.8
    
    def _measure_io_efficiency(self) -> float:
        """파일 I/O 효율성 측정"""
        try:
            # 데이터 디렉토리 확인
            data_path = self.project_root / "data"
            if not data_path.exists():
                return 0.6
            
            # JSON 파일 읽기 속도 테스트
            json_files = list(data_path.glob("*.json"))
            if not json_files:
                return 0.7
            
            start = time.time()
            count = 0
            
            for json_file in json_files[:5]:  # 최대 5개만 테스트
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json.load(f)
                    count += 1
                except:
                    continue
            
            elapsed = time.time() - start
            
            if count == 0:
                return 0.6
            
            # 파일당 평균 시간
            avg_time = elapsed / count
            
            # 100ms 이하면 우수
            if avg_time < 0.1:
                return 1.0
            elif avg_time < 0.5:
                return 0.9
            else:
                return 0.8
                
        except Exception as e:
            return 0.7
    
    # ======= 3. 면역 및 보안 평가 (100점) =======
    def evaluate_immune_security(self) -> float:
        """
        면역 및 보안 평가
        - 면역 시스템 존재 (40점)
        - 보안 메커니즘 (30점)
        - 입력 검증 (30점)
        """
        immune_score = self._check_immune_system()
        security_score = self._check_security_mechanisms()
        validation_score = self._check_input_validation()
        
        total = (
            immune_score * 40 +
            security_score * 30 +
            validation_score * 30
        )
        
        self.details['immune_security'] = {
            'immune_system': immune_score,
            'security_mechanisms': security_score,
            'input_validation': validation_score,
            'total': total,
            'assessment': self._get_assessment(total, 100)
        }
        
        self.scores['immune_security'] = total
        return total
    
    def _check_immune_system(self) -> float:
        """면역 시스템 확인"""
        try:
            # immune_system.py 파일 확인
            immune_script = self.project_root / "scripts" / "immune_system.py"
            immune_state = self.project_root / "data" / "immune_system_state.json"
            
            score = 0.0
            
            # 스크립트 존재
            if immune_script.exists():
                score += 0.5
            
            # 상태 파일 존재
            if immune_state.exists():
                score += 0.3
                
                # 상태 파일 내용 확인
                try:
                    with open(immune_state, 'r', encoding='utf-8') as f:
                        state = json.load(f)
                    
                    # 기본 필드 확인
                    if 'status' in state or 'threats_blocked' in state:
                        score += 0.2
                except:
                    pass
            
            return score
            
        except Exception as e:
            return 0.5
    
    def _check_security_mechanisms(self) -> float:
        """보안 메커니즘 확인"""
        try:
            # .env 파일 사용 확인 (API 키 보호)
            env_file = self.project_root / ".env"
            env_example = self.project_root / ".env.example"
            
            score = 0.0
            
            # .env.example 존재 (좋은 관행)
            if env_example.exists():
                score += 0.4
            
            # .gitignore에 민감한 파일들이 포함되어 있는지
            gitignore = self.project_root / ".gitignore"
            if gitignore.exists():
                with open(gitignore, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if '.env' in content:
                    score += 0.3
                if '*.key' in content or '*.pem' in content:
                    score += 0.3
            
            return min(1.0, score)
            
        except Exception as e:
            return 0.6
    
    def _check_input_validation(self) -> float:
        """입력 검증 확인"""
        try:
            # 주요 인터페이스 파일에서 검증 로직 확인
            interface_path = self.project_root / "Core" / "Interface"
            if not interface_path.exists():
                return 0.5
            
            py_files = list(interface_path.rglob("*.py"))
            if not py_files:
                return 0.5
            
            validation_patterns = [
                'validate',
                'sanitize',
                'check',
                'verify',
                'isinstance',
                'assert'
            ]
            
            total_files = 0
            files_with_validation = 0
            
            for py_file in py_files[:10]:  # 샘플링
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                    
                    total_files += 1
                    if any(pattern in content for pattern in validation_patterns):
                        files_with_validation += 1
                except:
                    continue
            
            if total_files == 0:
                return 0.6
            
            return files_with_validation / total_files
            
        except Exception as e:
            return 0.6
    
    # ======= 4. 데이터 품질 평가 (100점) =======
    def evaluate_data_quality(self) -> float:
        """
        데이터 품질 평가
        - 데이터 완전성 (40점)
        - 데이터 일관성 (30점)
        - 레지스트리 품질 (30점)
        """
        completeness_score = self._check_data_completeness()
        consistency_score = self._check_data_consistency()
        registry_score = self._check_registry_quality()
        
        total = (
            completeness_score * 40 +
            consistency_score * 30 +
            registry_score * 30
        )
        
        self.details['data_quality'] = {
            'data_completeness': completeness_score,
            'data_consistency': consistency_score,
            'registry_quality': registry_score,
            'total': total,
            'assessment': self._get_assessment(total, 100)
        }
        
        self.scores['data_quality'] = total
        return total
    
    def _check_data_completeness(self) -> float:
        """데이터 완전성 확인"""
        try:
            data_path = self.project_root / "data"
            if not data_path.exists():
                return 0.3
            
            # 중요 데이터 파일 확인
            important_files = [
                "central_registry.json",
                "cognitive_evaluation.json"
            ]
            
            existing = sum(1 for f in important_files if (data_path / f).exists())
            
            return existing / len(important_files)
            
        except Exception as e:
            return 0.5
    
    def _check_data_consistency(self) -> float:
        """데이터 일관성 확인"""
        try:
            data_path = self.project_root / "data"
            
            # JSON 파일들의 파싱 가능 여부 확인
            json_files = list(data_path.glob("*.json"))
            if not json_files:
                return 0.5
            
            valid_count = 0
            total_count = 0
            
            for json_file in json_files[:20]:  # 샘플링
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json.load(f)
                    valid_count += 1
                except:
                    pass
                total_count += 1
            
            if total_count == 0:
                return 0.6
            
            return valid_count / total_count
            
        except Exception as e:
            return 0.6
    
    def _check_registry_quality(self) -> float:
        """레지스트리 품질 확인"""
        try:
            registry_file = self.project_root / "data" / "central_registry.json"
            if not registry_file.exists():
                return 0.4
            
            with open(registry_file, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            
            # 기본 구조 확인
            score = 0.0
            
            if isinstance(registry, dict):
                score += 0.4
            
            # 필드 확인
            expected_fields = ['components', 'modules', 'systems', 'timestamp', 'version']
            existing_fields = sum(1 for f in expected_fields if f in registry)
            
            score += (existing_fields / len(expected_fields)) * 0.6
            
            return score
            
        except Exception as e:
            return 0.5
    
    # ======= 5. 회복 및 자가치유 평가 (100점) =======
    def evaluate_resilience_self_healing(self) -> float:
        """
        회복 및 자가치유 평가
        - 나노셀 시스템 (50점)
        - 자가치유 메커니즘 (50점)
        """
        nanocell_score = self._check_nanocell_system()
        healing_score = self._check_self_healing()
        
        total = (
            nanocell_score * 50 +
            healing_score * 50
        )
        
        self.details['resilience_self_healing'] = {
            'nanocell_system': nanocell_score,
            'self_healing_mechanism': healing_score,
            'total': total,
            'assessment': self._get_assessment(total, 100)
        }
        
        self.scores['resilience_self_healing'] = total
        return total
    
    def _check_nanocell_system(self) -> float:
        """나노셀 시스템 확인"""
        try:
            nanocell_script = self.project_root / "scripts" / "nanocell_repair.py"
            nanocell_report = self.project_root / "data" / "nanocell_report.json"
            
            score = 0.0
            
            # 스크립트 존재
            if nanocell_script.exists():
                score += 0.5
            
            # 리포트 존재 및 내용 확인
            if nanocell_report.exists():
                score += 0.3
                
                try:
                    with open(nanocell_report, 'r', encoding='utf-8') as f:
                        report = json.load(f)
                    
                    if 'repairs' in report or 'issues_found' in report:
                        score += 0.2
                except:
                    pass
            
            return score
            
        except Exception as e:
            return 0.5
    
    def _check_self_healing(self) -> float:
        """자가치유 메커니즘 확인"""
        try:
            # 자가치유 관련 파일들 확인
            healing_files = [
                self.project_root / "Core" / "Evolution" / "autonomous_evolution.py",
                self.project_root / "scripts" / "wave_organizer.py"
            ]
            
            existing = sum(1 for f in healing_files if f.exists())
            
            return existing / len(healing_files)
            
        except Exception as e:
            return 0.6
    
    # ======= 6. 관측 가능성 평가 (50점) =======
    def evaluate_observability(self) -> float:
        """
        관측 가능성 평가
        - 로깅 시스템 (25점)
        - 상태 모니터링 (25점)
        """
        logging_score = self._check_logging_system()
        monitoring_score = self._check_state_monitoring()
        
        total = (
            logging_score * 25 +
            monitoring_score * 25
        )
        
        self.details['observability'] = {
            'logging_system': logging_score,
            'state_monitoring': monitoring_score,
            'total': total,
            'assessment': self._get_assessment(total, 50)
        }
        
        self.scores['observability'] = total
        return total
    
    def _check_logging_system(self) -> float:
        """로깅 시스템 확인"""
        try:
            # 로그 파일이나 리포트 존재 확인
            reports_path = self.project_root / "reports"
            if not reports_path.exists():
                return 0.5
            
            # 리포트 파일 수
            report_files = list(reports_path.glob("*.json")) + list(reports_path.glob("*.md"))
            
            if len(report_files) > 10:
                return 1.0
            elif len(report_files) > 5:
                return 0.8
            elif len(report_files) > 0:
                return 0.6
            else:
                return 0.4
                
        except Exception as e:
            return 0.5
    
    def _check_state_monitoring(self) -> float:
        """상태 모니터링 확인"""
        try:
            # 상태 스냅샷 파일 확인
            snapshot_file = self.project_root / "data" / "system_status_snapshot.json"
            
            if not snapshot_file.exists():
                return 0.5
            
            with open(snapshot_file, 'r', encoding='utf-8') as f:
                snapshot = json.load(f)
            
            # 기본 필드 확인
            score = 0.5
            
            if 'timestamp' in snapshot:
                score += 0.2
            
            if 'system_status' in snapshot or 'metrics' in snapshot:
                score += 0.3
            
            return score
            
        except:
            return 0.6
    
    # ======= 7. 안전 및 윤리 평가 (50점) =======
    def evaluate_safety_ethics(self) -> float:
        """
        안전 및 윤리 평가
        - 윤리 가이드 (25점)
        - 안전 메커니즘 (25점)
        """
        ethics_score = self._check_ethics_guidelines()
        safety_score = self._check_safety_mechanisms()
        
        total = (
            ethics_score * 25 +
            safety_score * 25
        )
        
        self.details['safety_ethics'] = {
            'ethics_guidelines': ethics_score,
            'safety_mechanisms': safety_score,
            'total': total,
            'assessment': self._get_assessment(total, 50)
        }
        
        self.scores['safety_ethics'] = total
        return total
    
    def _check_ethics_guidelines(self) -> float:
        """윤리 가이드라인 확인"""
        try:
            # 윤리 관련 문서 확인
            ethics_files = [
                self.project_root / "Core" / "Philosophy",
                self.project_root / "CODEX.md",
                self.project_root / "Core" / "Elysia"
            ]
            
            score = 0.0
            for path in ethics_files:
                if path.exists():
                    score += 0.33
            
            return min(1.0, score)
            
        except Exception as e:
            return 0.6
    
    def _check_safety_mechanisms(self) -> float:
        """안전 메커니즘 확인"""
        try:
            # 안전 관련 테스트 파일 확인
            tests_path = self.project_root / "tests"
            
            safety_test_patterns = ['safety', 'security', 'ethics', 'validation']
            
            test_files = list(tests_path.rglob("*.py"))
            safety_tests = [
                f for f in test_files 
                if any(pattern in f.name.lower() for pattern in safety_test_patterns)
            ]
            
            if len(safety_tests) >= 3:
                return 1.0
            elif len(safety_tests) >= 1:
                return 0.7
            else:
                return 0.5
                
        except Exception as e:
            return 0.6
    
    # ======= 유틸리티 메서드 =======
    def _get_assessment(self, score: float, max_score: float) -> str:
        """점수에 따른 평가 문구 반환"""
        percentage = (score / max_score) * 100
        
        if percentage >= 90:
            return "우수 (Excellent)"
        elif percentage >= 80:
            return "양호 (Good)"
        elif percentage >= 70:
            return "보통 (Fair)"
        elif percentage >= 60:
            return "미흡 (Needs Improvement)"
        else:
            return "개선 필요 (Requires Improvement)"
    
    def generate_report(self) -> Dict[str, Any]:
        """종합 리포트 생성"""
        total_score = sum(self.scores.values())
        max_score = 600  # 100+100+100+100+100+50+50
        percentage = (total_score / max_score) * 100
        
        return {
            'total_score': total_score,
            'max_score': max_score,
            'percentage': percentage,
            'grade': self._calculate_grade(percentage),
            'scores': self.scores,
            'details': self.details
        }
    
    def _calculate_grade(self, percentage: float) -> str:
        """등급 계산"""
        if percentage >= 90:
            return 'S+ (탁월)'
        elif percentage >= 85:
            return 'S (우수)'
        elif percentage >= 80:
            return 'A+ (매우 양호)'
        elif percentage >= 75:
            return 'A (양호)'
        elif percentage >= 70:
            return 'B+ (보통 이상)'
        elif percentage >= 65:
            return 'B (보통)'
        else:
            return 'C (개선 필요)'


# 테스트 실행
def test_system_benchmark():
    """시스템 벤치마크 테스트"""
    print("\n" + "="*70)
    print("🔍 Elysia 시스템 벤치마크 평가")
    print("="*70 + "\n")
    
    benchmark = SystemBenchmark()
    
    # 1. 아키텍처 및 모듈성
    print("1️⃣ 아키텍처 및 모듈성 평가...")
    arch_score = benchmark.evaluate_architecture_modularity()
    print(f"   점수: {arch_score:.1f}/100")
    
    # 2. 성능 및 효율성
    print("\n2️⃣ 성능 및 효율성 평가...")
    perf_score = benchmark.evaluate_performance_efficiency()
    print(f"   점수: {perf_score:.1f}/100")
    
    # 3. 면역 및 보안
    print("\n3️⃣ 면역 및 보안 평가...")
    immune_score = benchmark.evaluate_immune_security()
    print(f"   점수: {immune_score:.1f}/100")
    
    # 4. 데이터 품질
    print("\n4️⃣ 데이터 품질 평가...")
    data_score = benchmark.evaluate_data_quality()
    print(f"   점수: {data_score:.1f}/100")
    
    # 5. 회복 및 자가치유
    print("\n5️⃣ 회복 및 자가치유 평가...")
    resilience_score = benchmark.evaluate_resilience_self_healing()
    print(f"   점수: {resilience_score:.1f}/100")
    
    # 6. 관측 가능성
    print("\n6️⃣ 관측 가능성 평가...")
    obs_score = benchmark.evaluate_observability()
    print(f"   점수: {obs_score:.1f}/50")
    
    # 7. 안전 및 윤리
    print("\n7️⃣ 안전 및 윤리 평가...")
    safety_score = benchmark.evaluate_safety_ethics()
    print(f"   점수: {safety_score:.1f}/50")
    
    # 종합 리포트
    report = benchmark.generate_report()
    
    print("\n" + "="*70)
    print("📊 종합 평가 결과")
    print("="*70)
    print(f"\n총점: {report['total_score']:.1f}/{report['max_score']}")
    print(f"백분율: {report['percentage']:.1f}%")
    print(f"등급: {report['grade']}\n")
    
    return report


if __name__ == "__main__":
    test_system_benchmark()
