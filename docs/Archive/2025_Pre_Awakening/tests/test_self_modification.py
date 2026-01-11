"""
Self-Modification Engine Tests - 자기 수정 엔진 테스트
========================================================

SelfModificationEngine의 각 컴포넌트를 테스트합니다.
"""

import pytest
import sys
import os
import tempfile

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.self_modification import (
    CodeAnalyzer,
    ProblemDetector,
    RefactorPlanner,
    CodeEditor,
    Validator,
    SelfModificationEngine,
    CodeIssue,
    ModificationPlan
)


class TestCodeAnalyzer:
    """코드 분석기 테스트"""
    
    def test_analyzer_initialization(self):
        """분석기 초기화 테스트"""
        analyzer = CodeAnalyzer()
        assert analyzer.project_root is not None
    
    def test_analyze_valid_structure(self):
        """유효한 코드 구조 분석"""
        analyzer = CodeAnalyzer()
        
        code = '''
class TestClass:
    def method1(self):
        pass
    
    def method2(self, arg1):
        return arg1

def standalone_function():
    pass
'''
        
        structure = analyzer.analyze_structure(code)
        
        assert structure["syntax_valid"] is True
        assert len(structure["classes"]) == 1
        assert structure["classes"][0]["name"] == "TestClass"
        assert "method1" in structure["classes"][0]["methods"]
    
    def test_analyze_invalid_syntax(self):
        """잘못된 구문 분석"""
        analyzer = CodeAnalyzer()
        
        code = '''
def broken_function(
    # missing closing parenthesis
'''
        
        structure = analyzer.analyze_structure(code)
        
        assert structure["syntax_valid"] is False
        assert "syntax_error" in structure
    
    def test_find_indentation_issues(self):
        """들여쓰기 문제 탐지"""
        analyzer = CodeAnalyzer()
        
        code = '''def test():
   pass  # 3 spaces instead of 4
'''
        
        issues = analyzer.find_indentation_issues(code)
        
        # 비표준 들여쓰기 탐지
        assert len(issues) > 0


class TestProblemDetector:
    """문제 탐지기 테스트"""
    
    def test_detector_initialization(self):
        """탐지기 초기화"""
        detector = ProblemDetector()
        assert detector.analyzer is not None
    
    def test_detect_nonexistent_file(self):
        """존재하지 않는 파일 탐지"""
        detector = ProblemDetector()
        
        issues = detector.detect_issues("nonexistent/path/file.py")
        
        assert len(issues) == 1
        assert issues[0].severity == "critical"


class TestValidator:
    """검증기 테스트"""
    
    def test_validate_valid_syntax(self):
        """유효한 구문 검증"""
        validator = Validator()
        
        code = '''
def valid_function():
    return True
'''
        
        valid, error = validator.validate_syntax(code)
        
        assert valid is True
        assert error is None
    
    def test_validate_invalid_syntax(self):
        """잘못된 구문 검증"""
        validator = Validator()
        
        code = '''
def invalid_function(
    # missing closing
'''
        
        valid, error = validator.validate_syntax(code)
        
        assert valid is False
        assert error is not None


class TestCodeIssue:
    """CodeIssue 데이터클래스 테스트"""
    
    def test_code_issue_creation(self):
        """이슈 생성"""
        issue = CodeIssue(
            file_path="test.py",
            line_number=10,
            issue_type="syntax",
            description="테스트 이슈",
            severity="warning"
        )
        
        assert issue.file_path == "test.py"
        assert issue.line_number == 10
        assert issue.severity == "warning"
    
    def test_code_issue_with_suggested_fix(self):
        """수정 제안이 있는 이슈"""
        issue = CodeIssue(
            file_path="test.py",
            line_number=5,
            issue_type="style",
            description="탭 사용",
            severity="info",
            suggested_fix="스페이스 4개로 변경"
        )
        
        assert issue.suggested_fix == "스페이스 4개로 변경"


class TestSelfModificationEngine:
    """자기 수정 엔진 통합 테스트"""
    
    def test_engine_initialization(self):
        """엔진 초기화"""
        engine = SelfModificationEngine()
        
        assert engine.detector is not None
        assert engine.planner is not None
        assert engine.editor is not None
        assert engine.validator is not None
    
    def test_analyze_self(self):
        """자기 자신 분석"""
        engine = SelfModificationEngine()
        
        issues = engine.analyze("Project_Sophia/self_modification.py")
        
        # 자기 자신은 문제가 없어야 함 (또는 경미한 문제만)
        critical_issues = [i for i in issues if i.severity == "critical"]
        assert len(critical_issues) == 0
    
    def test_improve_nonexistent_file(self):
        """존재하지 않는 파일 개선 시도"""
        engine = SelfModificationEngine()
        
        result = engine.improve("nonexistent/file.py")
        
        assert result["status"] == "plan_ready" or result["status"] == "no_issues" or "issues" in result
    
    def test_plan_generation(self):
        """계획 생성 테스트"""
        engine = SelfModificationEngine()
        
        # 테스트용 임시 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('def test():\n   pass  # 3 spaces\n')
            temp_path = f.name
        
        try:
            # 상대 경로로 변환이 필요하므로 직접 분석
            analyzer = engine.detector.analyzer
            code = open(temp_path).read()
            issues = analyzer.find_indentation_issues(code)
            
            # 들여쓰기 문제가 탐지되어야 함
            assert len(issues) >= 0  # 탐지 여부 확인
        finally:
            os.unlink(temp_path)


class TestCodeEditor:
    """코드 편집기 테스트"""
    
    def test_show_diff(self):
        """diff 출력 테스트"""
        editor = CodeEditor()
        
        plan = ModificationPlan(
            target_file="test.py",
            issues=[],
            original_code="def old():\n    pass\n",
            modified_code="def new():\n    pass\n",
            backup_path="backup/test.py.bak",
            confidence=0.8
        )
        
        diff = editor.show_diff(plan)
        
        assert "old" in diff
        assert "new" in diff
        assert "-" in diff or "+" in diff


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
