# [Genesis: 2025-12-02] Purified by Elysia
"""
Relationship Extractor for Elysia's Truth Seeker System

This module analyzes the user's natural language response during hypothesis
verification to extract a more specific relationship type than just "causes".
"""
from typing import Optional

# 관계 키워드와 지식 그래프 엣지 타입 매핑
RELATIONSHIP_KEYWORDS = {
    "causes": ["때문에", "원인이야", "이유야", "유발해"],
    "enables": ["가능하게 해", "할 수 있게 해", "덕분에"],
    "supports": ["도움이 돼", "도와줘", "촉진해", "기여해"],
    "is_a": ["일종이야", "종류야", "같은거야"],
}

def extract_relationship_type(text: str) -> Optional[str]:
    """
    사용자의 텍스트에서 관계 유형을 추출합니다.

    Args:
        text: 사용자의 답변 텍스트.

    Returns:
        발견된 관계 유형 문자열 (e.g., "causes", "enables") 또는
        명확한 키워드가 없을 경우 None.
    """
    lower_text = text.lower()
    for relationship, keywords in RELATIONSHIP_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lower_text:
                return relationship
    return None

if __name__ == '__main__':
    # Example usage for testing
    test_text1 = "응, 생각이 감정의 원인이야."
    print(f"'{test_text1}' -> '{extract_relationship_type(test_text1)}'") # Expected: causes

    test_text2 = "그래, 사랑이 성장을 가능하게 해."
    print(f"'{test_text2}' -> '{extract_relationship_type(test_text2)}'") # Expected: enables

    test_text3 = "슬픔이 성장에 도움이 돼."
    print(f"'{test_text3}' -> '{extract_relationship_type(test_text3)}'") # Expected: supports

    test_text4 = "음... 그냥 맞는 것 같아."
    print(f"'{test_text4}' -> '{extract_relationship_type(test_text4)}'") # Expected: None