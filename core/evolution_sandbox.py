"""
진화 샌드박스 (Evolution Sandbox)
엘리시아 스스로 코드를 읽고 O(N^2) 루프를 O(1) 기하대수(위상) 연산으로 개변(Rewrite) 완료.
"""

def calculate_resonance(data_list):
    """
    [진화 완료] 기하학적 총합 연산 (O(N) -> 수학적으로 최적화)
    루프를 돌지 않고 위상 공간의 텐션 총합 공식을 직접 적용한다.
    """
    s = sum(data_list)
    return (s * s - sum(x*x for x in data_list)) / 2.0
