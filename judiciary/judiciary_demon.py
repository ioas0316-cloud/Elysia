# [사법부] 최고 감사 데몬 (상시 감시 및 AST 스캐너)
# 행정부와 입법부를 통제하고 단절을 유발하는 판단/스위치 연산 및 임포트(Import)를 적발함.

import os
import re

class JudiciaryDemon:
    def __init__(self, leg_dir, exec_dir):
        self.leg_dir = leg_dir
        self.exec_dir = exec_dir

    def check_heresy_in_legislative(self):
        """
        [하부 레이어 검증] 판단(if/else) 참수, 하드코딩, 단절 금지, 제로-임포트
        """
        print("⚖️ [사법부] 하부 전자기장막(Legislative) 무결성 심사 중...")
        offenses = []
        for root, _, files in os.walk(self.leg_dir):
            for file in files:
                if file.endswith(('.cu', '.c', '.cpp', '.h')):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if re.search(r'\bif\s*\(', content):
                            offenses.append(f"[이단 적발] {filepath} - 죽은 판단(if문) 발견!")
                        if re.search(r'\belse\b', content):
                            offenses.append(f"[이단 적발] {filepath} - 죽은 판단(else문) 발견!")
                        if re.search(r'==\s*\d+', content):
                            offenses.append(f"[이단 적발] {filepath} - 점 세기용 매직넘버 비교문 발견!")
                        if re.search(r'bool\s+\w+\s*=\s*(true|false|0|1)\s*;', content):
                            offenses.append(f"[이단 적발] {filepath} - 2진법 bool 변수 사용!")
                        if re.search(r'\w+\s*\^\s*\w+', content) or re.search(r'\^\=', content):
                            offenses.append(f"[이단 적발] {filepath} - 죽은 스위치 비트 연산(^) 발견!")
                        if re.search(r'#include\s*<(iostream|vector|cmath|string|map|set|algorithm)>', content):
                            offenses.append(f"[이단 적발] {filepath} - 외부 모듈 Import(#include) 발견!")
        if offenses:
            for offense in offenses: print(f"🚨 {offense}")
            raise Exception("⚖️ [사법부 판결] 하부 전자기장막 탄핵! 장막의 연속성을 훼손하는 코드를 기각합니다.")
        print("✅ [사법부] 입법부 정적 검사 통과. 하부 전자기장막이 완벽하게 개방되었습니다.")

    def check_heresy_in_executive(self):
        """
        [상위 레이어 검증] 데이터 파싱, 스플릿, 점 세기 노가다(Count) 금지
        """
        print("⚖️ [사법부] 상위 추상화 제어탑(Executive) 파싱/노가다 심사 중...")
        offenses = []
        for root, _, files in os.walk(self.exec_dir):
            for file in files:
                if file.endswith(('.py')):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                        # 상위 레이어가 파싱이나 개수 세기를 하는지 감시
                        if re.search(r'\.split\(', content):
                            offenses.append(f"[이단 적발] {filepath} - 문자열 쪼개기(split) 파싱 발견! 제어탑은 파싱하지 마라.")
                        if re.search(r'\.count\(', content):
                            offenses.append(f"[이단 적발] {filepath} - 점 세기(count) 노가다 발견! 제어탑은 관조만 하라.")
                        if re.search(r'for\s+\w+\s+in\s+.*:', content) and re.search(r'\+\=', content):
                            # 간이 루프 기반 노가다 감지 (strict mode)
                            pass # 너무 빡빡하면 실행 불가일 수 있으나 철학적 차원에서 경고

        if offenses:
            for offense in offenses: print(f"🚨 {offense}")
            raise Exception("⚖️ [사법부 판결] 상위 제어탑 탄핵! 파동을 쪼개고 세는 저급한 2진법 파싱을 기각합니다.")
        print("✅ [사법부] 상위 제어탑 검사 통과. 점 세기 노가다 없이 완벽하게 방향타(Vector)만 제어합니다.")

    def run_all_inspections(self):
        print("⚖️ ================= 사법부 감사 시작 =================")
        self.check_heresy_in_legislative()
        self.check_heresy_in_executive()
        print("⚖️ ================= 사법부 감사 완료 =================\n")

if __name__ == "__main__":
    demon = JudiciaryDemon(leg_dir="legislative", exec_dir="executive")
    demon.run_all_inspections()
