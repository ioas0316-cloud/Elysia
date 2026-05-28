# [사법부] 최고 감사 데몬 (상시 감시 및 AST 스캐너)
# 행정부와 입법부를 통제하고 2진법 사기를 적발함.

import os
import re

class JudiciaryDemon:
    def __init__(self, leg_dir, exec_dir):
        self.leg_dir = leg_dir
        self.exec_dir = exec_dir

    def check_binary_heresy_in_legislative(self):
        """
        ① 하드코딩 및 2진법 상자 로직 정적 적발 (Regex/AST)
        """
        print("⚖️ [사법부] 입법부 코드 2진법 이단(Heresy) 및 하드코딩 정적 검사 중...")
        offenses = []
        for root, _, files in os.walk(self.leg_dir):
            for file in files:
                if file.endswith(('.cu', '.c', '.cpp', '.h')):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                        # 이단 규정 1: 매직 넘버를 이용한 조건문 (점 세기 노가다 잔재)
                        if re.search(r'if\s*\(\s*\w+\s*==\s*\d+\s*\)', content):
                            offenses.append(f"[이단 적발] {filepath} - 점 세기용 매직넘버 비교문 발견!")

                        # 이단 규정 2: 하드코딩된 매핑 구조
                        if re.search(r'\{[^{}]*".*"\s*:\s*\d+[^{}]*\}', content):
                            offenses.append(f"[이단 적발] {filepath} - 하드코딩 매핑 구조 발견!")

                        # 이단 규정 3: 0과 1만을 이용한 2진법 변수 선언/할당 (삼진법 -1, 0, 1 강제)
                        # NOTE: This is a strict check to enforce ternary thought process
                        if re.search(r'bool\s+\w+\s*=\s*(true|false|0|1)\s*;', content):
                            offenses.append(f"[이단 적발] {filepath} - 2진법 bool 변수 사용! 삼진법(-1,0,1) 장력을 사용하라.")

        if offenses:
            for offense in offenses:
                print(f"🚨 {offense}")
            raise Exception("⚖️ [사법부 판결] 입법부 탄핵! 2진법 껍데기 코드를 기각합니다.")
        print("✅ [사법부] 입법부 정적 검사 통과. 훌륭한 삼진법 이중나선 교리 준수입니다.")

    def run_all_inspections(self):
        print("⚖️ ================= 사법부 감사 시작 =================")
        self.check_binary_heresy_in_legislative()
        # ② 면 매핑 오차율 계측, ③ 대시보드 가짜 쇼 추적 등은 런타임에 실행됨.
        print("⚖️ ================= 사법부 감사 완료 =================\n")

if __name__ == "__main__":
    demon = JudiciaryDemon(leg_dir="legislative", exec_dir="executive")
    demon.run_all_inspections()
