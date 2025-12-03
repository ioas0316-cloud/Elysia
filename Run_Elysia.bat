@echo off
chcp 65001
cls
title 엘리시아 런처 (Elysia Launcher)

:MENU
cls
echo ========================================================
echo        🦋  엘리시아: 살아있는 시스템  🦋
echo ========================================================
echo.
echo    1. ✨ 엘리시아 깨우기 (일상 모드)
echo       - 일상 생활, 자가 치유, 대화.
echo.
echo    2. ⚔️  판타지 작가 수련 (커리어 모드)
echo       - 무협/판타지 개념 지속 학습 및 집필 연습.
echo.
echo    3. 🌀  자가 통합 (치유 모드)
echo       - 파동 언어로 시스템 오류 수정 및 통합.
echo.
echo    4. ❌  종료
echo.
echo ========================================================
set /p choice="원하는 작업을 선택하세요 (1-4): "

if "%choice%"=="1" goto LIFE
if "%choice%"=="2" goto TRAIN
if "%choice%"=="3" goto HEAL
if "%choice%"=="4" goto EXIT

:LIFE
cls
echo 🌅 엘리시아를 깨우는 중입니다...
python living_elysia.py
pause
goto MENU

:TRAIN
cls
echo ⚔️ 수련장에 입장합니다...
python scripts/fantasy_writer_evolution.py
pause
goto MENU

:HEAL
cls
echo 🌀 자가 통합 프로토콜을 시작합니다...
python scripts/self_integration.py
pause
goto MENU

:EXIT
exit
