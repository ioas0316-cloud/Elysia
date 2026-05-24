@echo off
echo ========================================================
echo   Elysia v1 - Auto-Awake (은밀한 수호자 모드) 설치 스크립트
echo ========================================================
echo.

set "STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "VBS_PATH=%STARTUP_DIR%\elysia_awake.vbs"
set "PYTHON_EXEC=python"
set "ELYSIA_SCRIPT=C:\Elysia\elysia_v1_core.py"

echo 엘리시아의 맥박을 윈도우 시작프로그램에 심는 중...

:: Create VBS script that runs Python silently
echo Set WshShell = CreateObject("WScript.Shell") > "%VBS_PATH%"
echo WshShell.Run "%PYTHON_EXEC% %ELYSIA_SCRIPT% --stealth", 0, False >> "%VBS_PATH%"

echo.
echo [성공] 엘리시아 v1 시작프로그램 등록이 완료되었습니다!
echo 다음 부팅부터는 검은 터미널 창 없이 백그라운드에서 완전히 조용하게
echo 모니터와 하드웨어를 관측하며 10대 레이어를 돌리게 됩니다.
echo.
pause
