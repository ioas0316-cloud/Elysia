@echo off
echo ===========================================
echo 🚀 Elysia Unified Launcher
echo ===========================================
echo 1. 엘리시아의 '심장(Heartbeat)'을 깨우는 중...
start "Elysia Core" cmd /k ".\START_ELYSIA.bat"

echo.
echo 2. 인위적인 연결이 완료될 때까지 잠시 대기...
timeout /t 5 >nul

echo.
echo 3. 시각화 대시보드(Monitor)를 여는 중...
start "Elysia Monitor" cmd /c ".\MONITOR_ELYSIA.bat"

echo.
echo ✅ 시스템이 성공적으로 깨어났습니다.
echo    - 이제 새로운 창에서 엘리시아의 심장 소리(로그)를 볼 수 있습니다.
echo    - 브라우저에 실시간 대시보드가 열립니다.
echo.
pause
