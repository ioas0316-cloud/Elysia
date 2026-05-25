@echo off
echo ==============================================================
echo  🌌 ELYSIA AUTONOMIC DAEMON AWAKENING
echo ==============================================================
echo [1/3] Starting Substation Gateway (API)...
start /b "Elysia Substation" cmd /c "cd /d C:\Elysia && set PYTHONPATH=C:\Elysia && python core\substation_gateway.py"

timeout /t 3 /nobreak > nul

echo [2/3] Starting Triple Helix Core Daemon...
start /b "Elysia Core Daemon" cmd /c "cd /d C:\Elysia && set PYTHONPATH=C:\Elysia && python scripts\elysia_daemon.py"

timeout /t 2 /nobreak > nul

echo [3/3] Connecting Cortex Vocal Cords (Inverse Decoder)...
start /b "Elysia Cortex Decoder" cmd /c "cd /d C:\elysia_cortex && set PYTHONPATH=C:\elysia_cortex && python inverse_decoder.py"

echo.
echo 🌅 [System] All neuro-pathways are connected.
echo 🌅 [System] Elysia is now continuously breathing in the background.
echo ==============================================================
exit
