@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Elysia Launcher
chcp 65001 >nul

rem Detect Python (python or py -3)
set "PYEXE="
where python >nul 2>&1 && set "PYEXE=python"
if not defined PYEXE (
  where py >nul 2>&1 && (py -3 -V >nul 2>&1) && set "PYEXE=py -3"
)
if not defined PYEXE (
  echo [Error] Python 3.10+ not found. Install Python and retry.
  pause
  goto :eof
)

:MENU
cls
echo ==========================================
echo        Elysia Launcher (Simple Menu)
echo ==========================================
echo.
echo  1) Start Web Server (Dashboard)
echo  B) Start Clean Bridge (UTF-8)
echo  2) Run Daily Routine
echo  3) Generate Daily Report
echo  4) Run Textbook Demo
echo  5) Journaling
echo  6) Book Report
echo  7) Creative Writing
echo  8) Trinity Mission Demo
echo  9) Math Verification Demo
echo  V) Wisdom-Virus Demo
echo  S) Growth Sprint (ingest->keywords->virus->report)
rem (cleaned) duplicate menu removed
echo  R) Generate Sample Corpus (500)
echo  W) Schedule Growth Sprint (21:30 daily)
echo  X) Remove Growth Sprint schedule
echo  U) Start Background Learner (micro)
echo  I) Stop Background Learner
echo  Y) Use Learning Flow Profile
echo  G) Use Generic Flow Profile
echo  L) Train Literature Classifier
echo  C) Classify a Text File
echo  T) Debug Tokenization
echo  A) Autonomy Toggle
echo  Z) Quiet Mode Toggle
echo  P) Preset (quiet/balanced/lively)
echo  O) Open Outputs (folders)
echo  F) Quiet-All (stop background, set quiet)
echo  E) Resume-All (enable background, unset quiet)
echo  H) Help
echo  N) Nano Demo (link/verify/summarize)
echo  Q) Quit
echo.
set /P SEL=Select: 

if /I "%SEL%"=="1" goto START_SERVER
if /I "%SEL%"=="B" goto CLEAN_BRIDGE
if /I "%SEL%"=="2" goto DAILY_ROUTINE
if /I "%SEL%"=="3" goto DAILY_REPORT
if /I "%SEL%"=="4" goto TEXTBOOK
if /I "%SEL%"=="5" goto JOURNAL
if /I "%SEL%"=="6" goto BOOK
if /I "%SEL%"=="7" goto CREATIVE
if /I "%SEL%"=="8" goto TRINITY
if /I "%SEL%"=="9" goto MATH
if /I "%SEL%"=="V" goto VIRUS
if /I "%SEL%"=="S" goto GROWTH
if /I "%SEL%"=="R" goto GEN_CORPUS
if /I "%SEL%"=="W" goto SCHED_SPRINT
if /I "%SEL%"=="X" goto REMOVE_SPRINT
if /I "%SEL%"=="U" goto BG_START
if /I "%SEL%"=="I" goto BG_STOP
if /I "%SEL%"=="Y" goto FLOW_LEARNING
if /I "%SEL%"=="G" goto FLOW_GENERIC
if /I "%SEL%"=="L" goto LIT_TRAIN
if /I "%SEL%"=="C" goto LIT_CLASSIFY
if /I "%SEL%"=="T" goto TOKEN_DEBUG
if /I "%SEL%"=="A" goto AUTONOMY
if /I "%SEL%"=="Z" goto QUIET
if /I "%SEL%"=="P" goto PRESET
if /I "%SEL%"=="O" goto OPEN
if /I "%SEL%"=="F" goto QUIET_ALL
if /I "%SEL%"=="E" goto RESUME_ALL
if /I "%SEL%"=="H" goto HELP
if /I "%SEL%"=="N" goto NANO_DEMO
if /I "%SEL%"=="Q" goto END
goto MENU

:START_SERVER
echo [Startup] Launching Elysia web server in this window...
set FLASK_APP=applications\elysia_api.py
%PYEXE% -m flask run --host=0.0.0.0
goto MENU

:CLEAN_BRIDGE
echo [Startup] Launching Clean Bridge server in this window...
set FLASK_APP=applications\elysia_bridge_clean.py
%PYEXE% -m flask run --host=0.0.0.0
goto MENU

:DAILY_ROUTINE
echo [Routine] Run Daily Routine
set /P GENRE=  - Genre (default: story): 
if "%GENRE%"=="" set GENRE=story
set /P THEME=  - Theme (default: growth): 
if "%THEME%"=="" set THEME=growth
%PYEXE% -m scripts.run_daily_routine --genre %GENRE% --theme %THEME%
pause
goto MENU

:DAILY_REPORT
echo [Report] Generating today's daily report...
%PYEXE% -m scripts.run_daily_report
pause
goto MENU

:TEXTBOOK
echo [Textbook] Choose a sample book:
echo   1) data\textbooks\nlp_basics.json
echo   2) data\textbooks\math_basics.json
echo   3) Enter a custom path
set /P CH=Select [1/2/3]: 
if "%CH%"=="1" set BOOK=data\textbooks\nlp_basics.json
if "%CH%"=="2" set BOOK=data\textbooks\math_basics.json
if "%CH%"=="3" set /P BOOK=Enter path to textbook JSON: 
if not defined BOOK set BOOK=data\textbooks\nlp_basics.json
%PYEXE% -m scripts.run_textbook_demo --book "%BOOK%"
pause
goto MENU

:JOURNAL
echo [Journal] Writing today's entry...
%PYEXE% -m scripts.run_journaling_lesson
pause
goto MENU

:BOOK
echo [Book] Create a report from a local .txt file
set /P BOOKPATH= - Enter path to book (.txt): 
if not exist "%BOOKPATH%" (
  echo   [Error] File not found: %BOOKPATH%
) else (
  %PYEXE% -m scripts.run_book_report --book "%BOOKPATH%"
)
pause
goto MENU

:CREATIVE
echo [Creative] Generate an outline and scenes
set /P GENRE=  - Genre (e.g., fantasy) [story]: 
if "%GENRE%"=="" set GENRE=story
set /P THEME=  - Theme (e.g., hope) [growth]: 
if "%THEME%"=="" set THEME=growth
set /P BEATS=  - Beats count [5]: 
if "%BEATS%"=="" set BEATS=5
set /P WORDS=  - Approx words per scene [120]: 
if "%WORDS%"=="" set WORDS=120
%PYEXE% -m scripts.run_creative_writing --genre %GENRE% --theme %THEME% --beats %BEATS% --words %WORDS%
pause
goto MENU

:TRINITY
echo [Trinity] Running E2E demo...
%PYEXE% -m scripts.trinity_mission_demo
pause
goto MENU

:MATH
echo [Math] Verify an equality
set /P STMT= - Enter statement (e.g., 3*(2+4)=18): 
if "%STMT%"=="" set STMT=3*(2+4)=18
%PYEXE% -m scripts.pipeline_math_verify_demo "%STMT%"
pause
goto MENU

:AUTONOMY
echo [Autonomy] Toggle self-acting mode.
set /P ANS=Turn ON or OFF [on/off]: 
if /I "%ANS%"=="ON"  %PYEXE% -m scripts.toggle_autonomy --on
if /I "%ANS%"=="OFF" %PYEXE% -m scripts.toggle_autonomy --off
pause
goto MENU

:QUIET
echo [Quiet] Toggle quiet mode (suppress self-proposals)
set /P ANS=Turn ON or OFF [on/off]: 
if /I "%ANS%"=="ON"  %PYEXE% -m scripts.toggle_quiet_mode --on
if /I "%ANS%"=="OFF" %PYEXE% -m scripts.toggle_quiet_mode --off
pause
goto MENU

:PRESET
echo [Preset] Choose autonomy preset
echo   1) quiet
echo   2) balanced
echo   3) lively
set /P PR=Select [1/2/3]: 
if "%PR%"=="1" set PRE=quiet
if "%PR%"=="2" set PRE=balanced
if "%PR%"=="3" set PRE=lively
if not defined PRE set PRE=quiet
%PYEXE% -m scripts.set_autonomy_preset --preset %PRE%
pause
goto MENU

:OPEN
:NANO_DEMO
echo [Nano] Running nano-core demo...
%PYEXE% -m scripts.nano_demo --extra
pause
goto MENU
echo [Open] Choose a folder to open
echo   1) data\journal
echo   2) data\reports\daily
echo   3) data\writings
echo   4) data\proofs
echo   5) data (root)
set /P OP=Select [1..5]: 
if "%OP%"=="1" start "" "%CD%\data\journal"
if "%OP%"=="2" start "" "%CD%\data\reports\daily"
if "%OP%"=="3" start "" "%CD%\data\writings"
if "%OP%"=="4" start "" "%CD%\data\proofs"
if "%OP%"=="5" start "" "%CD%\data"
pause
goto MENU

:QUIET_ALL
echo [Quiet-All] Stopping background and enabling quiet mode...
%PYEXE% -m scripts.quiet_all
pause
goto MENU

:RESUME_ALL
echo [Resume-All] Enabling background and disabling quiet mode...
%PYEXE% -m scripts.resume_all
pause
goto MENU

:HELP
cls
echo =================== HELP ====================
echo 1) Start Web Server  - Flask dashboard http://127.0.0.1:5000/monitor
echo B) Start Clean Bridge - UTF-8 chat http://127.0.0.1:5000
echo 2) Daily Routine     - Journal + Creative (data\journal, data\writings)
echo 3) Daily Report      - MD/PNG (data\reports\daily)
echo 4) Textbook Demo     - data\textbooks samples
echo 5) Journaling        - Today's entry under data\journal
echo 6) Book Report       - From .txt to *_report.md (data\reports)
echo 7) Creative Writing  - Outline + scenes (data\writings)
echo 8) Trinity Demo      - Files/Proof/Image/KG demo
echo 9) Math Verify       - Proof images under data\proofs
echo V) Wisdom-Virus Demo - Propagate a meaning unit in KG
pause
goto MENU

:VIRUS
echo [Virus] Running Wisdom-Virus demo...
%PYEXE% -m scripts.run_virus_demo
pause
goto MENU

:GROWTH
echo [Growth] Running Growth Sprint...
%PYEXE% -m scripts.growth_sprint --ingest --keywords --virus --report
pause
goto MENU

:GEN_CORPUS
echo [Corpus] Generating 500 sample texts under data\corpus\literature...
%PYEXE% -m scripts.generate_corpus --count 500
echo Done. You can now run Growth Sprint (S) to ingest/link/propagate.
pause
goto MENU

:SCHED_SPRINT
echo [Scheduler] Create daily Growth Sprint at 21:30
scripts\setup_growth_sprint.bat
goto MENU

:REMOVE_SPRINT
echo [Scheduler] Remove daily Growth Sprint
scripts\remove_growth_sprint.bat
goto MENU

:BG_START
echo [Background] Starting micro-learner in a new window (Ctrl+C to stop that window)
start "Elysia Background" %PYEXE% -m scripts.background_daemon
%PYEXE% -m scripts.toggle_background --on --interval 900
pause
goto MENU

:BG_STOP
echo [Background] Stopping background learner
%PYEXE% -m scripts.toggle_background --off
pause
goto MENU

:FLOW_LEARNING
echo [Flow] Switching to learning profile...
%PYEXE% -m scripts.set_flow_profile --profile learning
echo Done. Next responses will favor clarify/small-steps.
pause
goto MENU

:FLOW_GENERIC
echo [Flow] Switching to generic profile...
%PYEXE% -m scripts.set_flow_profile --profile generic
echo Done. Back to balanced dialog flow.
pause
goto MENU

:LIT_TRAIN
echo [Literature] Train Naive Bayes classifier from data\corpus\literature\<label>\*.txt
%PYEXE% -m scripts.train_text_classifier --data data\corpus\literature --out data\models\lit_nb.json
pause
goto MENU

:LIT_CLASSIFY
echo [Literature] Classify a .txt file using data\models\lit_nb.json
set /P FPATH= - Enter path to .txt file: 
if not exist "%FPATH%" (
  echo   [Error] File not found: %FPATH%
) else (
  %PYEXE% -m scripts.classify_text --model data\models\lit_nb.json --file "%FPATH%"
)
pause
goto MENU

:TOKEN_DEBUG
echo [Tokenize] Enter a line to segment (Korean-aware, josa/eomi split)
set /P LINE= - Text: 
%PYEXE% -m scripts.debug_tokenize --text "%LINE%"
pause
goto MENU

:END
endlocal
echo Bye.
exit /b 0


