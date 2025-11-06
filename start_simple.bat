@ECHO OFF
SETLOCAL ENABLEDELAYEDEXPANSION
TITLE Elysia Quick Launcher
CHCP 437 >NUL

REM Detect Python (python or py -3)
SET "PYEXE="
WHERE python >NUL 2>&1 && SET "PYEXE=python"
IF NOT DEFINED PYEXE (
  WHERE py >NUL 2>&1 && (py -3 -V >NUL 2>&1) && SET "PYEXE=py -3"
)
IF NOT DEFINED PYEXE (
  ECHO Python not found. Install Python 3.10+.
  PAUSE
  GOTO :EOF
)

:MENU
CLS
ECHO ======================================
ECHO Elysia Quick Launcher
ECHO ======================================
ECHO 1. Journaling
ECHO 2. Daily Report
ECHO 3. Daily Routine
ECHO 4. Creative Writing (defaults)
ECHO 5. Open data folder
ECHO 6. Textbook Demo (sample)
ECHO 7. Math Verification
ECHO 8. Autonomy Toggle (on/off)
ECHO 9. Quiet Mode Toggle (on/off)
ECHO 0. Preset (quiet/balanced/lively)
ECHO W. Start Web Server
ECHO I. Ingest Corpus (dialog/story)
ECHO Q. Quit
ECHO.
SET /P CH=Select: 
IF /I "%CH%"=="1" GOTO J
IF /I "%CH%"=="2" GOTO DR
IF /I "%CH%"=="3" GOTO ROUTINE
IF /I "%CH%"=="4" GOTO CW
IF /I "%CH%"=="5" GOTO OPEN
IF /I "%CH%"=="6" GOTO TBOOK
IF /I "%CH%"=="7" GOTO MV
IF /I "%CH%"=="8" GOTO AUTO
IF /I "%CH%"=="9" GOTO QUIET
IF /I "%CH%"=="0" GOTO PRESET
IF /I "%CH%"=="W" GOTO WEB
IF /I "%CH%"=="I" GOTO INGEST
IF /I "%CH%"=="Q" GOTO END
GOTO MENU

:J
%PYEXE% -m scripts.run_journaling_lesson
PAUSE
GOTO MENU

:DR
%PYEXE% -m scripts.run_daily_report
PAUSE
GOTO MENU

:ROUTINE
%PYEXE% -m scripts.run_daily_routine --genre story --theme growth
PAUSE
GOTO MENU

:CW
%PYEXE% -m scripts.run_creative_writing --genre story --theme growth --beats 4 --words 80
PAUSE
GOTO MENU

:OPEN
START "" "%CD%\data"
PAUSE
GOTO MENU

:TBOOK
ECHO 1) data\textbooks\nlp_basics.json
ECHO 2) data\textbooks\math_basics.json
ECHO 3) Custom path
SET /P TB=Select [1/2/3]: 
IF "%TB%"=="1" SET BOOK=data\textbooks\nlp_basics.json
IF "%TB%"=="2" SET BOOK=data\textbooks\math_basics.json
IF "%TB%"=="3" SET /P BOOK=Enter path: 
IF NOT DEFINED BOOK SET BOOK=data\textbooks\nlp_basics.json
%PYEXE% -m scripts.run_textbook_demo --book "%BOOK%"
PAUSE
GOTO MENU

:MV
SET /P STMT=Enter equality (e.g., 3*(2+4)=18): 
IF "%STMT%"=="" SET STMT=3*(2+4)=18
%PYEXE% -m scripts.pipeline_math_verify_demo "%STMT%"
PAUSE
GOTO MENU

:AUTO
SET /P ANS=Turn autonomy ON or OFF [on/off]: 
IF /I "%ANS%"=="ON"  %PYEXE% -m scripts.toggle_autonomy --on
IF /I "%ANS%"=="OFF" %PYEXE% -m scripts.toggle_autonomy --off
PAUSE
GOTO MENU

:QUIET
SET /P ANS=Turn quiet mode ON or OFF [on/off]: 
IF /I "%ANS%"=="ON"  %PYEXE% -m scripts.toggle_quiet_mode --on
IF /I "%ANS%"=="OFF" %PYEXE% -m scripts.toggle_quiet_mode --off
PAUSE
GOTO MENU

:PRESET
ECHO 1) quiet   2) balanced   3) lively
SET /P PR=Select preset [1/2/3]: 
IF "%PR%"=="1" SET PRE=quiet
IF "%PR%"=="2" SET PRE=balanced
IF "%PR%"=="3" SET PRE=lively
IF NOT DEFINED PRE SET PRE=quiet
%PYEXE% -m scripts.set_autonomy_preset --preset %PRE%
PAUSE
GOTO MENU

:WEB
SET FLASK_APP=applications/elysia_api.py
START "Elysia Web" %PYEXE% -m flask run --host=0.0.0.0
START http://127.0.0.1:5000/monitor
PAUSE
GOTO MENU

:INGEST
ECHO 1) Ingest dialogue file  2) Ingest story file
SET /P IG=Select [1/2]: 
IF "%IG%"=="1" (
  SET /P PTH=Enter dialogue txt path [data\corpus\dialogues\sample_dialog.txt]: 
  IF "%PTH%"=="" SET PTH=data\corpus\dialogues\sample_dialog.txt
  %PYEXE% -m scripts.ingest_dialog_corpus --path "%PTH%"
) ELSE IF "%IG%"=="2" (
  SET /P PTH=Enter story txt path [data\corpus\stories\sample_story.txt]: 
  IF "%PTH%"=="" SET PTH=data\corpus\stories\sample_story.txt
  %PYEXE% -m scripts.ingest_stories --path "%PTH%"
) ELSE (
  ECHO Invalid selection.
)
PAUSE
GOTO MENU

:END
ECHO Bye.
ENDLOCAL
EXIT /B 0
