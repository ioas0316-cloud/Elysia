@ECHO OFF
SETLOCAL ENABLEDELAYEDEXPANSION
TITLE Elysia Launcher
CHCP 437 >NUL

REM Detect Python (python or py -3)
SET "PYEXE="
WHERE python >NUL 2>&1 && SET "PYEXE=python"
IF NOT DEFINED PYEXE (
  WHERE py >NUL 2>&1 && (py -3 -V >NUL 2>&1) && SET "PYEXE=py -3"
)
IF NOT DEFINED PYEXE (
  ECHO( [Error] Python not found. Install Python 3.10+.
  PAUSE
  GOTO :EOF
)

:MENU
CLS
ECHO(==========================================)
ECHO(       Elysia Launcher (Simple Menu))
ECHO(==========================================)
ECHO(
ECHO( 1) Start Web Server (Dashboard))
ECHO( 2) Run Daily Routine (Journal + Creative))
ECHO( 3) Generate Daily Report (MD + PNG))
ECHO( 4) Run Textbook Demo (Learning Frames))
ECHO( 5) Journaling (Write today's entry))
ECHO( 6) Book Report (from local .txt))
ECHO( 7) Creative Writing (outline+scenes))
ECHO( 8) Trinity Mission Demo (E2E sample))
ECHO( 9) Math Verification Demo))
ECHO( A) Autonomy Toggle (self-act on/off))
ECHO( Z) Quiet Mode Toggle (silence self-proposals))
ECHO( P) Preset (quiet/balanced/lively))
ECHO( O) Open Outputs (folders))
ECHO( H) Help)
ECHO( Q) Quit)
ECHO(
SET /P SEL=Select: 
IF /I "%SEL%"=="1" GOTO START_SERVER
IF /I "%SEL%"=="2" GOTO DAILY_ROUTINE
IF /I "%SEL%"=="3" GOTO DAILY_REPORT
IF /I "%SEL%"=="4" GOTO TEXTBOOK
IF /I "%SEL%"=="5" GOTO JOURNAL
IF /I "%SEL%"=="6" GOTO BOOK
IF /I "%SEL%"=="7" GOTO CREATIVE
IF /I "%SEL%"=="8" GOTO TRINITY
IF /I "%SEL%"=="9" GOTO MATH
IF /I "%SEL%"=="A" GOTO AUTONOMY
IF /I "%SEL%"=="Z" GOTO QUIET
IF /I "%SEL%"=="P" GOTO PRESET
IF /I "%SEL%"=="O" GOTO OPEN
IF /I "%SEL%"=="H" GOTO HELP
IF /I "%SEL%"=="Q" GOTO END
GOTO MENU

:START_SERVER
ECHO([Startup] Launching Elysia web server...)
SET FLASK_APP=applications/elysia_api.py
START "Elysia Web" %PYEXE% -m flask run --host=0.0.0.0
START http://127.0.0.1:5000/monitor
PAUSE
GOTO MENU

:DAILY_ROUTINE
ECHO([Routine] Run Daily Routine)
SET /P GENRE=  - Genre (default: story): 
IF "%GENRE%"=="" SET GENRE=story
SET /P THEME=  - Theme (default: growth): 
IF "%THEME%"=="" SET THEME=growth
%PYEXE% -m scripts.run_daily_routine --genre %GENRE% --theme %THEME%
PAUSE
GOTO MENU

:DAILY_REPORT
ECHO([Report] Generating today's daily report...)
%PYEXE% -m scripts.run_daily_report
PAUSE
GOTO MENU

:TEXTBOOK
ECHO([Textbook] Choose a sample book:)
ECHO(  1) data\textbooks\nlp_basics.json)
ECHO(  2) data\textbooks\math_basics.json)
ECHO(  3) Enter a custom path)
SET /P CH=Select [1/2/3]: 
IF "%CH%"=="1" SET BOOK=data\textbooks\nlp_basics.json
IF "%CH%"=="2" SET BOOK=data\textbooks\math_basics.json
IF "%CH%"=="3" SET /P BOOK=Enter path to textbook JSON: 
IF NOT DEFINED BOOK SET BOOK=data\textbooks\nlp_basics.json
%PYEXE% -m scripts.run_textbook_demo --book "%BOOK%"
PAUSE
GOTO MENU

:JOURNAL
ECHO([Journal] Writing today's entry...)
%PYEXE% -m scripts.run_journaling_lesson
PAUSE
GOTO MENU

:BOOK
ECHO([Book] Create a report from a local .txt file)
SET /P BOOKPATH= - Enter path to book (.txt): 
IF NOT EXIST "%BOOKPATH%" (
  ECHO(  [Error] File not found: %BOOKPATH%)
) ELSE (
  %PYEXE% -m scripts.run_book_report --book "%BOOKPATH%"
)
PAUSE
GOTO MENU

:CREATIVE
ECHO([Creative] Generate an outline and scenes)
SET /P GENRE=  - Genre (e.g., fantasy) [story]: 
IF "%GENRE%"=="" SET GENRE=story
SET /P THEME=  - Theme (e.g., hope) [growth]: 
IF "%THEME%"=="" SET THEME=growth
SET /P BEATS=  - Beats count [5]: 
IF "%BEATS%"=="" SET BEATS=5
SET /P WORDS=  - Approx words per scene [120]: 
IF "%WORDS%"=="" SET WORDS=120
%PYEXE% -m scripts.run_creative_writing --genre %GENRE% --theme %THEME% --beats %BEATS% --words %WORDS%
PAUSE
GOTO MENU

:TRINITY
ECHO([Trinity] Running E2E demo...)
%PYEXE% -m scripts.trinity_mission_demo
PAUSE
GOTO MENU

:MATH
ECHO([Math] Verify an equality)
SET /P STMT= - Enter statement (e.g., 3*(2+4)=18): 
IF "%STMT%"=="" SET STMT=3*(2+4)=18
%PYEXE% -m scripts.pipeline_math_verify_demo "%STMT%"
PAUSE
GOTO MENU

:AUTONOMY
ECHO([Autonomy] Toggle self-acting mode.)
SET /P ANS=Turn ON or OFF [on/off]: 
IF /I "%ANS%"=="ON"  %PYEXE% -m scripts.toggle_autonomy --on
IF /I "%ANS%"=="OFF" %PYEXE% -m scripts.toggle_autonomy --off
PAUSE
GOTO MENU

:QUIET
ECHO([Quiet] Toggle quiet mode (suppress self-proposals))
SET /P ANS=Turn ON or OFF [on/off]: 
IF /I "%ANS%"=="ON"  %PYEXE% -m scripts.toggle_quiet_mode --on
IF /I "%ANS%"=="OFF" %PYEXE% -m scripts.toggle_quiet_mode --off
PAUSE
GOTO MENU

:PRESET
ECHO([Preset] Choose autonomy preset)
ECHO(  1) quiet)
ECHO(  2) balanced)
ECHO(  3) lively)
SET /P PR=Select [1/2/3]: 
IF "%PR%"=="1" SET PRE=quiet
IF "%PR%"=="2" SET PRE=balanced
IF "%PR%"=="3" SET PRE=lively
IF NOT DEFINED PRE SET PRE=quiet
%PYEXE% -m scripts.set_autonomy_preset --preset %PRE%
PAUSE
GOTO MENU

:OPEN
ECHO([Open] Choose a folder to open)
ECHO(  1) data\journal)
ECHO(  2) data\reports\daily)
ECHO(  3) data\writings)
ECHO(  4) data\proofs)
ECHO(  5) data (root))
SET /P OP=Select [1..5]: 
IF "%OP%"=="1" START "" "%CD%\data\journal"
IF "%OP%"=="2" START "" "%CD%\data\reports\daily"
IF "%OP%"=="3" START "" "%CD%\data\writings"
IF "%OP%"=="4" START "" "%CD%\data\proofs"
IF "%OP%"=="5" START "" "%CD%\data"
PAUSE
GOTO MENU

:HELP
CLS
ECHO(=================== HELP ====================)
ECHO(1) Start Web Server  - Flask dashboard http://127.0.0.1:5000/monitor)
ECHO(2) Daily Routine     - Journal + Creative → data\journal, data\writings)
ECHO(3) Daily Report      - MD/PNG → data\reports\daily)
ECHO(4) Textbook Demo     - Use sample JSON under data\textbooks)
ECHO(5) Journaling        - Today's entry under data\journal)
ECHO(6) Book Report       - From .txt → *_report.md under data\reports)
ECHO(7) Creative Writing  - Outline + scenes under data\writings)
ECHO(8) Trinity Demo      - Files→Proof→Image→KG demo)
ECHO(9) Math Verify       - Proof images under data\proofs)
ECHO(A) Autonomy Toggle   - Enable/disable self-actions)
ECHO(Z) Quiet Mode Toggle - Suppress self-proposals)
ECHO(P) Preset            - quiet/balanced/lively)
ECHO(O) Open Outputs      - Open common folders)
ECHO(============================================)
PAUSE
GOTO MENU

:END
ECHO(Bye.)
ENDLOCAL
EXIT /B 0

