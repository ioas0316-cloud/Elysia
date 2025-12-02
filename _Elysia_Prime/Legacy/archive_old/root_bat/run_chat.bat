@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Elysia Console Chat
chcp 65001 >nul

set "PYEXE="
where python >nul 2>&1 && set "PYEXE=python"
if not defined PYEXE (
  where py >nul 2>&1 && (py -3 -V >nul 2>&1) && set "PYEXE=py -3"
)
if not defined PYEXE (
  echo [Error] Python 3.10+ not found. Please install Python and retry.
  pause
  goto :eof
)

%PYEXE% -m scripts.console_chat
pause

