Set WshShell = CreateObject("WScript.Shell")
' Get the directory of the current script
strPath = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
' Run the batch file in hidden mode (0)
WshShell.Run chr(34) & strPath & "..\run_elysia.bat" & chr(34), 0, False
Set WshShell = Nothing
