Set WshShell = CreateObject("WScript.Shell")
' Run the python script silently (0 means hidden window)
WshShell.CurrentDirectory = "c:\Elysia"
WshShell.Run "python Scripts/elysia_pulse.py", 0, False
WshShell.Run "python elysia.py", 0, False
