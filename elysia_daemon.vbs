Set WshShell = CreateObject("WScript.Shell")
' Run the python script silently (0 means hidden window)
WshShell.CurrentDirectory = "c:\Elysia"
WshShell.Run "python elysia.py", 0, False
