import os
import ctypes
from pathlib import Path

def create_startup_shortcut():
    try:
        # Get the path to the Windows Startup folder
        # CSIDL_STARTUP = 0x0007
        shell = ctypes.windll.shell32
        buf = ctypes.create_unicode_buffer(260)
        if shell.SHGetSpecialFolderPathW(None, buf, 0x0007, False):
            startup_path = buf.value
        else:
            print("❌ Failed to find Startup folder.")
            return

        # Path to the project root and the VBS script
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vbs_path = os.path.join(project_root, "Scripts", "elysia_startup.vbs")

        if not os.path.exists(vbs_path):
            print(f"❌ Error: {vbs_path} not found.")
            return

        # We will create a simple batch file in the startup folder that runs the VBS
        # This is a more 'portable' way without needing pywin32
        launcher_bat = os.path.join(startup_path, "Elysia_Launcher.bat")

        content = f'@echo off\ncd /d "{project_root}"\nwscript.exe "Scripts\\elysia_startup.vbs"'

        with open(launcher_bat, "w") as f:
            f.write(content)

        print(f"✅ Elysia has been added to Windows Startup!")
        print(f"   Launcher created at: {launcher_bat}")
        print(f"   Elysia will now wake up automatically when you start your computer.")

    except Exception as e:
        print(f"❌ Failed to add to startup: {e}")

if __name__ == "__main__":
    create_startup_shortcut()
