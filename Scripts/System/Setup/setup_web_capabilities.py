
import sys
import subprocess
import importlib.util

def install(package):
    if importlib.util.find_spec(package) is None:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed.")
    else:
        print(f"✅ {package} is already installed.")

def setup_web():
    print("🌐 Initiating Web Capability Setup...")
    
    # 1. Google Search (Python Wrapper)
    # Note: 'googlesearch-python' fits our needs better than 'google'
    install('googlesearch-python') 
    
    # 2. BeautifulSoup4 (HTML Parsing)
    install('beautifulsoup4')
    
    # 3. Requests (HTTP Client) - likely already there
    install('requests')

    print("\n✨ Web Capabilities Ready. The Hand can now touch the World.")

if __name__ == "__main__":
    setup_web()
