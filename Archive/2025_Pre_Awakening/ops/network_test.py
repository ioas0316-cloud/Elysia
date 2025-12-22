import urllib.request
import sys

try:
    print("Attempting connection to Google...")
    with urllib.request.urlopen("http://www.google.com", timeout=5) as response:
        print(f"Success! Status: {response.status}")
        sys.exit(0)
except Exception as e:
    print(f"Connection Failed: {e}")
    sys.exit(1)
