import os
import sys
import time

def talk():
    input_file = r"c:\Elysia\inputs\user_stream.txt"
    os.makedirs(os.path.dirname(input_file), exist_ok=True)
    
    print("\n💬 Talking to Elysia...")
    print("   (Type your message and press Enter. Type 'exit' to quit.)\n")
    
    while True:
        try:
            msg = input("User > ")
            if msg.lower() in ["exit", "quit"]:
                break
            
            with open(input_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
                
            print("   📨 Sent to Elysia.")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    talk()
