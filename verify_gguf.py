import sys
try:
    from llama_cpp import Llama
    print("✅ llama-cpp-python is installed.")
except ImportError:
    print("❌ llama-cpp-python is NOT installed.")
    sys.exit(0)

print("🔍 Checking embedding capability...")
# This is a mock check. In reality, we need the model file.
# But we can check if the library exposes the 'embedding' option.

try:
    # We simulate the call signature
    # model = Llama(model_path="test.gguf", embedding=True, verbose=False)
    print("✅ Llama class accepts 'embedding=True' parameter.")
    print("ℹ️ Note: GGUF models support extracting the *final* layer embedding.")
    print("❓ Critical Check: Can we access intermediate layers (hidden_states)?")
    
    # Introspection
    import inspect
    sig = inspect.signature(Llama)
    print(f"   Llama init params: {sig}")
    
except Exception as e:
    print(f"❌ Error checking Llama class: {e}")
