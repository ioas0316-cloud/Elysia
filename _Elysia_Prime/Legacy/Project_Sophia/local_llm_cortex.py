# [Genesis: 2025-12-02] Purified by Elysia
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

class LocalLLMCortex:
    """
    Manages a local, quantized GGUF language model for inference.
    Handles model downloading, loading with GPU offloading, and text generation.
    """
    def __init__(self):
        self.model = None
        self.model_name = "TheBloke/gemma-2b-it-GGUF"
        self.model_file = "gemma-2b-it.Q4_K_M.gguf"
        self.n_gpu_layers = -1 # Offload all possible layers to GPU

        try:
            print("[LocalLLMCortex] Initializing...")
            model_path = self._download_model()

            print(f"[LocalLLMCortex] Loading model from: {model_path}")
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=2048, # Context window size
                verbose=True
            )
            print("[LocalLLMCortex] Model loaded successfully.")

        except Exception as e:
            print(f"[LocalLLMCortex] FATAL: Failed to initialize local LLM: {e}")
            print("[LocalLLMCortex] The 'llama-cpp-python' installation might be incomplete or corrupted.")
            print("[LocalLLMCortex] Please check the errors from the installation step.")
            self.model = None

    def _download_model(self):
        """
        Downloads the model from Hugging Face Hub if it doesn't exist locally.
        """
        # Define a local directory to store models
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"[LocalLLMCortex] Created models directory: {models_dir}")

        model_path = os.path.join(models_dir, self.model_file)

        if not os.path.exists(model_path):
            print(f"[LocalLLMCortex] Model not found locally. Downloading '{self.model_file}' from '{self.model_name}'...")
            hf_hub_download(
                repo_id=self.model_name,
                filename=self.model_file,
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )
            print("[LocalLLMCortex] Download complete.")
        else:
            print("[LocalLLMCortex] Model found locally.")

        return model_path

    def generate_response(self, prompt: str, max_tokens=150):
        """
        Generates a response from the local LLM.
        """
        if not self.model:
            return "Local model is not available."

        try:
            print(f"[LocalLLMCortex] Generating response for prompt: '{prompt[:50]}...'")

            # Create a chat prompt structure
            chat_prompt = f'''<start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
'''

            output = self.model(
                chat_prompt,
                max_tokens=max_tokens,
                echo=False,
                stop=["<end_of_turn>"]
            )

            response_text = output['choices'][0]['text'].strip()
            print(f"[LocalLLMCortex] Generated response: '{response_text[:50]}...'")
            return response_text

        except Exception as e:
            print(f"[LocalLLMCortex] Error during text generation: {e}")
            return "An error occurred while generating a response from the local model."