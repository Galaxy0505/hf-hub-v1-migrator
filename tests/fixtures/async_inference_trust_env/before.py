from huggingface_hub import AsyncInferenceClient


client = AsyncInferenceClient(model="gpt2", trust_env=True)
