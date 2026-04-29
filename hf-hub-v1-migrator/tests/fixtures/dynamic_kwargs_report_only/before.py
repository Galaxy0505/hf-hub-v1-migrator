from huggingface_hub import hf_hub_download


hf_kwargs = {"use_auth_token": True, "resume_download": True}
path = hf_hub_download("bert-base-uncased", "config.json", **hf_kwargs)
