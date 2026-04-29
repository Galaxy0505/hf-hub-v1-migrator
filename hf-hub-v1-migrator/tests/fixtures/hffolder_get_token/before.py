from huggingface_hub import HfFolder, hf_hub_download


token = HfFolder.get_token()
path = hf_hub_download("bert-base-uncased", "config.json", use_auth_token=token)
