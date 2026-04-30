from huggingface_hub import hf_hub_download, get_token


token = get_token()
path = hf_hub_download("bert-base-uncased", "config.json", token=token)
