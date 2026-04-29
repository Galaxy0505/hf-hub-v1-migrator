import requests
from huggingface_hub import hf_hub_download


try:
    requests.get("https://example.com")
except requests.HTTPError as exc:
    raise RuntimeError("non-hub request failed") from exc

path = hf_hub_download("bert-base-uncased", "config.json")
