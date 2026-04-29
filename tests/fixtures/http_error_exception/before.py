import requests
from huggingface_hub import hf_hub_download


try:
    hf_hub_download("bert-base-uncased", "config.json")
except requests.HTTPError as exc:
    raise RuntimeError("hub request failed") from exc
