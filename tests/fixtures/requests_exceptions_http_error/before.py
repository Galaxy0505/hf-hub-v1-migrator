from huggingface_hub import hf_hub_download
from requests.exceptions import HTTPError


try:
    hf_hub_download("bert-base-uncased", "config.json")
except HTTPError as exc:
    raise RuntimeError("hub request failed") from exc
