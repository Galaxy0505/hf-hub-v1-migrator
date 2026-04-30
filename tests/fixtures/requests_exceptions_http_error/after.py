from huggingface_hub import hf_hub_download, HfHubHttpError
from requests.exceptions import HTTPError


try:
    hf_hub_download("bert-base-uncased", "config.json")
except HfHubHttpError as exc:
    raise RuntimeError("hub request failed") from exc
