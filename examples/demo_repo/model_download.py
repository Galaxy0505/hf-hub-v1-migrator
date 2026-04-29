import requests
from huggingface_hub import HfFolder, InferenceApi, hf_hub_download


def download_config(repo_id: str):
    token = HfFolder.get_token()
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            use_auth_token=token,
            resume_download=True,
        )
    except requests.HTTPError as exc:
        raise RuntimeError("Hugging Face download failed") from exc


def run_inference(repo_id: str):
    client = InferenceApi(repo_id=repo_id)
    return client(inputs="hello")
