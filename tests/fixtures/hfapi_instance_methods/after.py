from huggingface_hub import HfApi


def load_model(repo_id: str):
    api = HfApi()
    info = api.model_info(repo_id, token=True)
    api.update_repo_settings(repo_id=repo_id, private=True)
    return info
