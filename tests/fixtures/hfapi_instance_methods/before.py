from huggingface_hub import HfApi


def load_model(repo_id: str):
    api = HfApi()
    info = api.model_info(repo_id, use_auth_token=True)
    api.update_repo_visibility(repo_id=repo_id, private=True)
    return info
