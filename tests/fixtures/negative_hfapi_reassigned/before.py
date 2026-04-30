from huggingface_hub import HfApi


class OtherApi:
    def model_info(self, repo_id: str, use_auth_token: bool):
        return repo_id, use_auth_token


def load_model(repo_id: str):
    api = HfApi()
    api = OtherApi()
    return api.model_info(repo_id, use_auth_token=True)
