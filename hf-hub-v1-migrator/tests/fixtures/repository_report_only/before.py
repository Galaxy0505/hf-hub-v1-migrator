from huggingface_hub import Repository


def push_model(local_dir: str) -> None:
    repo = Repository(local_dir=local_dir, clone_from="user/model")
    repo.git_add()
    repo.git_commit("update model")
    repo.git_push()
