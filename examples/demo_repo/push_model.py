from huggingface_hub import Repository


def push_model(local_dir: str):
    repo = Repository(local_dir=local_dir, clone_from="Galaxy0505/demo-model")
    repo.git_add()
    repo.git_commit("update model")
    repo.git_push()
